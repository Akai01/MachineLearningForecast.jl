"""
    BacktestResult

Result of [`backtest`](@ref). Both fields are columntables (NamedTuples of
vectors, Tables.jl-compatible):

- `forecasts`: one row per fold × horizon step, with columns `:fold`, `:origin`
  (the last training timestamp the row's forecast starts from), `:step`
  (1..horizon), the spec's time column, `:y` (actual), and `:y_hat` (forecast).
  A panel backtest adds the id column after `:step`. With `levels`, the columns
  `lo_<L>` and `hi_<L>` follow `:y_hat`.
- `scores`: one row per fold × metric (columns `:fold`, `:origin`, `:metric`,
  `:value`). A fold's `:origin` is `missing` when its series start their
  forecasts at different timestamps. [`summarize`](@ref) reduces it to the mean
  and standard deviation of each metric over the folds.

# Example
```julia
results = backtest(fc, df; horizon=28, initial=730)
results.forecasts.y_hat   # every forecast, fold by fold
results.scores            # the score of each fold and metric
summarize(results)        # mean and std of each metric over folds
```
"""
struct BacktestResult{F<:NamedTuple,S<:NamedTuple}
    forecasts::F
    scores::S
end

"""
    summarize(r::BacktestResult) -> NamedTuple

Summarize the per-fold scores of a [`backtest`](@ref): one row per metric, in the
order the metrics were given, with columns `:metric` (`Symbol`), `:mean` (the
unweighted mean of the metric over folds), `:std` (the sample standard deviation
over folds, with divisor `n - 1`) and `:nfolds` (`Int`). `:mean` and `:std` are
`Float64`. With a single fold `:std` is `NaN`. A `NaN` or `Inf` score in any fold
makes `:mean` (and `:std`) non-finite. With `metrics=()` every column is empty.

`std` is the spread of the per-fold scores. Expanding-window folds share their
training data and, when `step < horizon`, their test rows, so it describes the
variability across origins and is not a standard error of `mean`.

# Example
```julia
r = backtest(fc, df; horizon=28, initial=730, metrics=(mae, rmse))
summarize(r).mean   # [mae, rmse] averaged over folds
```
"""
function summarize(r::BacktestResult)
    sc = r.scores
    metric = unique(sc.metric)
    means = Float64[]
    stds = Float64[]
    nfolds = Int[]
    for name in metric
        vals = sc.value[sc.metric .== name]
        push!(means, Statistics.mean(vals))
        push!(stds, Statistics.std(vals))
        push!(nfolds, length(vals))
    end
    return (metric=metric, mean=means, std=stds, nfolds=nfolds)
end

"""
    score(kind, m, fold) -> Real

Score metric `m` of input `kind` (see [`metric_inputs`](@ref)) on one backtest fold.
`fold` holds `fc`, the forecaster, `history`, the fold's training target (a vector
for a single series, a `Dict` by series id for a panel), `ids` (a panel only, the
series of each scored row), `y`, `ŷ` and `S`, the forecast samples (a matrix with one row
per scored row, `nothing` unless the backtest has `levels` or a metric of kind
[`Samples`](@ref)).
"""
score(::Point, m, fold::NamedTuple) = m(fold.y, fold.ŷ)
score(::Scaled, m, fold::NamedTuple) = _score_scaled(fold.fc, m, fold)
score(::Samples, m, fold::NamedTuple) = m(fold.y, fold.S)

_score_scaled(::Forecaster{M,S,Nothing}, m, fold) where {M,S<:ForecastStrategy} =
    m(fold.y, fold.ŷ; y_train=fold.history)

# First-appearance order keeps the float sum reproducible.
function _score_scaled(fc::Forecaster, m, fold)
    ids = fold.ids
    rows = Dict{eltype(ids),Vector{Int}}()
    for (j, id) in enumerate(ids)
        push!(get!(rows, id, Int[]), j)
    end
    total = 0.0
    for id in unique(ids)
        r = rows[id]
        v = _in_series(fc, id) do
            m(fold.y[r], fold.ŷ[r]; y_train=fold.history[id])
        end
        total += length(r) * v
    end
    return total / length(fold.y)
end

"Where backtest folds start: from the data's start, or from each series' end."
abstract type FoldScheme end

"Folds growing from the start: the first trains on `initial` timestamps."
struct FromStart <: FoldScheme
    initial::Int
    step::Int
end

"The last `windows` folds of each series, counted back from its own end."
struct FromEnd <: FoldScheme
    windows::Int
    step::Int
end

function _fold_scheme(who::AbstractString, initial, windows, step::Integer)
    (initial === nothing) == (windows === nothing) && throw(ArgumentError(
        "$who needs exactly one of initial (folds growing from the start of the " *
        "data) and windows (folds counted back from each series' end), got " *
        "initial=$(repr(initial)) and windows=$(repr(windows)). Pass e.g. " *
        "initial=730 or windows=3."))
    if initial !== nothing
        _nobool(initial, "$who initial", "Pass the size of the first training " *
                "window, e.g. initial=730.")
        initial ≥ 1 || throw(ArgumentError(
            "$who initial must be ≥ 1, got $initial. Pass the size of the first " *
            "training window, e.g. initial=730."))
        return FromStart(initial, step)
    end
    _nobool(windows, "$who windows", "Pass the number of folds to hold out at the " *
            "end of each series, e.g. windows=3.")
    windows ≥ 1 || throw(ArgumentError(
        "$who windows must be ≥ 1, got $windows. Pass the number of folds to hold " *
        "out at the end of each series, e.g. windows=3."))
    return FromEnd(windows, step)
end

_scheme_kwargs(s::FromStart) = (initial=s.initial, step=s.step)
_scheme_kwargs(s::FromEnd) = (windows=s.windows, step=s.step)

function _origins(s::FromStart, n, horizon, fc)
    mh = minhistory(fc)
    s.initial > mh || throw(ArgumentError(
        "backtest initial=$(s.initial) must exceed $(_history_name(fc)) " *
        "($mh rows) so the first training window has at least one usable row. " *
        "Pass initial=$(mh + 1) or more, or reduce $(_reduce_hint(fc))."))
    origins = s.initial:s.step:(n - horizon)
    isempty(origins) && throw(ArgumentError(
        "no complete backtest folds: data has $n rows, but the first fold needs " *
        "initial + horizon = $(s.initial + horizon). Provide more data or reduce " *
        "initial/horizon."))
    return origins
end

function _origins(s::FromEnd, n, horizon, fc)
    mh = minhistory(fc)
    first = n - horizon - (s.windows - 1) * s.step
    first > mh || throw(ArgumentError(
        "backtest windows=$(s.windows) with step=$(s.step) and horizon=$horizon " *
        "would train its first fold on $first of the $n rows, but $(_history_who(fc)) " *
        "more than $mh. Reduce windows, step or horizon$(_transform_hint(fc)), or " *
        "provide more data."))
    return first:s.step:(n - horizon)
end

"""
    backtest(fc::Forecaster, data; horizon, initial=nothing, windows=nothing,
             step=horizon, metrics=(mae, rmse), levels=nothing) -> BacktestResult

Expanding-window time-series cross-validation. Pass exactly one of:

- `initial`: train on rows `1:initial`, forecast `horizon` steps, evaluate
  against the held-out actuals; advance the origin by `step` rows and repeat
  while a full horizon of actuals remains.
- `windows`: hold out the last `windows` folds, `step` apart, the last ending
  at the end of the data. For a single series this is `initial` chosen so the
  final fold ends at the last row.

For a **panel** forecaster (one with `id` set), `initial` cuts folds on the
*global timestamp grid*, since rows are spread across series: `initial`,
`step` and `horizon` count distinct timestamps. Each fold trains on every row
at or before the origin timestamp, forecasts each series forward from its own
last observed timestamp, and scores the result by joining forecasts to actuals
on `(id, time)`, so ragged series contribute wherever they have actuals.
`windows` instead counts the folds back from **each series' own end**, as the
M competitions hold out each series' last values: in fold `k`, every series
trains on its rows up to its own cutoff, and one global model is fitted on all
of them. If the series are not aligned in calendar time, the training set of a fold
can contain values from after another series' cutoff, a known pitfall of global
models (Hewamalage, Ackermann & Bergmeir 2023). With `initial`, all series are cut at
the same timestamp, which avoids it. For a panel, the `forecasts` table carries the id
column too.
A fold does not score a series that has no rows up to the fold's origin, too few of them
to train on (it is skipped with a warning), or no actuals after it, so the set of scored
series can change from fold to fold.

`metrics` is a tuple, vector or NamedTuple of callables `(y, ŷ) -> Float64` (e.g.
[`mae`](@ref), [`rmse`](@ref), [`smape`](@ref)). Each is named by
[`metric_name`](@ref) (`:mae`, `:mase_12` for [`Mase`](@ref)`(12)`) in the `scores`
table, or by its key in a NamedTuple, `metrics=(error=mae, scaled=Mase(12))`; the names
must be distinct. Scaled metrics that also need the fold's training target
([`mase`](@ref), [`rmsse`](@ref), [`Mase`](@ref), [`Rmsse`](@ref), or your own, declared
via [`metric_inputs`](@ref)) are called as `metric(y, ŷ; y_train=...)` with that
fold's training window supplied automatically (on a panel, each series' own, so a
scale never mixes series). Pass `Mase(12)`, not the type `Mase`.
Unscaled metrics pool the fold's rows; scaled metrics are per series, weighted by scored
points; [`summarize`](@ref) gives the unweighted mean over folds. Scaled metrics follow
Hyndman & Koehler (2006): each series' errors are divided by that series' own
training-window naive error, then averaged over series (here weighted by scored points,
our choice). Unscaled metrics are pooled across series, so on a panel they are dominated
by the large-scale series; use a scaled metric to compare series of different scale.

`levels` (percentages in (1, 100), such as `(80, 95)`) adds the columns `lo_<L>` and
`hi_<L>` to `forecasts`, as [`forecast`](@ref) does. A metric that scores samples
([`crps`](@ref), [`fair_crps`](@ref), [`pinball_loss`](@ref), [`Coverage`](@ref),
[`IntervalWidth`](@ref), or your own declared with [`metric_inputs`](@ref)) is called as
`metric(y, S)` with the fold's samples and pools the rows of all series, like an unscaled
point metric. `levels` and a samples metric both need a forecaster with an interval
method ([`ConformalSeasonalPool`](@ref)). The samples of a fold are drawn from that fold's
training window only. Every fold draws from its own copy of the method's `rng`, so the
folds start from the same random numbers on different histories.

When the feature set includes [`Exogenous`](@ref) columns, they are sliced from
the held-out data automatically (they are "known" in a backtest).
A backtest uses the realised values of the covariates, so it scores ex-post forecasts
(Hyndman & Athanasopoulos 2021, §7.6). For covariates not known at the origin (a
competitor's price, the weather) this typically understates the error of the real,
ex-ante forecast, which also carries the error of forecasting the covariate. For
covariates known in advance (calendar effects, planned promotions, prices you set)
ex-ante and ex-post agree. To measure ex-ante error, supply in the held-out rows the
covariate values that were available at the origin (a forecast, or a lagged column).

# Example
```julia
results = backtest(fc, df; horizon=28, initial=730, step=28,
                   metrics=(mae, rmse, smape, Mase(7)))
summarize(results)   # mean and std of each metric over folds
```
"""
function backtest(fc::Forecaster, data; horizon::Integer,
                  initial::Union{Nothing,Integer}=nothing,
                  windows::Union{Nothing,Integer}=nothing,
                  step::Integer=horizon, metrics=(mae, rmse), levels=nothing)
    _nobool(horizon, "backtest horizon", "Pass the number of steps each fold " *
            "forecasts, e.g. horizon=28.")
    horizon ≥ 1 || throw(ArgumentError(
        "backtest horizon must be ≥ 1, got $horizon. Pass the number of steps each " *
        "fold forecasts, e.g. horizon=28."))
    _nobool(step, "backtest step", "Pass how many steps each fold moves the origin, " *
            "e.g. step=$horizon, which equals horizon (the default).")
    scheme = _fold_scheme("backtest", initial, windows, step)
    step ≥ 1 || throw(ArgumentError(
        "backtest step must be ≥ 1, got $step. Pass how many steps each fold moves " *
        "the origin, e.g. step=$horizon, which equals horizon (the default)."))
    _check_horizon(fc, horizon)
    named = _named_metrics(metrics)
    lv = _check_levels(levels)
    if _wants_samples(lv, named)
        _require_intervals(fc, "backtest with levels or a sample-based metric")
        _check_support(fc.intervals, fc, lv, Int(horizon), "backtest horizon")
    end
    tbl = normalize_table(data)
    return _backtest_spec(fc, tbl, horizon, scheme, named, lv)
end

_wants_samples(lv, named) =
    !isempty(lv) || any(p -> metric_inputs(last(p)) isa Samples, named)

# Names the fold, so a sampler error says where it happened.
function _in_fold(f, k, origin)
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        at = origin === missing ? "fold $k" : "fold $k (origin $origin)"
        throw(ArgumentError("backtest $at: " * err.msg))
    end
end

_backtest_spec(fc::Forecaster{M,S,Symbol}, tbl, horizon, scheme, metrics,
               lv) where {M,S} = _backtest_panel(fc, tbl, horizon, scheme, metrics, lv)

function _backtest_spec(fc::Forecaster{M,S,Nothing}, tbl, horizon, scheme, metrics,
                        lv) where {M,S}
    t_all = require_column(tbl, fc.time, "time")
    validate_time_column(t_all, fc.time, fc.freq)
    y_all = target_vector(tbl, fc.target)
    origins = _origins(scheme, nrows(tbl), horizon, fc)
    _check_fold_rows(fc.intervals, fc, origins, t_all)
    exogcols = exogenous_columns(fc.features)

    T = eltype(t_all)
    fold_col = Int[]
    origin_col = T[]
    step_col = Int[]
    time_col = T[]
    y_col = Float64[]
    yhat_col = Float64[]
    m_fold = Int[]
    m_origin = Union{Missing,T}[]
    m_metric = Symbol[]
    m_value = Float64[]
    level_cols = [Float64[] for _ in 1:(2 * length(lv))]
    need_S = _wants_samples(lv, metrics)
    for (k, o) in enumerate(origins)
        sub = row_subset(tbl, 1:o)
        fitted = need_S ? fit(fc, sub) : _fit_fold(fc.intervals, fc, sub, 1:o)
        nd = isempty(exogcols) ? nothing :
             row_subset(tbl[Tuple([fc.time; exogcols])], (o + 1):(o + horizon))
        fcast, draws = need_S ?
            _in_fold(() -> _forecast_samples(fitted, [Int(horizon)], nd), k, t_all[o]) :
            (forecast(fitted, horizon; future=nd), nothing)
        samp = need_S ? only(draws) : nothing
        ytrue = y_all[(o + 1):(o + horizon)]
        yhat = fcast.y_hat
        append!(fold_col, fill(k, horizon))
        append!(origin_col, fill(t_all[o], horizon))
        append!(step_col, 1:horizon)
        append!(time_col, fcast[fc.time])
        append!(y_col, ytrue)
        append!(yhat_col, yhat)
        isempty(lv) || foreach(append!, level_cols, _interval_columns(samp, lv))
        fold = (; fc, history=y_all[1:o], y=ytrue, ŷ=yhat, S=samp)
        for (name, m) in metrics
            push!(m_fold, k)
            push!(m_origin, t_all[o])
            push!(m_metric, name)
            push!(m_value, Float64(score(metric_inputs(m), m, fold)))
        end
    end
    base = NamedTuple{(:fold, :origin, :step, fc.time, :y, :y_hat)}(
        (fold_col, origin_col, step_col, time_col, y_col, yhat_col))
    forecasts = merge(base, _level_table(keys(base), lv, level_cols))
    scores = (fold=m_fold, origin=m_origin, metric=m_metric, value=m_value)
    return BacktestResult(forecasts, scores)
end

function Base.show(io::IO, ::MIME"text/plain", r::BacktestResult)
    # Count from forecasts: it is filled even when metrics=().
    h = isempty(r.forecasts.step) ? 0 : maximum(r.forecasts.step)
    nfolds = length(unique(r.forecasts.fold))
    println(io, "BacktestResult: $nfolds fold$(nfolds == 1 ? "" : "s"), horizon $h")
    s = summarize(r)
    for i in eachindex(s.metric)
        println(io, "  ", rpad(string(s.metric[i]), 8), " = ",
                round(s.mean[i]; sigdigits=5), "  (mean over folds)")
    end
    print(io, "  (see .forecasts and .scores; summarize(r) for mean and std)")
end

Base.show(io::IO, r::BacktestResult) =
    print(io, "BacktestResult(", length(unique(r.forecasts.fold)), " folds)")
