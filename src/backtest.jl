"""
    BacktestResult

Result of [`backtest`](@ref). Both fields are columntables (NamedTuples of
vectors, Tables.jl-compatible):

- `folds`: one row per fold × horizon step, with columns `:fold`, `:origin`
  (the last training timestamp the row's forecast starts from), `:step`
  (1..horizon), the spec's time column, `:y` (actual), and `:y_hat` (forecast).
  A panel backtest adds the id column after `:step`.
- `metrics`: one row per fold × metric (columns `:fold`, `:origin`, `:metric`,
  `:value`), plus one overall summary row per metric with `fold=0` (`:value`
  is the mean of the per-fold values, `:origin` is `missing`). A fold's
  `:origin` is `missing` when its series start their forecasts at different
  timestamps.

# Example
```julia
results = backtest(fc, df; horizon=28, initial=730)
results.folds.y_hat   # every forecast, fold by fold
results.metrics       # per-fold and overall scores
```
"""
struct BacktestResult
    folds::NamedTuple
    metrics::NamedTuple
end

_metric_name(m) = Symbol(m)

function _check_metric_names(ms)
    labels = [_metric_name(m) for m in ms]
    allunique(labels) && return nothing
    name = first(l for l in labels if count(==(l), labels) > 1)
    n = count(==(name), labels)
    throw(ArgumentError(
        "backtest metrics must have distinct names, but $n metrics are named " *
        "$(repr(name)), so their per-fold rows would be merged into one wrong " *
        "overall score. Remove the repeated metric, or give each variant its own " *
        "named function (closures from one factory share a name), e.g. " *
        "metrics=(mae, rmse)."))
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

function _origins(s::FromStart, n, horizon, mh)
    s.initial > mh || throw(ArgumentError(
        "backtest initial=$(s.initial) must exceed the feature set's minimum history " *
        "($mh rows) so the first training window has at least one usable row. " *
        "Pass initial=$(mh + 1) or more, or reduce lags/windows."))
    origins = s.initial:s.step:(n - horizon)
    isempty(origins) && throw(ArgumentError(
        "no complete backtest folds: data has $n rows, but the first fold needs " *
        "initial + horizon = $(s.initial + horizon). Provide more data or reduce " *
        "initial/horizon."))
    return origins
end

function _origins(s::FromEnd, n, horizon, mh)
    first = n - horizon - (s.windows - 1) * s.step
    first > mh || throw(ArgumentError(
        "backtest windows=$(s.windows) with step=$(s.step) and horizon=$horizon " *
        "would train its first fold on $first of the $n rows, but the feature set " *
        "needs more than $mh. Reduce windows, step or horizon, or provide more data."))
    return first:s.step:(n - horizon)
end

"""
    backtest(fc::Forecaster, data; horizon, initial=nothing, windows=nothing,
             step=horizon, metrics=(mae, rmse)) -> BacktestResult

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
the same timestamp, which avoids it. For a panel, the `folds` table carries the id column
too.
A fold does not score a series that has no rows up to the fold's origin, too few of them
to train on (it is skipped with a warning), or no actuals after it, so the set of scored
series can change from fold to fold.

`metrics` is a tuple of callables `(y, ŷ) -> Float64` (e.g. [`mae`](@ref),
[`rmse`](@ref), [`smape`](@ref)). Scaled metrics that also need the fold's
training target — [`mase`](@ref), or your own, declared via
[`needs_ytrain`](@ref) — are called as `metric(y, ŷ; y_train=...)` with that
fold's training window supplied automatically (on a panel, each series' own, so a
scale never mixes series).
Unscaled metrics pool the fold's rows; scaled metrics are per series, weighted by scored
points; the overall row is the unweighted mean over folds. Scaled metrics follow Hyndman
& Koehler (2006): each series' errors are divided by that series' own training-window
naive error, then averaged over series (here weighted by scored points, our choice).
Unscaled metrics are pooled across series, so on a panel they are dominated by the
large-scale series; use a scaled metric to compare series of different scale.

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
                   metrics=(mae, rmse, smape))
results.metrics   # per-fold and overall scores
```
"""
function backtest(fc::Forecaster, data; horizon::Integer,
                  initial::Union{Nothing,Integer}=nothing,
                  windows::Union{Nothing,Integer}=nothing,
                  step::Integer=horizon, metrics=(mae, rmse))
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
    if fc.strategy isa Direct && horizon > fc.strategy.max_horizon
        throw(ArgumentError(
            "backtest horizon=$horizon exceeds the Direct strategy's max_horizon=" *
            "$(fc.strategy.max_horizon). Use Direct($horizon) or reduce horizon."))
    end
    iterable = applicable(iterate, metrics)
    # A one-shot iterator must survive validation and every fold.
    ms = iterable && !(metrics isa Union{Tuple,NamedTuple,AbstractVector}) ?
         collect(metrics) : metrics
    iterable && all(_iscallable, ms) ||
        throw(ArgumentError(
            "backtest metrics must be a tuple or vector of metric functions, got " *
            "$(repr(metrics)). Pass e.g. metrics=(mae, rmse), or metrics=(mae,) " *
            "for one metric."))
    _check_metric_names(ms)
    tbl = normalize_table(data)
    return _backtest_spec(fc, tbl, horizon, scheme, ms)
end

_backtest_spec(fc::Forecaster{M,S,Symbol}, tbl, horizon, scheme,
               metrics) where {M,S} =
    _backtest_panel(fc, tbl, horizon, scheme, metrics)

function _backtest_spec(fc::Forecaster{M,S,Nothing}, tbl, horizon, scheme,
                        metrics) where {M,S}
    t_all = require_column(tbl, fc.time, "time")
    validate_time_column(t_all, fc.time, fc.freq)
    y_all = target_vector(tbl, fc.target)
    origins = _origins(scheme, nrows(tbl), horizon, minhistory(fc.features))
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
    for (k, o) in enumerate(origins)
        fitted = fit(fc, row_subset(tbl, 1:o))
        nd = isempty(exogcols) ? nothing :
             row_subset(tbl[Tuple([fc.time; exogcols])], (o + 1):(o + horizon))
        fcast = forecast(fitted, horizon; new_data=nd)
        ytrue = y_all[(o + 1):(o + horizon)]
        yhat = fcast.y_hat
        append!(fold_col, fill(k, horizon))
        append!(origin_col, fill(t_all[o], horizon))
        append!(step_col, 1:horizon)
        append!(time_col, fcast[fc.time])
        append!(y_col, ytrue)
        append!(yhat_col, yhat)
        for m in metrics
            push!(m_fold, k)
            push!(m_origin, t_all[o])
            push!(m_metric, _metric_name(m))
            push!(m_value, Float64(_apply_metric(m, ytrue, yhat, y_all[1:o])))
        end
    end
    for m in metrics
        name = _metric_name(m)
        vals = [m_value[i] for i in eachindex(m_value)
                if m_metric[i] == name && m_fold[i] > 0]
        push!(m_fold, 0)
        push!(m_origin, missing)
        push!(m_metric, name)
        push!(m_value, Statistics.mean(vals))
    end
    folds = NamedTuple{(:fold, :origin, :step, fc.time, :y, :y_hat)}(
        (fold_col, origin_col, step_col, time_col, y_col, yhat_col))
    mtable = (fold=m_fold, origin=m_origin, metric=m_metric, value=m_value)
    return BacktestResult(folds, mtable)
end

function Base.show(io::IO, ::MIME"text/plain", r::BacktestResult)
    # Count from folds: it is filled even when metrics=().
    h = isempty(r.folds.step) ? 0 : maximum(r.folds.step)
    nfolds = length(unique(r.folds.fold))
    println(io, "BacktestResult: $nfolds fold$(nfolds == 1 ? "" : "s"), horizon $h")
    for i in eachindex(r.metrics.fold)
        r.metrics.fold[i] == 0 || continue
        println(io, "  ", rpad(string(r.metrics.metric[i]), 8), " = ",
                round(r.metrics.value[i]; sigdigits=5), "  (mean over folds)")
    end
    print(io, "  (see .folds and .metrics for details)")
end

Base.show(io::IO, r::BacktestResult) =
    print(io, "BacktestResult(", length(unique(r.folds.fold)), " folds)")
