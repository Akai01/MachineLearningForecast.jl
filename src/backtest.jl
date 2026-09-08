# Expanding-window backtesting (time-series cross-validation).

"""
    BacktestResult

Result of [`backtest`](@ref). Both fields are columntables (NamedTuples of
vectors, Tables.jl-compatible):

- `folds`: one row per fold × horizon step, with columns `:origin` (the last
  training timestamp of the fold), `:step` (1..horizon), the spec's time
  column, `:y` (actual), and `:y_hat` (forecast).
- `metrics`: one row per fold × metric (columns `:fold`, `:origin`, `:metric`,
  `:value`), plus one overall summary row per metric with `fold=0` (`:value`
  is the mean of the per-fold values, `:origin` is `missing`).
"""
struct BacktestResult
    folds::NamedTuple
    metrics::NamedTuple
end

_metric_name(m) = Symbol(m)

"""
    backtest(fc::Forecaster, data; horizon, initial, step=horizon,
             metrics=(mae, rmse)) -> BacktestResult

Expanding-window time-series cross-validation: train on rows `1:initial`,
forecast `horizon` steps, evaluate against the held-out actuals; advance the
origin by `step` rows and repeat while a full horizon of actuals remains.

`metrics` is a tuple of callables `(y, ŷ) -> Float64` (e.g. [`mae`](@ref),
[`rmse`](@ref), [`smape`](@ref)). Scaled metrics that also need the fold's
training target — [`mase`](@ref), or your own, declared via
[`needs_ytrain`](@ref) — are called as `metric(y, ŷ; y_train=...)` with that
fold's training window supplied automatically. When the feature set includes
[`Exogenous`](@ref) columns, they are sliced from the held-out data
automatically (they are "known" in a backtest).

# Example
```julia
results = backtest(fc, df; horizon=28, initial=730, step=28,
                   metrics=(mae, rmse, smape))
results.metrics   # per-fold and overall scores
```
"""
function backtest(fc::Forecaster, data; horizon::Integer, initial::Integer,
                  step::Integer=horizon, metrics=(mae, rmse))
    horizon ≥ 1 || throw(ArgumentError("backtest horizon must be ≥ 1, got $horizon."))
    initial ≥ 1 || throw(ArgumentError("backtest initial must be ≥ 1, got $initial."))
    step ≥ 1 || throw(ArgumentError("backtest step must be ≥ 1, got $step."))
    if fc.strategy isa Direct && horizon > fc.strategy.max_horizon
        throw(ArgumentError(
            "backtest horizon=$horizon exceeds the Direct strategy's max_horizon=" *
            "$(fc.strategy.max_horizon). Use Direct($horizon) or reduce horizon."))
    end
    tbl = normalize_table(data)
    t_all = require_column(tbl, fc.time, "time")
    validate_time_column(t_all, fc.time, fc.freq)
    y_all = target_vector(tbl, fc.target)
    n = nrows(tbl)
    mh = minhistory(fc.features)
    initial > mh || throw(ArgumentError(
        "backtest initial=$initial must exceed the feature set's minimum history " *
        "($mh rows) so the first training window has at least one usable row."))
    origins = initial:step:(n - horizon)
    isempty(origins) && throw(ArgumentError(
        "no complete backtest folds: data has $n rows, but the first fold needs " *
        "initial + horizon = $(initial + horizon). Provide more data or reduce " *
        "initial/horizon."))
    exogcols = exogenouscolumns(fc.features)

    T = eltype(t_all)
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
        fitted = fit(fc, rowsubset(tbl, 1:o))
        nd = isempty(exogcols) ? nothing :
             rowsubset(tbl[Tuple([fc.time; exogcols])], (o + 1):(o + horizon))
        fcast = forecast(fitted, horizon; new_data=nd)
        ytrue = y_all[(o + 1):(o + horizon)]
        yhat = fcast.y_hat
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
    # Overall summary: mean of the per-fold values, one row per metric, fold=0.
    for m in metrics
        name = _metric_name(m)
        vals = [m_value[i] for i in eachindex(m_value) if m_metric[i] == name && m_fold[i] > 0]
        push!(m_fold, 0)
        push!(m_origin, missing)
        push!(m_metric, name)
        push!(m_value, Statistics.mean(vals))
    end
    folds = NamedTuple{(:origin, :step, fc.time, :y, :y_hat)}(
        (origin_col, step_col, time_col, y_col, yhat_col))
    mtable = (fold=m_fold, origin=m_origin, metric=m_metric, value=m_value)
    return BacktestResult(folds, mtable)
end

function Base.show(io::IO, ::MIME"text/plain", r::BacktestResult)
    # Derive both counts from `folds`, which is populated even when `metrics=()`.
    h = isempty(r.folds.origin) ? 0 : count(==(first(r.folds.origin)), r.folds.origin)
    nfolds = h == 0 ? 0 : length(r.folds.origin) ÷ h
    println(io, "BacktestResult: $nfolds fold$(nfolds == 1 ? "" : "s"), horizon $h")
    for i in eachindex(r.metrics.fold)
        r.metrics.fold[i] == 0 || continue
        println(io, "  ", rpad(string(r.metrics.metric[i]), 8), " = ",
                round(r.metrics.value[i]; sigdigits=5), "  (mean over folds)")
    end
    print(io, "  (see .folds and .metrics for details)")
end

function Base.show(io::IO, r::BacktestResult)
    h = isempty(r.folds.origin) ? 0 : count(==(first(r.folds.origin)), r.folds.origin)
    print(io, "BacktestResult(", h == 0 ? 0 : length(r.folds.origin) ÷ h, " folds)")
end
