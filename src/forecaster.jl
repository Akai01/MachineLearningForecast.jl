# Forecaster spec (immutable), FittedForecaster (fitted state), fit, and the
# strategy-agnostic forecast entry point.

"Column names used by `forecast`/`backtest` result tables; the time column may not shadow them."
const RESERVED_OUTPUT_NAMES = (:origin, :step, :y, :y_hat)

"""
    Forecaster(model; features, strategy=Recursive(), freq, target=:y, time=:ds)

Immutable forecasting specification.

- `model`: any MLJ `Deterministic` regressor instance (EvoTrees, DecisionTree,
  XGBoost, linear models, ... — anything implementing the MLJ model interface).
  Used as a prototype; `fit` never mutates it.
- `features`: a [`FeatureSet`](@ref) describing the design matrix.
- `strategy`: a [`ForecastStrategy`](@ref) — [`Recursive`](@ref) (default) or
  [`Direct`](@ref).
- `freq`: the series frequency as a `Dates.Period`, e.g. `Day(1)`, `Month(1)`.
- `target`: name of the target column (default `:y`).
- `time`: name of the time column (default `:ds`), of a type supporting
  `+` with `freq` (`Date`, `DateTime`).
- `id`: name of the series-id column for **panel (multi-series) data**, or
  `nothing` (the default) for a single series. When set, `data` is a long-format
  panel: one row per (series, timestamp). Features are materialised *within*
  each series — a lag never reaches across a series boundary — and the rows of
  every series are then pooled to fit **one global model**, so short or noisy
  series borrow strength from the rest. Series may be ragged (different start
  dates, end dates and lengths); each forecasts forward from its own last
  timestamp. `forecast` then returns an id column alongside the timestamps.

Fit with [`fit`](@ref), which returns a [`FittedForecaster`](@ref); the spec
itself is never mutated.

# Example
```julia
using MachineLearningForecast, Dates
fc = Forecaster(model;
                features=FeatureSet(Lag(1), Lag(7), Calendar(:dayofweek)),
                strategy=Recursive(), freq=Day(1), target=:y, time=:ds)

# a panel of many series, fitted as one global model
panel = Forecaster(model;
                   features=FeatureSet(Lag(1), Lag(7)),
                   strategy=Recursive(), freq=Day(1),
                   target=:y, time=:ds, id=:unique_id)
```
"""
struct Forecaster{M,S<:ForecastStrategy,I<:Union{Nothing,Symbol}}
    model::M
    features::FeatureSet
    strategy::S
    freq::Period
    target::Symbol
    time::Symbol
    id::I
    function Forecaster(model::M, features::FeatureSet, strategy::S, freq::Period,
                        target::Symbol, time::Symbol,
                        id::I=nothing) where {M,S<:ForecastStrategy,I<:Union{Nothing,Symbol}}
        model isa MLJModelInterface.Model || throw(ArgumentError(
            "model must be an MLJ model instance (subtype of MLJModelInterface.Model), " *
            "got $(typeof(model)). Pass e.g. EvoTreeRegressor(), " *
            "DecisionTreeRegressor(), or any other MLJ regressor."))
        model isa MLJModelInterface.Deterministic || @warn(
            "model $(typeof(model)) is not an MLJModelInterface.Deterministic " *
            "regressor; MachineLearningForecast expects point predictions and may fail at " *
            "predict time.")
        ts = MLJModelInterface.target_scitype(model)
        if !(AbstractVector{MLJModelInterface.Continuous} <: ts)
            @warn "model $(typeof(model)) declares target_scitype $ts, which does " *
                  "not cover AbstractVector{Continuous}; forecasting a continuous " *
                  "target may fail or behave unexpectedly."
        end
        target == time && throw(ArgumentError(
            "target and time must be different columns, both were :$target."))
        # A feature that emits the target column would feed the target straight
        # into the design matrix — a silent, total leak that backtests as a
        # perfect score. Reject it at construction.
        outs = outputnames(features)
        target in outs && throw(ArgumentError(
            "the feature set produces a column named :$target, which is the target " *
            "column. That would feed the target into its own design matrix (a total " *
            "leak: every backtest would score near-perfectly). Rename the exogenous/" *
            "custom column, or use a lag of the target, e.g. Lag(1)."))
        time in outs && throw(ArgumentError(
            "the feature set produces a column named :$time, which is the time " *
            "column. Timestamps cannot be used as a numeric feature directly — use " *
            "Calendar(...), Fourier(...), or a CustomFeature for a trend index."))
        # `backtest`/`forecast` return tables with these fixed column names.
        if time in RESERVED_OUTPUT_NAMES
            throw(ArgumentError(
                "time=:$time collides with a column name reserved by forecast()/" *
                "backtest() results ($(join(":" .* string.(RESERVED_OUTPUT_NAMES), ", "))). " *
                "Rename the time column in your data."))
        end
        if id !== nothing
            id == target && throw(ArgumentError(
                "id and target must be different columns, both were :$id."))
            id == time && throw(ArgumentError(
                "id and time must be different columns, both were :$id."))
            id in outs && throw(ArgumentError(
                "the feature set produces a column named :$id, which is the series id " *
                "column. Rename the feature's output column, or the id column."))
            id in RESERVED_OUTPUT_NAMES && throw(ArgumentError(
                "id=:$id collides with a column name reserved by forecast()/backtest() " *
                "results ($(join(":" .* string.(RESERVED_OUTPUT_NAMES), ", "))). " *
                "Rename the id column in your data."))
        end
        new{M,S,I}(model, features, strategy, freq, target, time, id)
    end
end

function Forecaster(model; features::FeatureSet, strategy::ForecastStrategy=Recursive(),
                    freq::Period, target::Symbol=:y, time::Symbol=:ds,
                    id::Union{Nothing,Symbol}=nothing)
    return Forecaster(model, features, strategy, freq, target, time, id)
end

"Is this a panel (multi-series) specification?"
ispanel(::Forecaster{M,S,Nothing}) where {M,S} = false
ispanel(::Forecaster{M,S,Symbol}) where {M,S} = true

"""
    SeriesState

Everything `forecast` needs about one series: its id (`nothing` for a
single-series forecaster), the training target history that lag and rolling
features are computed from, the first and last training timestamps, and the
number of training rows (which fixes the Fourier step index).

Read them off a fitted panel forecaster via `f.series`; for a single-series
forecaster the same values are reachable directly as `f.y_history`, `f.t_last`,
`f.t_start` and `f.n_train`.

# Example
```julia
f = fit(panel_spec, panel)
for st in f.series
    println(st.id, ": ", st.n_train, " rows ending ", st.t_last)
end
```
"""
struct SeriesState
    id::Any
    y_history::Vector{Float64}
    t_last::Any
    t_start::Any
    n_train::Int
end

"""
    FittedForecaster

The result of [`fit`](@ref): the immutable spec plus everything needed to
forecast — the fitted MLJ machine(s) (one for [`Recursive`](@ref),
`max_horizon` for [`Direct`](@ref)), the training target history (needed to
compute lag features at forecast time), the first/last training timestamps, and
the stable feature column order. Use with [`forecast`](@ref).
"""
struct FittedForecaster{F<:Forecaster}
    spec::F
    machines::Vector{MLJBase.Machine}
    series::Vector{SeriesState}
    feature_names::Vector{Symbol}
end

# Single-series state stays reachable as `fitted.y_history` etc., as documented,
# but those names are meaningless for a panel — say so rather than silently
# returning the first series.
const _SERIES_PROPS = (:y_history, :t_last, :t_start, :n_train)

function Base.getproperty(f::FittedForecaster, name::Symbol)
    name in _SERIES_PROPS || return getfield(f, name)
    st = getfield(f, :series)
    length(st) == 1 || throw(ArgumentError(
        "`fitted.$name` is single-series state, but this forecaster was fit on " *
        "$(length(st)) series. Use `fitted.series` for the per-series states, " *
        "each of which has .id, .y_history, .t_last, .t_start and .n_train."))
    return getfield(only(st), name)
end

Base.propertynames(f::FittedForecaster, private::Bool=false) =
    (fieldnames(FittedForecaster)..., _SERIES_PROPS...)

"""
    nseries(f::FittedForecaster) -> Int

The number of series the forecaster was fitted on: `1` for a single series, or
the number of usable series in the panel. Series too short for the feature set
are skipped at fit time, so this can be smaller than the number of ids in the
input.

# Example
```julia
f = fit(Forecaster(model; features=FeatureSet(Lag(1)), freq=Day(1), id=:unique_id),
        panel)
nseries(f)                       # e.g. 3
[s.id for s in f.series]         # the ids, in first-appearance order
```
"""
nseries(f::FittedForecaster) = length(getfield(f, :series))

"""
    fit(fc::Forecaster, data) -> FittedForecaster

Fit the forecaster on `data`, any Tables.jl-compatible table containing the
spec's time and target columns (plus any exogenous columns). The time column
must be sorted, duplicate-free, and gap-free with respect to `fc.freq`.

`fit` is a function owned by MachineLearningForecast (not an extension of `MLJBase.fit` or
`StatsAPI.fit`); qualify as `MachineLearningForecast.fit` when another `fit` is in scope.

# Example
```julia
fitted = fit(fc, df)
fcast  = forecast(fitted, 28)
```
"""
function fit(fc::Forecaster, data)
    tbl = normalize_table(data)
    return _fit_spec(fc, tbl)
end

# Single series: validate the one time column and fit directly.
function _fit_spec(fc::Forecaster{M,S,Nothing}, tbl::NamedTuple) where {M,S}
    t = require_column(tbl, fc.time, "time")
    validate_time_column(t, fc.time, fc.freq)
    return _fit(fc, fc.strategy, tbl)
end

# Panel: group by id, then fit ONE global model on the pooled rows (see panel.jl).
_fit_spec(fc::Forecaster{M,S,Symbol}, tbl::NamedTuple) where {M,S} = _fit_panel(fc, tbl)

# Shared by both strategies: materialize the design matrix and package the state.
_training_frame(fc::Forecaster, tbl::NamedTuple) =
    build_training_frame(fc.features, tbl, fc.target, fc.time)

function _fitted(fc::Forecaster, tbl::NamedTuple, machines::Vector{MLJBase.Machine})
    t = tbl[fc.time]
    st = SeriesState(nothing, target_vector(tbl, fc.target), t[end], t[1], length(t))
    return FittedForecaster(fc, machines, [st], outputnames(fc.features))
end

"""
    forecast(f::FittedForecaster, h::Int; new_data=nothing) -> NamedTuple

Forecast `h` steps past the end of the training data. Returns a columntable
(NamedTuple of vectors) with the spec's time column (the future timestamps
`t_last + freq, ..., t_last + h*freq`) and `:y_hat` (the point forecasts).
Being a columntable, the result is Tables.jl-compatible — pass it to
`DataFrame`, `CSV.write`, etc. as-is.

If the feature set contains any [`Exogenous`](@ref) feature, `new_data` is
required: a table with the time column and every exogenous column, covering
all `h` future timestamps (extra rows are ignored). Without exogenous
features, `new_data` must be omitted (it is ignored with a warning otherwise).

# Example
```julia
fcast = forecast(fitted, 28)
fcast = forecast(fitted, 28; new_data=future_exog)   # with Exogenous features
fcast.y_hat                                          # the point forecasts
```
"""
function forecast(f::FittedForecaster, h::Integer; new_data=nothing)
    h ≥ 1 || throw(ArgumentError("forecast horizon h must be ≥ 1, got $h."))
    return _forecast_spec(f, f.spec, h, new_data)
end

_forecast_spec(f::FittedForecaster, spec::Forecaster{M,S,Symbol}, h, new_data) where {M,S} =
    _forecast_panel(f, h, new_data)

function _forecast_spec(f::FittedForecaster, spec::Forecaster{M,S,Nothing},
                        h::Integer, new_data) where {M,S}
    st = only(getfield(f, :series))
    grid = future_grid(st.t_start, spec.freq, st.n_train, h)
    exogcols = exogenouscolumns(spec.features)
    exog_rows = nothing
    if !isempty(exogcols)
        new_data === nothing && throw(ArgumentError(
            "features contain Exogenous($(join(":" .* string.(exogcols), ", "))) but " *
            "forecast() was called without new_data. Pass a table with columns " *
            "(:$(spec.time), $(join(":" .* string.(exogcols), ", "))) covering " *
            "$(first(grid)) … $(last(grid))."))
        exog_rows = _exogenous_rows(spec, exogcols, grid, new_data)
    elseif new_data !== nothing
        @warn "new_data was passed but the feature set has no Exogenous features; " *
              "it will be ignored."
    end
    return _forecast(f, spec.strategy, h, grid, exog_rows)
end

# Align new_data to the forecast grid: one NamedTuple of exogenous values per step.
function _exogenous_rows(spec::Forecaster, exogcols::Vector{Symbol}, grid, new_data)
    nd = normalize_table(new_data)
    needed = [spec.time; exogcols]
    absent = [c for c in needed if !haskey(nd, c)]
    isempty(absent) || throw(ArgumentError(
        "new_data is missing column$(length(absent) == 1 ? "" : "s") " *
        "$(join(":" .* string.(absent), ", ")); it must contain the time column " *
        ":$(spec.time) and the exogenous column$(length(exogcols) == 1 ? "" : "s") " *
        "$(join(":" .* string.(exogcols), ", "))."))
    tcol = nd[spec.time]
    # A Date/DateTime (or otherwise incompatible) eltype would make every
    # lookup miss and be reported as "missing timestamps" — diagnose it directly.
    eltype(tcol) == eltype(grid) || throw(ArgumentError(
        "new_data's time column :$(spec.time) has element type $(eltype(tcol)) but " *
        "the training time column is $(eltype(grid)). Convert it (e.g. " *
        "`Date.(col)` / `DateTime.(col)`) so timestamps compare equal."))
    lookup = Dict{eltype(grid),Int}()
    for i in eachindex(tcol)
        haskey(lookup, tcol[i]) && throw(ArgumentError(
            "new_data has duplicate timestamps in column :$(spec.time); first " *
            "duplicate at $(tcol[i]). Deduplicate new_data — otherwise which row " *
            "supplies each forecast step is arbitrary."))
        lookup[tcol[i]] = i
    end
    absent_t = [t for t in grid if !haskey(lookup, t)]
    isempty(absent_t) || throw(ArgumentError(
        "new_data is missing $(length(absent_t)) of the $(length(grid)) timestamps " *
        "required for the forecast grid $(first(grid)) … $(last(grid)); first " *
        "missing: $(first(absent_t)). Provide exogenous values for every future step."))
    names = Tuple(exogcols)
    rows = Vector{NamedTuple}(undef, length(grid))
    for (s, t) in enumerate(grid)
        i = lookup[t]
        row = NamedTuple{names}(Tuple(nd[c][i] for c in exogcols))
        for c in exogcols
            ismissing(row[c]) && throw(ArgumentError(
                "new_data has a missing value in exogenous column :$c at time $t. " *
                "Provide complete exogenous values for every future step."))
        end
        rows[s] = row
    end
    return rows
end

# Assemble one future feature row in-place into preallocated length-1 column
# vectors (reused across steps), in the stable feature_names order.
function _fill_row!(colvecs::Vector{Vector{Float64}}, features::FeatureSet,
                    y_hist::AbstractVector{Float64}, t_next, n_next::Integer,
                    exog_row)
    j = 1
    for feat in features
        targ = _uses_step_index(feat) ? n_next : t_next
        for v in featurevalues(feat, y_hist, targ, exog_row)
            colvecs[j][1] = Float64(v)
            j += 1
        end
    end
    return nothing
end

function _prediction_row(f::FittedForecaster)
    colvecs = [Vector{Float64}(undef, 1) for _ in f.feature_names]
    row = NamedTuple{Tuple(f.feature_names)}(Tuple(colvecs))
    return colvecs, row
end

# The (time, y_hat) columntable returned by forecast.
_forecast_table(spec::Forecaster, grid, preds::Vector{Float64}) =
    NamedTuple{(spec.time, :y_hat)}((grid, preds))

function Base.show(io::IO, fc::Forecaster)
    print(io, "Forecaster(", nameof(typeof(fc.model)), ", ",
          length(fc.features), " features, ", fc.strategy, ", freq=", fc.freq,
          ", target=:", fc.target, ", time=:", fc.time, ")")
end

function Base.show(io::IO, f::FittedForecaster)
    spec = f.spec
    print(io, "FittedForecaster(", nameof(typeof(spec.model)), ", ",
          length(spec.features), " features, ", spec.strategy,
          ", freq=", spec.freq, ", trained on ", f.n_train,
          " rows ending ", f.t_last, ")")
end
