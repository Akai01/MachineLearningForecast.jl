"""
Column names used by `forecast`/`backtest` result tables; the time column may
not shadow them.
"""
const RESERVED_OUTPUT_NAMES = (:origin, :step, :y, :y_hat)

"""
    Forecaster(model; features, strategy=Recursive(), freq, target=:y, time=:ds, id=nothing,
               transform=NoTransform())

Immutable forecasting specification.

- `model`: any MLJ `Deterministic` regressor instance (EvoTrees, DecisionTree,
  XGBoost, linear models, ... — anything implementing the MLJ model interface),
  or a `Probabilistic` one, whose predicted mean is the forecast. Used as a
  prototype; `fit` never mutates it.
- `features`: a [`FeatureSet`](@ref) describing the design matrix.
- `strategy`: a [`ForecastStrategy`](@ref) — [`Recursive`](@ref) (default) or
  [`Direct`](@ref).
- `freq`: the series frequency as a positive `Dates.Period`, e.g. `Day(1)`,
  `Month(1)`; a sum such as `Day(1) + Hour(1)` is not accepted.
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
- `transform`: a [`TargetTransform`](@ref) applied to each series' target
  before features are built, e.g. [`LastScale`](@ref) or
  `TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))`. Its state is
  fitted per series on the training history only, and forecasts are mapped
  back to the original scale. The default [`NoTransform`](@ref) leaves the
  target as given.

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
struct Forecaster{M,S<:ForecastStrategy,I<:Union{Nothing,Symbol},X<:TargetTransform}
    model::M
    features::FeatureSet
    strategy::S
    freq::Period
    target::Symbol
    time::Symbol
    id::I
    transform::X
    function Forecaster(model; features, strategy::ForecastStrategy=Recursive(), freq,
                        target::Symbol=:y, time::Symbol=:ds,
                        id::Union{Nothing,Symbol}=nothing,
                        transform::TargetTransform=NoTransform())
        features isa FeatureSet || throw(ArgumentError(
            "features must be a FeatureSet such as FeatureSet(Lag(1), Lag(7)), got " *
            "$(repr(features)). Wrap the features in FeatureSet(...)."))
        freq isa Period || throw(ArgumentError(
            "freq must be a Dates.Period such as Day(1) or Month(1), got " *
            "$(repr(freq)); a sum such as Day(1) + Hour(1) is not accepted. Pass e.g. " *
            "freq=Day(1)."))
        freq > zero(freq) || throw(ArgumentError(
            "freq must be a positive period, got $freq. Pass e.g. freq=Day(1) or " *
            "freq=Month(1)."))
        model isa Type && model <: MLJModelInterface.Model && throw(ArgumentError(
            "model must be an MLJ model instance, got the type $(nameof(model)). Pass " *
            "an instance instead, e.g. $(nameof(model))()."))
        model isa MLJModelInterface.Model || throw(ArgumentError(
            "model must be an MLJ model instance (subtype of MLJModelInterface.Model), " *
            "got $(typeof(model)). Pass e.g. EvoTreeRegressor(), " *
            "DecisionTreeRegressor(), or any other MLJ regressor."))
        model isa Union{MLJModelInterface.Deterministic,
                        MLJModelInterface.Probabilistic} || @warn(
            "model $(typeof(model)) is neither an MLJModelInterface.Deterministic nor " *
            "a Probabilistic regressor; MachineLearningForecast expects point " *
            "predictions or predicted distributions and may fail at predict time.")
        ts = MLJModelInterface.target_scitype(model)
        if !(AbstractVector{MLJModelInterface.Continuous} <: ts)
            @warn "model $(typeof(model)) declares target_scitype $ts, which does " *
                  "not cover AbstractVector{Continuous}; forecasting a continuous " *
                  "target may fail or behave unexpectedly."
        end
        # Every feature table is Continuous; Unknown skips, as in MLJ.
        is = MLJModelInterface.input_scitype(model)
        if is !== MLJModelInterface.Unknown &&
           !(MLJModelInterface.Table{AbstractVector{MLJModelInterface.Continuous}} <: is)
            @warn "model $(typeof(model)) declares input_scitype $is, which does not " *
                  "cover a table of Continuous columns, the only input " *
                  "MachineLearningForecast passes; fitting may fail. Use a regressor " *
                  "that accepts Continuous features."
        end
        target == time && throw(ArgumentError(
            "target and time must be different columns, both were :$target. Pass the " *
            "names of your target and time columns, e.g. target=:y, time=:ds."))
        # A target-named feature is a silent, total leak.
        outs = outputnames(features)
        target in outs && throw(ArgumentError(
            "the feature set produces a column named :$target, which is the target " *
            "column. That would feed the target into its own design matrix (a total " *
            "leak: every backtest would score near-perfectly). Rename the exogenous/" *
            "custom column, or use a lag of the target, e.g. Lag(1)."))
        time in outs && throw(ArgumentError(
            "the feature set produces a column named :$time, which is the time " *
            "column. Timestamps cannot be used as a numeric feature directly: use " *
            "Calendar(...) or Fourier(...) for time-based features."))
        # forecast and backtest results use these column names.
        if time in RESERVED_OUTPUT_NAMES
            throw(ArgumentError(
                "time=:$time collides with a column name reserved by forecast()/" *
                "backtest() results " *
                "($(join(":" .* string.(RESERVED_OUTPUT_NAMES), ", "))). " *
                "Rename the time column in your data."))
        end
        if id !== nothing
            id == target && throw(ArgumentError(
                "id and target must be different columns, both were :$id. Pass the " *
                "name of your series-id column, e.g. id=:unique_id."))
            id == time && throw(ArgumentError(
                "id and time must be different columns, both were :$id. Pass the " *
                "name of your series-id column, e.g. id=:unique_id."))
            id in outs && throw(ArgumentError(
                "the feature set produces a column named :$id, which is the series id " *
                "column. Rename the feature's output column, or the id column."))
            id in RESERVED_OUTPUT_NAMES && throw(ArgumentError(
                "id=:$id collides with a column name reserved by forecast()/backtest() " *
                "results ($(join(":" .* string.(RESERVED_OUTPUT_NAMES), ", "))). " *
                "Rename the id column in your data."))
        end
        return new{typeof(model),typeof(strategy),typeof(id),typeof(transform)}(
            model, features, strategy, freq, target, time, id, transform)
    end
end

"""
Fields `reconstruct` (and so `tune`) may override. The data-shape fields
(`freq`, `target`, `time`, `id`) stay fixed: they describe the data, not the
model.
"""
const TUNABLE_FIELDS = (:model, :features, :strategy, :transform)

"""
Throw unless every key in `ks` is tunable. A name that is not a field is
unknown; a field outside `TUNABLE_FIELDS` describes the data.
"""
function _check_tunable(ks, lead, noun, subject)
    list(v) = join(":" .* string.(v), ", ")
    unknown = setdiff(ks, fieldnames(Forecaster))
    isempty(unknown) || throw(ArgumentError(
        "$lead unknown $noun$(length(unknown) == 1 ? "" : "s") $(list(unknown)); " *
        "valid $(noun)s are $(list(TUNABLE_FIELDS))."))
    data = setdiff(ks, TUNABLE_FIELDS)
    isempty(data) || throw(ArgumentError(
        "$subject$(length(data) == 1 ? "" : "s") $(list(data)) " *
        "$(length(data) == 1 ? "describes" : "describe") the data and cannot be " *
        "tuned; tunable $(noun)s are $(list(TUNABLE_FIELDS))."))
    return nothing
end

# Loops fieldnames so a new field is never dropped.
function reconstruct(fc::Forecaster; kwargs...)
    _check_tunable(keys(kwargs), "cannot reconstruct Forecaster with", "field", "field")
    vals = (; (f => get(kwargs, f, getfield(fc, f)) for f in fieldnames(Forecaster))...)
    return Forecaster(vals.model; (f => v for (f, v) in pairs(vals) if f !== :model)...)
end

"Is this a panel (multi-series) specification?"
ispanel(::Forecaster{M,S,Nothing}) where {M,S} = false
ispanel(::Forecaster{M,S,Symbol}) where {M,S} = true

"""
    SeriesState

Everything `forecast` needs about one series: its id (`nothing` for a
single-series forecaster), the training target history that lag and rolling
features are computed from, the first and last training timestamps, and the
number of training rows. `z_history` is the history after the spec's
[`TargetTransform`](@ref), which lag and rolling features and the step index
use, and `transform_state` is what [`untransform`](@ref) needs; without a
transform, `z_history` is `y_history`. `step0` is the context's step index
(`ctx.step`, see [`FeatureContext`](@ref)) of the first `z_history` value: `0`
for a single series, and in a panel the steps from the earliest usable series'
first (transformed) timestamp.

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
    z_history::Vector{Float64}
    transform_state::Any
    step0::Int
end

"What fit builds per series: its state and its training rows."
struct SeriesFrame{D<:NamedTuple}
    state::SeriesState
    X::D
    y::Vector{Float64}
end

"""
    TrainedModel{M,R}

One model trained by [`fit`](@ref): `model` is the copy of the spec's model that
was trained, and `fitresult` is what its `MLJModelInterface.fit` returned. Pass
both to MLJ's interface functions to inspect it, e.g.
`MLJModelInterface.fitted_params(t.model, t.fitresult)`.
"""
struct TrainedModel{M<:MLJModelInterface.Model,R}
    model::M
    fitresult::R
end

# The prototype is copied so fit never mutates it.
function _train(prototype::MLJModelInterface.Model, X, y)
    model = deepcopy(prototype)
    warning = MLJModelInterface.clean!(model)
    isempty(warning) || @warn warning
    fitresult, _, _ = MLJModelInterface.fit(model, 0,
                                            MLJModelInterface.reformat(model, X, y)...)
    return TrainedModel(model, fitresult)
end

# A comprehension costs ~50 ms of first-call compile.
function _collect(f, xs)
    out = [f(first(xs))]
    for i in 2:length(xs)
        v = f(xs[i])
        typeof(v) <: eltype(out) ||
            (out = Vector{Base.promote_typejoin(eltype(out), typeof(v))}(out))
        push!(out, v)
    end
    return out
end

# One series: its frame as is, no copy.
_stack_frames(::Forecaster{M,S,Nothing}, frames) where {M,S} =
    (only(frames).X, only(frames).y)
_stack_frames(::Forecaster{M,S,Symbol}, frames) where {M,S} =
    (_vcat_frames([fr.X for fr in frames]), reduce(vcat, [fr.y for fr in frames]))

# One series: ranges, no index vectors.
_pair_index(::Forecaster{M,S,Nothing}, lens, i) where {M,S} =
    (1:(only(lens) - i + 1), i:only(lens))
_pair_index(::Forecaster{M,S,Symbol}, lens, i) where {M,S} = _pair_vectors(lens, i)

function _train_models(fc::Forecaster, ::Recursive, frames)
    X, y = _stack_frames(fc, frames)
    return [_train(fc.model, X, y)]
end

function _train_models(fc::Forecaster, s::Direct, frames)
    lens = _collect(fr -> length(fr.y), frames)
    nmax = maximum(lens)
    nmax ≥ s.max_horizon || throw(_direct_rows_error(fc, s, nmax))
    _warn_short_series(fc, s, frames, lens)
    anch = anchors(fc.features)
    X, y = _stack_frames(fc, frames)
    return map(1:s.max_horizon) do i
        from, to = _pair_index(fc, lens, i)
        cols = Vector{Vector{Float64}}(undef, length(X))
        for c in eachindex(cols)
            cols[c] = values(X)[c][_gather_index(anch[c], from, to)]
        end
        _train(fc.model, NamedTuple{keys(X)}(Tuple(cols)), y[to])
    end
end

# A single series cannot be short after the check above.
_warn_short_series(::Forecaster{M,S,Nothing}, s, frames, lens) where {M,S} = nothing
function _warn_short_series(fc::Forecaster{M,S,Symbol}, s, frames, lens) where {M,S}
    short = [(fr.state.id, n) for (fr, n) in zip(frames, lens) if n < s.max_horizon]
    isempty(short) && return nothing
    k = length(short)
    names = join((string(fc.id, "=", repr(id), " (", n, " training rows)")
                  for (id, n) in first(short, 5)), ", ")
    @warn "Direct($(s.max_horizon)): $k series with fewer than $(s.max_horizon) " *
          "training rows $(k == 1 ? "is" : "are") left out of the step models beyond " *
          "their row count: $names$(k > 5 ? ", …" : "")"
    return nothing
end

_direct_rows_error(fc::Forecaster{M,S,Nothing}, s, nmax) where {M,S} = ArgumentError(
    "not enough data for Direct($(s.max_horizon)): after dropping the feature set's " *
    "$(minhistory(fc.features)) history rows, only $nmax training rows remain, but the " *
    "target must be shifted up to $(s.max_horizon - 1) steps. Provide at least " *
    "$(minhistory(fc.features) + s.max_horizon) rows, reduce max_horizon, or reduce " *
    "lags/windows.")
_direct_rows_error(fc::Forecaster{M,S,Symbol}, s, nmax) where {M,S} = ArgumentError(
    "not enough data for Direct($(s.max_horizon)): no series has enough rows to train " *
    "the step-$(nmax + 1) model. Shorten max_horizon, provide longer series, or reduce " *
    "lags/windows.")

_predict(t::TrainedModel, X) = MLJModelInterface.predict(
    t.model, t.fitresult, MLJModelInterface.reformat(t.model, X)...)

# A Probabilistic model's point forecast is its predicted mean.
_predict(t::TrainedModel{<:MLJModelInterface.Probabilistic}, X) =
    MLJModelInterface.predict_mean(t.model, t.fitresult,
                                   MLJModelInterface.reformat(t.model, X)...)

"""
    FittedForecaster

The result of [`fit`](@ref): the immutable spec plus everything needed to
forecast — the trained models (one for [`Recursive`](@ref), `max_horizon` for
[`Direct`](@ref)), the training target history (needed to compute lag features
at forecast time), the first/last training timestamps, and the stable feature
column order. Use with [`forecast`](@ref).

`fitted.models` holds the trained models, each with the trained copy of the
spec's model as `.model` and its MLJ fitresult as `.fitresult`.
`fitted.series` holds one [`SeriesState`](@ref) per series. For a single
series, `fitted.y_history`, `fitted.t_last`, `fitted.t_start` and
`fitted.n_train` read its one state directly; on a panel they throw an
`ArgumentError` pointing at `fitted.series`.

# Example
```julia
fitted = fit(fc, df)
fitted.models        # the trained model(s)
fitted.series        # one SeriesState per series
fitted.t_last        # single series only: the last training timestamp
forecast(fitted, 28)
```
"""
struct FittedForecaster{F<:Forecaster,T<:TrainedModel}
    spec::F
    models::Vector{T}
    series::Vector{SeriesState}
    feature_names::Vector{Symbol}
end

# Panels have no single history; refuse, don't guess.
const _SERIES_PROPS = (:y_history, :t_last, :t_start, :n_train, :z_history,
                       :transform_state)

function Base.getproperty(f::FittedForecaster, name::Symbol)
    name in _SERIES_PROPS || return getfield(f, name)
    return _series_prop(getfield(f, :spec), f, name)
end

_series_prop(::Forecaster{M,S,Nothing}, f, name) where {M,S} =
    getfield(only(getfield(f, :series)), name)
_series_prop(::Forecaster{M,S,Symbol}, f, name) where {M,S} = throw(ArgumentError(
    "`fitted.$name` is single-series state, but this forecaster was fit on " *
    "$(nseries(f)) series. Use `fitted.series` for the per-series states, " *
    "each of which has .id, .y_history, .t_last, .t_start and .n_train."))

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
using Dates
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
spec's time and target columns (plus any exogenous columns, and the id column
for a panel). For a single series the time column must be sorted,
duplicate-free, and gap-free with respect to `fc.freq`. For a panel, rows may
come in any order: `fit` sorts each series by time, and each series must be
duplicate-free and gap-free.

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

# One series: strict, one group, no copy.
function _series_groups(fc::Forecaster{M,S,Nothing}, tbl::NamedTuple,
                        base=1:nrows(tbl)) where {M,S}
    t = require_column(tbl, fc.time, "time")
    validate_time_column(t, fc.time, fc.freq, base)
    return [(nothing, tbl, base)]
end

_in_series(f, ::Forecaster{M,S,Nothing}, key) where {M,S} = f()

function _prepare_series(fc::Forecaster, id, sub::NamedTuple, rows)
    return _in_series(fc, id) do
        yraw = target_vector(sub, fc.target, rows)
        tz, z, state = _transform_table(fc, sub, yraw)
        return (; id, t=sub[fc.time], rows, yraw, tz, z, state)
    end
end

_usable_series(::Forecaster{M,S,Nothing}, prepared) where {M,S} = prepared

# A transform can empty a single series: no origin.
_fit_origin(::Forecaster{M,S,Nothing}, usable) where {M,S} = nothing
_step0(fc, tz, ::Nothing) = 0
_step0(fc, tz, origin) = _steps_between(origin, tz[fc.time][1], fc.freq)

function _fit_series(fc::Forecaster, p, origin)
    step0 = _step0(fc, p.tz, origin)
    X, y, _ = _in_series(fc, p.id) do
        build_training_frame(fc.features, p.tz, fc.target, fc.time,
                             p.rows[(length(p.yraw) - nrows(p.tz) + 1):end]; step0)
    end
    state = SeriesState(p.id, p.yraw, p.t[end], p.t[1], length(p.t), p.z, p.state, step0)
    return SeriesFrame(state, X, y)
end

function _fit_spec(fc::Forecaster, tbl::NamedTuple, base=1:nrows(tbl))
    groups = _series_groups(fc, tbl, base)
    prepared = _collect(g -> _prepare_series(fc, g...), groups)
    usable = _usable_series(fc, prepared)
    origin = _fit_origin(fc, usable)
    frames = _collect(p -> _fit_series(fc, p, origin), usable)
    models = _train_models(fc, fc.strategy, frames)
    states = SeriesState[]
    for fr in frames
        push!(states, fr.state)
    end
    return FittedForecaster(fc, models, states, outputnames(fc.features))
end

"""
    forecast(f::FittedForecaster, h::Integer; future=nothing) -> NamedTuple

Forecast `h` steps past the end of the training data. Returns a columntable
(NamedTuple of vectors) with the spec's time column (the future timestamps
`t_last + freq, ..., t_last + h*freq`) and `:y_hat` (the point forecasts).
Being a columntable, the result is Tables.jl-compatible — pass it to
`DataFrame`, `CSV.write`, etc. as-is.

If the feature set contains any [`Exogenous`](@ref) feature, `future` is
required: a table with the time column and every exogenous column, covering
all `h` future timestamps (extra rows are ignored). Without exogenous
features, `future` must be omitted (it is ignored with a warning otherwise).

For a panel forecaster (one with `id` set), the result also has the id column,
before the time column: `h` rows per series, each series forecast from its own
last training timestamp. `future` must then carry the id column too and cover
every series at each of its `h` future timestamps.

# Example
```julia
fcast = forecast(fitted, 28)
fcast = forecast(fitted, 28; future=future_exog)   # with Exogenous features
fcast.y_hat                                        # the point forecasts
```
"""
function forecast(f::FittedForecaster, h::Integer; future=nothing)
    _nobool(h, "forecast horizon h", "Pass the number of steps to forecast, e.g. " *
            "forecast(fitted, 28).")
    h ≥ 1 || throw(ArgumentError(
        "forecast horizon h must be ≥ 1, got $h. Pass the number of steps to " *
        "forecast, e.g. forecast(fitted, 28)."))
    return _forecast(f, fill(Int(h), nseries(f)), future)
end

# hs[k] is series k's horizon, which backtest may shorten.
function _forecast(f::FittedForecaster, hs::Vector{Int}, future)
    spec = f.spec
    states = getfield(f, :series)
    grids = _grids(states, spec.freq, hs)
    exog_rows = _exogenous_rows(spec, states, grids, future, hs)
    preds = _forecast_states(f, states, hs, grids, exog_rows)
    return _output_table(spec, states, grids, preds)
end

# Apart so the closure does not carry the model type.
_grids(states, freq, hs) = _collect(
    k -> future_grid(states[k].t_start, freq, states[k].n_train, hs[k]),
    eachindex(states))

_output_table(spec::Forecaster{M,S,Nothing}, states, grids, preds) where {M,S} =
    NamedTuple{(spec.time, :y_hat)}((only(grids), only(preds)))

function _check_horizon(f::FittedForecaster, h::Integer)
    lim = horizon_limit(f.spec.strategy)
    h ≤ lim || throw(ArgumentError(
        "strategy=Direct($lim) was fit with max_horizon=$lim but forecast(h=$h) was " *
        "requested. Refit with Direct($h) or use Recursive()."))
    return nothing
end

function _check_horizon(fc::Forecaster, h::Integer)
    lim = horizon_limit(fc.strategy)
    h ≤ lim || throw(ArgumentError(
        "backtest horizon=$h exceeds the Direct strategy's max_horizon=$lim. " *
        "Use Direct($h) or reduce horizon."))
    return nothing
end

# A step loop's state; it holds no model type.
struct _StepRun{S<:ForecastStrategy,G,E}
    features::FeatureSet
    strategy::S
    states::Vector{SeriesState}
    hs::Vector{Int}
    grids::G
    exog_rows::E
    histories::Vector{Vector{Float64}}
    scratch::Vector{Vector{Float64}}
    cols::Vector{Vector{Float64}}
    bufs::Vector{Vector{Float64}}
    active::Vector{Int}
    preds::Vector{Vector{Float64}}
end

function _StepRun(features::FeatureSet, strategy, names::Vector{Symbol}, states, hs, grids,
                  exog_rows)
    n = length(states)
    scratch = Vector{Vector{Float64}}(undef, length(names))
    cols = Vector{Vector{Float64}}(undef, length(names))
    for j in eachindex(names)
        scratch[j] = Vector{Float64}(undef, 1)
        cols[j] = Vector{Float64}(undef, n)
    end
    preds = Vector{Vector{Float64}}(undef, n)
    active = Vector{Int}(undef, n)
    for k in 1:n
        preds[k] = Vector{Float64}(undef, hs[k])
        active[k] = k
    end
    return _StepRun(features, strategy, states, hs, grids, exog_rows,
                    _histories(strategy, states, hs), scratch, cols,
                    _feature_buffers(features), active, preds)
end

# One row per series still being forecast.
function _fill_step!(run::_StepRun, s::Int)
    active, hs = run.active, run.hs
    m = 0
    for k in active
        hs[k] ≥ s || continue
        m += 1
        active[m] = k
    end
    changed = m != length(active)
    resize!(active, m)
    if changed
        # New columns on a shrink: a model may have shared them.
        for j in eachindex(run.cols)
            run.cols[j] = Vector{Float64}(undef, m)
        end
    end
    for r in 1:m
        k = active[r]
        ex = run.exog_rows === nothing ? nothing : run.exog_rows[k][s]
        ctx = _context(run.states[k], run.histories[k], run.grids[k][s], s, ex)
        _fill_row!(run.scratch, run.bufs, run.features, ctx)
        for j in eachindex(run.cols)
            run.cols[j][r] = run.scratch[j][1]
        end
    end
    return changed
end

function _store_step!(run::_StepRun, s::Int, ŷ, @nospecialize(trained))
    active = run.active
    length(ŷ) == length(active) || throw(ArgumentError(
        "model $(nameof(typeof(trained.model))) returned $(length(ŷ)) predictions for " *
        "$(length(active)) forecast row(s) at step $s; a model must return one " *
        "prediction per row."))
    for r in eachindex(active)
        k = active[r]
        v = Float64(ŷ[r])
        run.preds[k][s] = v
        _feeds_back(run.strategy) && push!(run.histories[k], v)
    end
    return nothing
end

function _forecast_states(f::FittedForecaster, states, hs::Vector{Int}, grids, exog_rows)
    spec = f.spec
    strategy = spec.strategy
    hmax = maximum(hs)
    _check_horizon(f, hmax)
    run = _StepRun(spec.features, strategy, f.feature_names, states, hs, grids, exog_rows)
    # Reused across steps: a model must not keep Xnew.
    row = NamedTuple{Tuple(f.feature_names)}(Tuple(run.cols))
    for s in 1:hmax
        _fill_step!(run, s) && (row = NamedTuple{Tuple(f.feature_names)}(Tuple(run.cols)))
        trained = _model_at(f.models, strategy, s)
        _store_step!(run, s, _predict(trained, row), trained)
    end
    return _untransform_all(spec.transform, run.preds, states)
end

# Apart: user transforms extend untransform.
function _untransform_all(transform, preds, states)
    out = Vector{Vector{Float64}}(undef, length(states))
    for k in eachindex(states)
        out[k] = untransform(transform, preds[k], states[k].transform_state)
    end
    return out
end

function _exogenous_rows(spec::Forecaster{M,S,Nothing}, states, grids, future,
                         hs) where {M,S}
    grid = only(grids)
    exogcols = exogenous_columns(spec.features)
    if isempty(exogcols)
        if future !== nothing
            @warn "`future` was passed but the feature set has no Exogenous " *
                  "features; it will be ignored."
        end
        return nothing
    end
    future === nothing && throw(ArgumentError(
        "features contain Exogenous($(join(":" .* string.(exogcols), ", "))) but " *
        "forecast() was called without `future`. Pass a table with columns " *
        "(:$(spec.time), $(join(":" .* string.(exogcols), ", "))) covering " *
        "$(first(grid)) … $(last(grid))."))
    nd = normalize_table(future)
    needed = [spec.time; exogcols]
    absent = [c for c in needed if !haskey(nd, c)]
    isempty(absent) || throw(ArgumentError(
        "`future` is missing column$(length(absent) == 1 ? "" : "s") " *
        "$(join(":" .* string.(absent), ", ")); it must contain the time column " *
        ":$(spec.time) and the exogenous column$(length(exogcols) == 1 ? "" : "s") " *
        "$(join(":" .* string.(exogcols), ", "))."))
    tcol = nd[spec.time]
    lookup = Dict{eltype(grid),Int}()
    for i in eachindex(tcol)
        v = _new_time(tcol[i], eltype(grid), spec.time, i)
        haskey(lookup, v) && throw(ArgumentError(
            "`future` has duplicate timestamps in column :$(spec.time); first " *
            "duplicate at $v. Deduplicate `future` — otherwise which row " *
            "supplies each forecast step is arbitrary."))
        lookup[v] = i
    end
    absent_t = [t for t in grid if !haskey(lookup, t)]
    isempty(absent_t) || throw(ArgumentError(
        "`future` is missing $(length(absent_t)) of the $(length(grid)) timestamps " *
        "required for the forecast grid $(first(grid)) … $(last(grid)); first " *
        "missing: $(first(absent_t)). Provide exogenous values for every future step."))
    names = Tuple(exogcols)
    rows = Vector{NamedTuple}(undef, length(grid))
    for (s, t) in enumerate(grid)
        i = lookup[t]
        row = NamedTuple{names}(Tuple(nd[c][i] for c in exogcols))
        for c in exogcols
            ismissing(row[c]) && throw(ArgumentError(
                "`future` has a missing value in exogenous column :$c at time $t. " *
                "Provide complete exogenous values for every future step."))
        end
        rows[s] = row
    end
    return [rows]
end

function _new_time(v, T::Type, time::Symbol, i::Integer)
    ismissing(v) && throw(ArgumentError(
        "`future`'s time column :$time has a missing value at row $i. Every " *
        "`future` row needs a timestamp; drop or fill that row."))
    # Name the eltype mismatch, not an obscure Dict key error.
    v isa T || throw(ArgumentError(
        "`future`'s time column :$time has element type $(typeof(v)) at row $i, " *
        "but the training time column is $T. Convert it (e.g. `Date.(col)` / " *
        "`DateTime.(col)`) so timestamps compare equal."))
    return v
end

_context(st::SeriesState, y_hist::Vector{Float64}, t_next, s::Integer, exog_row) =
    FeatureContext(y_hist, t_next, st.step0 + length(st.z_history) - 1 + s, exog_row)

function _fill_row!(colvecs::Vector{Vector{Float64}}, bufs::Vector{Vector{Float64}},
                    features::FeatureSet, ctx::FeatureContext)
    j = 1
    for (feat, buf) in zip(features, bufs)
        j = _fill_feature!(colvecs, j, buf, feat, ctx)::Int
    end
    return nothing
end

# Barrier: compiles per feature and context type.
function _fill_feature!(colvecs::Vector{Vector{Float64}}, j::Int, buf::Vector{Float64},
                        feat::AbstractFeature, ctx::FeatureContext)
    featurevalues!(buf, feat, ctx)
    for q in eachindex(buf)
        colvecs[j + q - 1][1] = buf[q]
    end
    return j + length(buf)
end

_feature_buffers(features::FeatureSet) =
    Vector{Float64}[Vector{Float64}(undef, length(outputnames(feat))) for feat in features]

function Base.show(io::IO, fc::Forecaster)
    print(io, "Forecaster(", nameof(typeof(fc.model)), ", ",
          length(fc.features), " features, ", fc.strategy, ", freq=", fc.freq,
          ", target=:", fc.target, ", time=:", fc.time,
          ispanel(fc) ? ", id=:$(fc.id)" : "",
          fc.transform isa NoTransform ? "" : ", transform=$(fc.transform)", ")")
end

function Base.show(io::IO, f::FittedForecaster)
    spec = f.spec
    trained = ispanel(spec) ? "$(nseries(f)) series" :
              "$(f.n_train) rows ending $(f.t_last)"
    print(io, "FittedForecaster(", nameof(typeof(spec.model)), ", ",
          length(spec.features), " features, ", spec.strategy,
          ", freq=", spec.freq, ", trained on ", trained, ")")
end
