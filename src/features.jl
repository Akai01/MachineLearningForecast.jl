"Ordered accumulator of materialized feature columns (name => column pairs)."
const ColumnAccumulator = Vector{Pair{Symbol,Vector{Union{Missing,Float64}}}}

"""
    AbstractFeature

Root of the feature hierarchy. Concrete features are one of:

- [`TargetFeature`](@ref): computed from the target's history (updated during
  recursive forecasting), e.g. [`Lag`](@ref), [`RollingMean`](@ref).
- [`TimeFeature`](@ref): computed from the timestamp only (always computable
  for future rows), e.g. [`Calendar`](@ref), [`Fourier`](@ref).
- [`ExogenousFeature`](@ref): passed through from user data (must be provided
  for the future via `new_data`), i.e. [`Exogenous`](@ref).

Every concrete feature implements three methods:

- `outputnames(f)`: the names of the columns it produces, a `Vector{Symbol}`.
- `minhistory(f)`: the number of history rows it needs before it is defined.
- [`featurevalues!`](@ref)`(out, f, ctx)`: writes its values for the one row
  that the [`FeatureContext`](@ref) `ctx` describes.

Training, `forecast` and `backtest` all compute a feature through
`featurevalues!`, so by default (unless the feature overrides `materialize!`) the
training columns and the forecast-time values cannot disagree. Optionally a
feature defines [`materialize!`](@ref) (a faster way to fill its training
columns), [`inputcolumns`](@ref) (the user columns it reads; an
[`ExogenousFeature`](@ref) must) and [`anchor`](@ref) (which row its columns
describe under `Direct`; a feature that subtypes `AbstractFeature` directly must).

`ctx.y` holds the target history strictly before the row's target: a
`TargetFeature` never sees the current or a later target value, so rolling and
difference features require `lag ≥ 1`. Under `Direct` the history is fixed at
the forecast origin. An `AtOrigin` feature (every `TargetFeature` by default)
should read only `ctx.y`: under `Direct` its training column comes from the
feature row, while `ctx.t` and `ctx.step` describe the feature row in training
and the target row when forecasting. See [`anchor`](@ref).
"""
abstract type AbstractFeature end

"Features computed from the target's history (must be updated during recursion)."
abstract type TargetFeature <: AbstractFeature end

"Features computed from the timestamp only (always computable for future rows)."
abstract type TimeFeature <: AbstractFeature end

"Features passed through from user data (must be provided for the future)."
abstract type ExogenousFeature <: AbstractFeature end

"""
    FeatureContext(y, t, step, exog)

The row a feature is asked about, passed to [`featurevalues!`](@ref). It is
read-only: do not keep a reference to it or to `y`.

- `y`: the target history strictly before the row's target (a `Vector` when
  forecasting, a view in training). It is empty at the first row when
  `minhistory(f) == 0`.
- `t`: the timestamp of the row's target.
- `step`: the 0-based step index of the row's target. It counts `freq` steps
  from the first (transformed) training timestamp, and from the earliest usable
  series' first (transformed) timestamp on a panel.
- `exog`: a `NamedTuple` of the user columns at the row (it may hold more than
  `inputcolumns(f)` when forecasting, so index it by name). When `inputcolumns(f)` is
  empty it may be `nothing` or a `NamedTuple` (forecasting with another exogenous
  feature): do not read it then.

# Example
```julia
using Dates
ctx = MachineLearningForecast.FeatureContext([1.0, 2.0, 3.0], Date(2022, 1, 4), 3, nothing)
ctx.y[end]   # 3.0
```
"""
struct FeatureContext{Y<:AbstractVector{Float64},T,E}
    y::Y
    t::T
    step::Int
    exog::E
end

"""
    TrainingSeries(y, t, data, step0)

What [`materialize!`](@ref) reads: one series' training columns. `y` is the
(transformed) target, `t` the time column, `data` the whole columntable (for
the columns named by [`inputcolumns`](@ref)) and `step0` the step index of row
1, so that row `i` has `ctx.step == step0 + i - 1`.

# Example
```julia
using Dates
s = MachineLearningForecast.TrainingSeries([1.0, 2.0], [Date(2022, 1, 1), Date(2022, 1, 2)],
                                           (;), 0)
s.step0   # 0
```
"""
struct TrainingSeries{T<:AbstractVector,D<:NamedTuple}
    y::Vector{Float64}
    t::T
    data::D
    step0::Int
end

"""
    featurevalues!(out, f, ctx)

Write the values of feature `f` for the row that the
[`FeatureContext`](@ref) `ctx` describes into `out`, a `Vector{Float64}` of
`length(outputnames(f))` entries (write every one of them); the return value
is ignored. This is the one method a feature must implement for its values:
training, `forecast` and `backtest` all call it, so by default (unless the feature
overrides [`materialize!`](@ref)) they cannot disagree.

# Example
```julia
import MachineLearningForecast: outputnames, minhistory, featurevalues!

struct LastTwo <: MachineLearningForecast.TargetFeature end
outputnames(::LastTwo) = [:last_two]
minhistory(::LastTwo) = 2
featurevalues!(out, ::LastTwo, ctx) = (out[1] = ctx.y[end] + ctx.y[end-1]; nothing)
```
"""
function featurevalues!(out, f::AbstractFeature, ctx)
    throw(ArgumentError(
        "$(typeof(f)) does not implement featurevalues!(out, f, ctx). The feature " *
        "contract changed from featurevalues(f, y_hist, t_next, exog_row) and a " *
        "vectorised materialize!(out, f, y, t, data) to one method that writes the " *
        "values of the row ctx describes; define MachineLearningForecast." *
        "featurevalues!(out, f::$(nameof(typeof(f))), ctx). See CONTRIBUTING.md."))
end

"""
    materialize!(out, f, s::TrainingSeries)

Push the training columns of feature `f` over the series `s` onto `out`
(a vector of `name => column` pairs), with `missing` in the first
`minhistory(f)` rows. The default calls [`featurevalues!`](@ref) for every row
from `minhistory(f) + 1` on, so a feature gets this for free; it leaves a row
`missing` when one of its [`inputcolumns`](@ref) is `missing` there. Define a
method only for speed or to validate a column before any row is read, as
[`Calendar`](@ref), [`Fourier`](@ref) and [`Exogenous`](@ref) do.

# Example
```julia
using MachineLearningForecast
using MachineLearningForecast: ColumnAccumulator, TrainingSeries, materialize!
out = ColumnAccumulator()
materialize!(out, Lag(1), TrainingSeries([1.0, 2.0, 3.0], 1:3, (;), 0))
out   # [:y_lag_1 => [missing, 1.0, 2.0]]
```
"""
function materialize!(out, f::AbstractFeature, s::TrainingSeries)
    cols = inputcolumns(f)
    for c in cols
        _require_input_column(s.data, c)
    end
    columns = [Vector{Union{Missing,Float64}}(missing, length(s.y))
               for _ in outputnames(f)]
    source = isempty(cols) ? nothing :
             NamedTuple{Tuple(cols)}(Tuple(s.data[c] for c in cols))
    _materialize_rows!(columns, f, s, source)
    for (name, col) in zip(outputnames(f), columns)
        push!(out, name => col)
    end
    return out
end

function _require_input_column(data::NamedTuple, c::Symbol)
    haskey(data, c) || throw(ArgumentError(
        "features contain Exogenous(:$c) but column :$c is not present in the " *
        "training data. Available columns: $(join(keys(data), ", ")). Add " *
        "column :$c to the data, or remove it from the Exogenous feature."))
    return nothing
end

# Barriers keep the context type concrete in the row loops.
function _materialize_rows!(columns, f, s::TrainingSeries, ::Nothing)
    buf = Vector{Float64}(undef, length(columns))
    for i in (minhistory(f) + 1):length(s.y)
        ctx = FeatureContext(view(s.y, 1:(i - 1)), s.t[i], s.step0 + i - 1, nothing)
        _store_row!(columns, buf, f, ctx, i)
    end
    return nothing
end

function _materialize_rows!(columns, f, s::TrainingSeries, source::NamedTuple)
    buf = Vector{Float64}(undef, length(columns))
    for i in (minhistory(f) + 1):length(s.y)
        exog = map(col -> col[i], source)
        any(ismissing, values(exog)) && continue
        ctx = FeatureContext(view(s.y, 1:(i - 1)), s.t[i], s.step0 + i - 1, exog)
        _store_row!(columns, buf, f, ctx, i)
    end
    return nothing
end

function _store_row!(columns, buf, f, ctx, i)
    featurevalues!(buf, f, ctx)
    for q in eachindex(buf)
        columns[q][i] = buf[q]
    end
    return nothing
end

"""
    Lag(k)

Lagged target `y_{t-k}` as feature column `y_lag_k`. Requires a whole number `k ≥ 1`
(a `Bool` is refused).

Under [`Direct`](@ref), `Lag(k)` counts back from the forecast origin, so at horizon `h`
it is `y[t0-k+1]` (`t0` is the index of the last observed value), not `y[target-k]`;
`Lag(m)` is in phase with the target only when `h = 1 (mod m)`; use `Lag.(1:m)`, which
holds an in-phase lag at every `h`, or [`Recursive`](@ref).

# Example
```julia
Lag(7)   # the target seven steps ago, column :y_lag_7
```
"""
struct Lag <: TargetFeature
    k::Int
    function Lag(k)
        k isa Real && !(k isa Bool) && isinteger(k) || throw(ArgumentError(
            "lag must be a whole number of steps, got Lag($(repr(k))). Use e.g. Lag(7)."))
        k ≥ 1 || throw(ArgumentError(
            "lag must be ≥ 1, got Lag($k), which would leak the current or a future " *
            "target. Use Lag(1) or a larger lag."))
        new(k)
    end
end

outputnames(f::Lag) = [Symbol("y_lag_", f.k)]
minhistory(f::Lag) = f.k

function featurevalues!(out::AbstractVector{Float64}, f::Lag, ctx::FeatureContext)
    out[1] = ctx.y[end - f.k + 1]
    return nothing
end

for (T, stem, fun, minw, example) in (
        (:RollingMean, "rollmean", :(Statistics.mean), 1,
         "RollingMean(7)        # mean of the 7 values before the current row"),
        (:RollingStd,  "rollstd",  :(Statistics.std),  2,
         "RollingStd(28)        # std of the 28 values before the current row"),
        (:RollingMin,  "rollmin",  :(minimum),         1,
         "RollingMin(14)        # min of the 14 values before the current row"),
        (:RollingMax,  "rollmax",  :(maximum),         1,
         "RollingMax(14)        # max of the 14 values before the current row"))
    docstr = """
        $T(window; lag=1)

    Rolling statistic of the target over `window` values ending `lag` steps
    before the current row, as column `y_$(stem)_<window>_lag_<lag>`. The shift
    `lag` is applied *before* the window, so with the default `lag=1` the window
    at row `t` covers `y[t-lag-window+1 : t-lag]` and never includes `y[t]`
    (no leakage). Requires `window ≥ $minw` and `lag ≥ 1`.

    # Example
    ```julia
    $example
    ```
    """
    @eval begin
        @doc $docstr
        struct $T <: TargetFeature
            window::Int
            lag::Int
            function $T(window::Integer, lag::Integer)
                window ≥ $minw || throw(ArgumentError(
                    string($(string(T)), " window must be ≥ ", $minw, ", got ", window,
                           ". Use e.g. ", $(string(T)), "(7).")))
                lag ≥ 1 || throw(ArgumentError(
                    string($(string(T)), " lag must be ≥ 1, got ", lag, ", which ",
                           "would include the current or a future target (leakage). ",
                           "Use lag=1, the default.")))
                new(Int(window), Int(lag))
            end
        end
        $T(window::Integer; lag::Integer=1) = $T(window, lag)
        outputnames(f::$T) = [Symbol("y_", $stem, "_", f.window, "_lag_", f.lag)]
        _roll_fun(::$T) = $fun
    end
end

"Union of the rolling-statistic features, for shared method definitions."
const RollingFeature = Union{RollingMean,RollingStd,RollingMin,RollingMax}

minhistory(f::RollingFeature) = f.window + f.lag - 1

function featurevalues!(out::AbstractVector{Float64}, f::RollingFeature,
                        ctx::FeatureContext)
    hi = length(ctx.y) - f.lag + 1
    out[1] = _roll_fun(f)(view(ctx.y, (hi - f.window + 1):hi))
    return nothing
end

"""
    Diff(k; lag=1)

Lagged difference of the target, `y_{t-lag} - y_{t-lag-k}`, as column
`y_diff_<k>_lag_<lag>`. Requires `k ≥ 1` and `lag ≥ 1`.

# Example
```julia
Diff(1)       # first difference of yesterday: y_{t-1} - y_{t-2}
Diff(7)       # week-over-week change as of yesterday: y_{t-1} - y_{t-8}
```
"""
struct Diff <: TargetFeature
    k::Int
    lag::Int
    function Diff(k::Integer, lag::Integer)
        k ≥ 1 || throw(ArgumentError(
            "Diff k must be ≥ 1, got $k. Use e.g. Diff(1) for the first difference."))
        lag ≥ 1 || throw(ArgumentError(
            "Diff lag must be ≥ 1, got $lag, which would use the current or a future " *
            "target (leakage). Use lag=1, the default."))
        new(Int(k), Int(lag))
    end
end
Diff(k::Integer; lag::Integer=1) = Diff(k, lag)

outputnames(f::Diff) = [Symbol("y_diff_", f.k, "_lag_", f.lag)]
minhistory(f::Diff) = f.k + f.lag

function featurevalues!(out::AbstractVector{Float64}, f::Diff, ctx::FeatureContext)
    i = length(ctx.y) + 1
    out[1] = ctx.y[i - f.lag] - ctx.y[i - f.lag - f.k]
    return nothing
end

const CALENDAR_PARTS = Dict{Symbol,Function}(
    :year       => Dates.year,
    :quarter    => Dates.quarterofyear,
    :month      => Dates.month,
    :weekofyear => Dates.week,
    :dayofweek  => Dates.dayofweek,
    :dayofmonth => Dates.dayofmonth,
    :dayofyear  => Dates.dayofyear,
    :hour       => Dates.hour,
    :minute     => Dates.minute,
)

"""
    Calendar(parts::Symbol...)

Calendar features extracted from the time column, one column per part. Valid
parts: `:year`, `:quarter`, `:month`, `:weekofyear`, `:dayofweek`,
`:dayofmonth`, `:dayofyear`, `:hour`, `:minute`.

# Example
```julia
Calendar(:dayofweek, :month)   # columns :dayofweek and :month
```
"""
struct Calendar <: TimeFeature
    parts::Vector{Symbol}
    function Calendar(parts::Vector{Symbol})
        isempty(parts) && throw(ArgumentError(
            "Calendar needs at least one part, e.g. Calendar(:dayofweek)."))
        bad = setdiff(parts, keys(CALENDAR_PARTS))
        isempty(bad) || throw(ArgumentError(
            "unknown Calendar part$(length(bad) == 1 ? "" : "s") " *
            "$(join(repr.(bad), ", ")); " *
            "valid parts are $(join(repr.(sort!(collect(keys(CALENDAR_PARTS)))), ", "))."))
        new(copy(parts))
    end
end
Calendar(parts::Symbol...) = Calendar(collect(Symbol, parts))

outputnames(f::Calendar) = f.parts
minhistory(::TimeFeature) = 0

function materialize!(out::ColumnAccumulator, f::Calendar, s::TrainingSeries)
    t = s.t
    for p in f.parts
        if p in (:hour, :minute)
            i = findfirst(v -> !(v isa Dates.AbstractDateTime), t)
            i === nothing || throw(ArgumentError(
                "Calendar($(repr(p))) needs a sub-daily time column, but the time " *
                "column has element type $(typeof(t[i])). Use a DateTime time column, " *
                "or drop $(repr(p)) from the Calendar feature."))
        end
        push!(out, p => Vector{Union{Missing,Float64}}(Float64.(CALENDAR_PARTS[p].(t))))
    end
    return out
end

function featurevalues!(out::AbstractVector{Float64}, f::Calendar, ctx::FeatureContext)
    for (j, p) in enumerate(f.parts)
        out[j] = Float64(CALENDAR_PARTS[p](ctx.t))::Float64
    end
    return nothing
end

"""
    Fourier(period, order)

Fourier seasonal terms `sin(2πkn/period)`, `cos(2πkn/period)` for `k = 1:order`,
where `n` is an integer step index (0-based from the first (transformed) training timestamp,
incremented by 1 per `freq` step). The index continues seamlessly into the
forecast horizon, so there is no phase jump at the train/forecast boundary. In a
panel, `n` counts from the earliest usable series' first (transformed) training
timestamp, so every series has the same Fourier values on the same date.

`period` is expressed in units of `freq` steps, e.g. `365.25` for annual
seasonality with daily data. `order` K produces 2K columns
(`fourier_<period>_sin_1..K`, then `fourier_<period>_cos_1..K`).

# Example
```julia
Fourier(365.25, 3)   # annual seasonality, 3 harmonics, for daily data
Fourier(7, 2)        # weekly seasonality, 2 harmonics, for daily data
```
"""
struct Fourier <: TimeFeature
    period::Float64
    order::Int
    function Fourier(period::Real, order::Integer)
        period > 0 || throw(ArgumentError(
            "Fourier period must be > 0 (in freq steps), got $period. " *
            "Use e.g. Fourier(7, 2) for weekly seasonality on daily data."))
        order ≥ 1 || throw(ArgumentError(
            "Fourier order must be ≥ 1, got $order. Use e.g. Fourier($period, 2) for " *
            "two harmonics."))
        new(Float64(period), Int(order))
    end
end

function outputnames(f::Fourier)
    p = replace(string(f.period), "." => "_")
    return vcat([Symbol("fourier_", p, "_sin_", k) for k in 1:f.order],
                [Symbol("fourier_", p, "_cos_", k) for k in 1:f.order])
end

# Reducing k * n first makes whole periods repeat bit for bit.
_fourier_angle(f::Fourier, k::Integer, n::Integer) = 2π * mod(k * n, f.period) / f.period

function materialize!(out::ColumnAccumulator, f::Fourier, s::TrainingSeries)
    idx = s.step0:(s.step0 + length(s.y) - 1)
    names = outputnames(f)
    θ = [_fourier_angle.(Ref(f), k, idx) for k in 1:f.order]
    append!(out, [names[k] => Vector{Union{Missing,Float64}}(sin.(θ[k]))
                  for k in 1:f.order])
    append!(out, [names[f.order + k] => Vector{Union{Missing,Float64}}(cos.(θ[k]))
                  for k in 1:f.order])
    return out
end

function featurevalues!(out::AbstractVector{Float64}, f::Fourier, ctx::FeatureContext)
    n = ctx.step
    for k in 1:f.order
        θ = _fourier_angle(f, k, n)
        out[k] = sin(θ)
        out[f.order + k] = cos(θ)
    end
    return nothing
end

"""
    Exogenous(cols::Symbol...)

Pass-through of user-supplied covariate columns that are known in the future
(e.g. promotions, prices, holidays). At `forecast` time these columns must be
supplied for the full horizon via the `new_data` keyword; in `backtest` they
are sliced from the held-out data automatically. Exogenous columns must be
numeric (`Real` or `Bool`).

A backtest feeds the realised values of the covariates, so its errors are a best case (too
low) unless the column holds the values that were actually known at the forecast origin.

# Example
```julia
Exogenous(:promo, :price)
```
"""
struct Exogenous <: ExogenousFeature
    cols::Vector{Symbol}
    function Exogenous(cols::Vector{Symbol})
        isempty(cols) && throw(ArgumentError(
            "Exogenous needs at least one column, e.g. Exogenous(:promo)."))
        new(copy(cols))
    end
end
Exogenous(cols::Symbol...) = Exogenous(collect(Symbol, cols))

outputnames(f::Exogenous) = f.cols
minhistory(::ExogenousFeature) = 0

"""
    inputcolumns(f) -> Vector{Symbol}

The user columns the feature `f` reads. `forecast` and `backtest` pick exactly these columns
from `new_data` (or from the held-out rows), and training reads them from the training data
too, so they must be present there: `fit` throws otherwise (unless the feature overrides
`materialize!`). The default is `Symbol[]`, [`Exogenous`](@ref) returns its columns, and an
[`ExogenousFeature`](@ref) of your own must define it, otherwise an `ArgumentError` is
thrown. Any feature may define it.

# Example
```julia
inputcolumns(Exogenous(:promo, :price))   # [:promo, :price]
inputcolumns(Lag(1))                      # Symbol[]
```
"""
inputcolumns(::AbstractFeature) = Symbol[]
inputcolumns(f::Exogenous) = f.cols
inputcolumns(f::ExogenousFeature) = throw(ArgumentError(
    "$(typeof(f)) is an ExogenousFeature but defines no inputcolumns method. " *
    "forecast and backtest read the columns it needs from new_data through " *
    "inputcolumns(f). Define MachineLearningForecast.inputcolumns(f::" *
    "$(nameof(typeof(f)))) = [...], a Vector{Symbol}."))

"""
    Anchor

Abstract supertype of [`AtOrigin`](@ref) and [`AtTarget`](@ref): which row a
feature's columns describe. See [`anchor`](@ref).

# Example
```julia
anchor(Lag(1)) isa Anchor   # true
```
"""
abstract type Anchor end

"""
    AtOrigin()

The [`anchor`](@ref) of a feature whose columns describe the feature row, that
is the target history up to the forecast origin (lags, rolling statistics).

# Example
```julia
anchor(Lag(1))   # AtOrigin()
```
"""
struct AtOrigin <: Anchor end

"""
    AtTarget()

The [`anchor`](@ref) of a feature whose columns describe the row being
predicted (calendar, Fourier and exogenous columns).

# Example
```julia
anchor(Calendar(:month))   # AtTarget()
```
"""
struct AtTarget <: Anchor end

"""
    anchor(f) -> Anchor

Which row the columns of feature `f` describe, [`AtOrigin`](@ref) or
[`AtTarget`](@ref). [`Direct`](@ref) fitting needs it: model `i` pairs the
target `i` steps ahead with the `AtOrigin` columns of the feature row and with
the `AtTarget` columns of the target row, which is what `forecast` feeds. A
[`TargetFeature`](@ref) is `AtOrigin`; a [`TimeFeature`](@ref) or an
[`ExogenousFeature`](@ref) is `AtTarget`. A feature of any other kind must
define it, otherwise an `ArgumentError` is thrown when `Direct` fits.

An `AtOrigin` feature may read only `ctx.y`: its training value comes from the
feature row and its forecast value from the origin, while `ctx.t`, `ctx.step`
and `ctx.exog` describe the context's row (the feature row in training, the
target row when forecasting), so reading them would train on one thing and
serve another. An `AtTarget` feature should not read `ctx.y`: under `Direct` it
is trained on history up to the target row but served the history at the
forecast origin. A feature that reads `ctx.t`, `ctx.step` or `ctx.exog` and not
`ctx.y` is `AtTarget`: subtype `TimeFeature` or `ExogenousFeature`, or define
`anchor(f) = AtTarget()`; one that reads both is correct only under `Recursive`.

# Example
```julia
anchor(Lag(1))             # AtOrigin()
anchor(Calendar(:month))   # AtTarget()
```
"""
anchor(::TargetFeature) = AtOrigin()
anchor(::TimeFeature) = AtTarget()
anchor(::ExogenousFeature) = AtTarget()
anchor(f::AbstractFeature) = throw(ArgumentError(
    "$(typeof(f)) has no anchor. Direct fitting must know whether its columns " *
    "describe the feature row (target history) or the target row (time, " *
    "exogenous). Subtype TargetFeature, TimeFeature or ExogenousFeature, or define " *
    "MachineLearningForecast.anchor(f::$(nameof(typeof(f)))) = " *
    "MachineLearningForecast.AtOrigin() or MachineLearningForecast.AtTarget()."))

function materialize!(out::ColumnAccumulator, f::Exogenous, s::TrainingSeries)
    for c in f.cols
        _require_input_column(s.data, c)
        push!(out, c => _exog_column(s.data[c], c))
    end
    return out
end

# A function barrier: data[c] is only known at run time.
function _exog_column(raw, c::Symbol)
    vals = Vector{Union{Missing,Float64}}(missing, length(raw))
    for i in eachindex(raw)
        ismissing(raw[i]) || (vals[i] = _to_float(raw[i], c))
    end
    return vals
end

function _to_float(v, name::Symbol)
    v isa Real && return Float64(v)
    throw(ArgumentError(
        "exogenous column :$name has non-numeric value $(repr(v)) " *
        "(type $(typeof(v))). Exogenous columns must be numeric (Real or " *
        "Bool). Encode categorical columns numerically first."))
end

function featurevalues!(out::AbstractVector{Float64}, f::Exogenous, ctx::FeatureContext)
    for (j, c) in enumerate(f.cols)
        haskey(ctx.exog, c) || throw(ArgumentError(
            "new_data is missing exogenous column :$c required by Exogenous. " *
            "Provide it for every forecast timestamp."))
        out[j] = _to_float(ctx.exog[c], c)
    end
    return nothing
end

"""
    CustomFeature(name, f, minhistory)

A user-defined target-history feature. `f` is a function of the target history
vector *up to and excluding* the current row, returning one scalar (the value
of the feature for that row). `minhistory` is the minimum number of history
values `f` needs; rows with less history get `missing` during training and are
dropped.

During recursive forecasting, `f` is called on the growing history vector
(actuals followed by earlier predictions), so it sees exactly the same kind of
input as during training.

A function that reads anything but its argument, such as a captured series or `parent(h)`,
bypasses the leakage guarantee: `f` must read only the history it is given.

In training `f` is called once per row on the history before it, so a function
that loops over the whole history makes `fit` O(n^2) in the series length; keep
it cheap or use a built-in rolling feature.

# Example
```julia
using Statistics
# mean of the entire history so far (needs at least 1 observation)
CustomFeature(:hist_mean, mean, 1)
```
"""
struct CustomFeature{F<:Function} <: TargetFeature
    name::Symbol
    f::F
    minhistory::Int
    function CustomFeature(name::Symbol, f::F, minhistory::Integer) where {F<:Function}
        minhistory ≥ 0 || throw(ArgumentError(
            "CustomFeature minhistory must be ≥ 0, got $minhistory. Pass the number " *
            "of history values f needs, or 0 if it needs none."))
        new{F}(name, f, Int(minhistory))
    end
end

# The default show would print the type parameter.
function Base.show(io::IO, f::CustomFeature)
    show(io, CustomFeature)
    print(io, "(")
    show(io, f.name)
    print(io, ", ")
    show(io, f.f)
    print(io, ", ")
    show(io, f.minhistory)
    print(io, ")")
end

outputnames(f::CustomFeature) = [f.name]
minhistory(f::CustomFeature) = f.minhistory

function featurevalues!(out::AbstractVector{Float64}, f::CustomFeature,
                        ctx::FeatureContext)
    out[1] = _custom_value(f, ctx.y, ctx.t)
    return nothing
end

function _custom_value(f::CustomFeature, h::AbstractVector, t)
    v = try
        f.f(h)
    catch err
        err isa BoundsError && isdefined(err, :a) && err.a === h || rethrow()
        throw(ArgumentError(
            "CustomFeature(:$(f.name)) indexed past its history at time $t, where f " *
            "sees $(_nvalues(h)): $(sprint(showerror, err)). If f needs more values, " *
            "raise minhistory (now $(f.minhistory)) to that number."))
    end
    # isa Real is the fast path; applicable keeps the rest.
    v isa Real || applicable(Float64, v) || throw(ArgumentError(
        "CustomFeature(:$(f.name)) returned $(repr(v)) at time $t, where f sees " *
        "$(_nvalues(h)), but f must return one real number, e.g. mean(h)."))
    return Float64(v)
end

_nvalues(h) = string(length(h), " history value", length(h) == 1 ? "" : "s")

# Features are value objects: equal params, equal hash.

function Base.:(==)(a::AbstractFeature, b::AbstractFeature)
    typeof(a) === typeof(b) || return false
    return all(getfield(a, i) == getfield(b, i) for i in 1:nfields(a))
end

function Base.hash(f::AbstractFeature, h::UInt)
    h = hash(typeof(f), h)
    for i in 1:nfields(f)
        h = hash(getfield(f, i), h)
    end
    return h
end
