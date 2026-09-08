# Feature type hierarchy and the per-feature dispatch contract:
#
#   outputnames(f)   -> Vector{Symbol}: names of the columns this feature produces
#   minhistory(f)    -> Int: minimum history rows needed before the feature is defined
#   materialize!(out, f, y, t, data) -> pushes outputnames(f) columns onto `out`
#                       (vectorized, for training; `missing` where undefined)
#   featurevalues(f, y_hist, t_next, exog_row) -> Tuple of values for one future row
#                       (for recursive/direct forecasting)
#
# Leakage rule: every TargetFeature uses only information strictly before the
# current row's target. Rolling/diff features therefore require `lag ≥ 1`.

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
"""
abstract type AbstractFeature end

"Features computed from the target's history (must be updated during recursion)."
abstract type TargetFeature <: AbstractFeature end

"Features computed from the timestamp only (always computable for future rows)."
abstract type TimeFeature <: AbstractFeature end

"Features passed through from user data (must be provided for the future)."
abstract type ExogenousFeature <: AbstractFeature end

# ---------------------------------------------------------------------------
# Lag
# ---------------------------------------------------------------------------

"""
    Lag(k)

Lagged target `y_{t-k}` as feature column `y_lag_k`. Requires `k ≥ 1`.

# Example
```julia
Lag(7)   # the target seven steps ago, column :y_lag_7
```
"""
struct Lag <: TargetFeature
    k::Int
    Lag(k) = k ≥ 1 ? new(k) :
        throw(ArgumentError("lag must be ≥ 1, got Lag($k). Lag(0) would leak the current target."))
end

outputnames(f::Lag) = [Symbol("y_lag_", f.k)]
minhistory(f::Lag) = f.k

function materialize!(out::ColumnAccumulator, f::Lag, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    n = length(y)
    col = Vector{Union{Missing,Float64}}(missing, n)
    for i in (f.k + 1):n
        col[i] = y[i - f.k]
    end
    push!(out, only(outputnames(f)) => col)
    return out
end

featurevalues(f::Lag, y_hist::AbstractVector, t_next, exog_row) =
    (y_hist[end - f.k + 1],)

# ---------------------------------------------------------------------------
# Rolling statistics: RollingMean, RollingStd, RollingMin, RollingMax
# ---------------------------------------------------------------------------

for (T, stem, fun, minw, example) in (
        (:RollingMean, "rollmean", :(Statistics.mean), 1, "RollingMean(7)        # mean of the 7 values before the current row"),
        (:RollingStd,  "rollstd",  :(Statistics.std),  2, "RollingStd(28)        # std of the 28 values before the current row"),
        (:RollingMin,  "rollmin",  :(minimum),         1, "RollingMin(14)        # min of the 14 values before the current row"),
        (:RollingMax,  "rollmax",  :(maximum),         1, "RollingMax(14)        # max of the 14 values before the current row"))
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
                    string($(string(T)), " window must be ≥ ", $minw, ", got ", window, ".")))
                lag ≥ 1 || throw(ArgumentError(
                    string($(string(T)), " lag must be ≥ 1, got ", lag,
                           ". lag=0 would include the current target (leakage).")))
                new(Int(window), Int(lag))
            end
        end
        $T(window::Integer; lag::Integer=1) = $T(window, lag)
        outputnames(f::$T) = [Symbol("y_", $stem, "_", f.window, "_lag_", f.lag)]
        _rollfun(::$T) = $fun
    end
end

"Union of the rolling-statistic features, for shared method definitions."
const RollingFeature = Union{RollingMean,RollingStd,RollingMin,RollingMax}

minhistory(f::RollingFeature) = f.window + f.lag - 1

function materialize!(out::ColumnAccumulator, f::RollingFeature, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    n = length(y)
    g = _rollfun(f)
    col = Vector{Union{Missing,Float64}}(missing, n)
    for i in (f.window + f.lag):n
        hi = i - f.lag
        col[i] = g(view(y, (hi - f.window + 1):hi))
    end
    push!(out, only(outputnames(f)) => col)
    return out
end

function featurevalues(f::RollingFeature, y_hist::AbstractVector, t_next, exog_row)
    hi = length(y_hist) - f.lag + 1
    return (_rollfun(f)(view(y_hist, (hi - f.window + 1):hi)),)
end

# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

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
        k ≥ 1 || throw(ArgumentError("Diff k must be ≥ 1, got $k."))
        lag ≥ 1 || throw(ArgumentError(
            "Diff lag must be ≥ 1, got $lag. lag=0 would use the current target (leakage)."))
        new(Int(k), Int(lag))
    end
end
Diff(k::Integer; lag::Integer=1) = Diff(k, lag)

outputnames(f::Diff) = [Symbol("y_diff_", f.k, "_lag_", f.lag)]
minhistory(f::Diff) = f.k + f.lag

function materialize!(out::ColumnAccumulator, f::Diff, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    n = length(y)
    col = Vector{Union{Missing,Float64}}(missing, n)
    for i in (f.k + f.lag + 1):n
        col[i] = y[i - f.lag] - y[i - f.lag - f.k]
    end
    push!(out, only(outputnames(f)) => col)
    return out
end

function featurevalues(f::Diff, y_hist::AbstractVector, t_next, exog_row)
    i = length(y_hist) + 1
    return (y_hist[i - f.lag] - y_hist[i - f.lag - f.k],)
end

# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

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
            "unknown Calendar part$(length(bad) == 1 ? "" : "s") $(join(repr.(bad), ", ")); " *
            "valid parts are $(join(repr.(sort!(collect(keys(CALENDAR_PARTS)))), ", "))."))
        new(parts)
    end
end
Calendar(parts::Symbol...) = Calendar(collect(Symbol, parts))

outputnames(f::Calendar) = f.parts
minhistory(::TimeFeature) = 0

function materialize!(out::ColumnAccumulator, f::Calendar, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    for p in f.parts
        if p in (:hour, :minute) && !(eltype(t) <: Dates.AbstractDateTime)
            throw(ArgumentError(
                "Calendar($(repr(p))) needs a sub-daily time column, but the time " *
                "column has element type $(eltype(t)). Use a DateTime time column, " *
                "or drop $(repr(p)) from the Calendar feature."))
        end
        push!(out, p => Vector{Union{Missing,Float64}}(Float64.(CALENDAR_PARTS[p].(t))))
    end
    return out
end

featurevalues(f::Calendar, y_hist::AbstractVector, t_next, exog_row) =
    Tuple(Float64(CALENDAR_PARTS[p](t_next)) for p in f.parts)

# ---------------------------------------------------------------------------
# Fourier
# ---------------------------------------------------------------------------

"""
    Fourier(period, order)

Fourier seasonal terms `sin(2πkn/period)`, `cos(2πkn/period)` for `k = 1:order`,
where `n` is an integer step index (0-based from the first training timestamp,
incremented by 1 per `freq` step). The index continues seamlessly into the
forecast horizon, so there is no phase jump at the train/forecast boundary.

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
        period > 0 || throw(ArgumentError("Fourier period must be > 0, got $period."))
        order ≥ 1 || throw(ArgumentError("Fourier order must be ≥ 1, got $order."))
        new(Float64(period), Int(order))
    end
end

function outputnames(f::Fourier)
    p = replace(string(f.period), "." => "_")
    return vcat([Symbol("fourier_", p, "_sin_", k) for k in 1:f.order],
                [Symbol("fourier_", p, "_cos_", k) for k in 1:f.order])
end

# Fourier's featurevalues receives the integer step index as `t_next`
# (see `uses_step_index`), not the timestamp.
_uses_step_index(::AbstractFeature) = false
_uses_step_index(::Fourier) = true

function materialize!(out::ColumnAccumulator, f::Fourier, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    idx = 0:(length(y) - 1)
    names = outputnames(f)
    sincols = [names[k] => Vector{Union{Missing,Float64}}(sin.(2π .* k .* idx ./ f.period))
               for k in 1:f.order]
    coscols = [names[f.order + k] => Vector{Union{Missing,Float64}}(cos.(2π .* k .* idx ./ f.period))
               for k in 1:f.order]
    append!(out, sincols)
    append!(out, coscols)
    return out
end

function featurevalues(f::Fourier, y_hist::AbstractVector, n::Integer, exog_row)
    sins = ntuple(k -> sin(2π * k * n / f.period), f.order)
    coss = ntuple(k -> cos(2π * k * n / f.period), f.order)
    return (sins..., coss...)
end

# ---------------------------------------------------------------------------
# Exogenous
# ---------------------------------------------------------------------------

"""
    Exogenous(cols::Symbol...)

Pass-through of user-supplied covariate columns that are known in the future
(e.g. promotions, prices, holidays). At `forecast` time these columns must be
supplied for the full horizon via the `new_data` keyword; in `backtest` they
are sliced from the held-out data automatically. In v1, exogenous columns must
be numeric (Real or Bool).

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
        new(cols)
    end
end
Exogenous(cols::Symbol...) = Exogenous(collect(Symbol, cols))

outputnames(f::Exogenous) = f.cols
minhistory(::ExogenousFeature) = 0

function materialize!(out::ColumnAccumulator, f::Exogenous, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    for c in f.cols
        haskey(data, c) || throw(ArgumentError(
            "features contain Exogenous(:$c) but column :$c is not present in the " *
            "training data. Available columns: $(join(keys(data), ", "))."))
        raw = data[c]
        vals = Vector{Union{Missing,Float64}}(missing, length(raw))
        for i in eachindex(raw)
            ismissing(raw[i]) || (vals[i] = _tofloat(raw[i], c))
        end
        push!(out, c => vals)
    end
    return out
end

function _tofloat(v, name::Symbol)
    v isa Real && return Float64(v)
    throw(ArgumentError(
        "exogenous column :$name has non-numeric value $(repr(v)) " *
        "(type $(typeof(v))). In v1 exogenous columns must be numeric (Real or " *
        "Bool). Encode categorical columns numerically first."))
end

function featurevalues(f::Exogenous, y_hist::AbstractVector, t_next, exog_row)
    return Tuple(begin
        haskey(exog_row, c) || throw(ArgumentError(
            "new_data is missing exogenous column :$c required by Exogenous. " *
            "Provide it for every forecast timestamp."))
        _tofloat(exog_row[c], c)
    end for c in f.cols)
end

# ---------------------------------------------------------------------------
# CustomFeature
# ---------------------------------------------------------------------------

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

# Example
```julia
# mean of the entire history so far (needs at least 1 observation)
CustomFeature(:hist_mean, mean, 1)
```
"""
struct CustomFeature <: TargetFeature
    name::Symbol
    f::Function
    minhistory::Int
    function CustomFeature(name::Symbol, f::Function, minhistory::Integer)
        minhistory ≥ 0 || throw(ArgumentError(
            "CustomFeature minhistory must be ≥ 0, got $minhistory."))
        new(name, f, Int(minhistory))
    end
end

outputnames(f::CustomFeature) = [f.name]
minhistory(f::CustomFeature) = f.minhistory

function materialize!(out::ColumnAccumulator, f::CustomFeature, y::AbstractVector{Float64},
                      t::AbstractVector, data::NamedTuple)
    n = length(y)
    col = Vector{Union{Missing,Float64}}(missing, n)
    for i in (f.minhistory + 1):n
        col[i] = Float64(f.f(view(y, 1:(i - 1))))
    end
    push!(out, f.name => col)
    return out
end

featurevalues(f::CustomFeature, y_hist::AbstractVector, t_next, exog_row) =
    (Float64(f.f(y_hist)),)

# ---------------------------------------------------------------------------
# Equality and hashing: features are value objects — two features with the same
# type and parameters are equal (useful in tests, tuning tables, and Dicts).
# ---------------------------------------------------------------------------

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
