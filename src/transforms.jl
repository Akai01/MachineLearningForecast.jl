"""
    TargetTransform

Root of the per-series target transforms. A transform maps one series' training
target `y` to the series `z` that the model learns and forecasts, and maps the
forecasts of `z` back. Its state is fitted on that series' own training history
only, so it never sees the forecast window, and [`backtest`](@ref) refits it in
every fold.

A transform implements two methods:

```julia
fit_transform(t, y)    # -> (z, state); z may be shorter, aligned to y's end
untransform(t, ẑ, state)   # -> the forecasts of y for the forecasts ẑ of z
```

Features are computed on `z`, and the first `length(y) - length(z)` training
rows are dropped. Pass a transform as `Forecaster(...; transform=...)`.

# Example
```julia
fc = Forecaster(model; features=FeatureSet(Lag.(1:12)...), freq=Month(1),
                transform=TransformChain(LastScale(12), Deseasonalize(12),
                                         LogRatio(12)))
```
"""
abstract type TargetTransform end

"""
    fit_transform(t::TargetTransform, y::AbstractVector{Float64}) -> (z, state)

Fit transform `t` on one series' training target `y` and return the transformed
series `z` (possibly shorter, aligned to the end of `y`) and the state that
[`untransform`](@ref) needs. Part of the [`TargetTransform`](@ref) contract.

# Example
```julia
z, state = fit_transform(LastScale(3), [2.0, 4.0, 6.0, 8.0])   # z = y ./ 6
```
"""
function fit_transform end

"""
    untransform(t::TargetTransform, ẑ::AbstractVector{Float64}, state) -> Vector{Float64}

Map the forecasts `ẑ` of the transformed series back to the original scale,
step 1 first. Part of the [`TargetTransform`](@ref) contract.

# Example
```julia
z, state = fit_transform(LastScale(3), [2.0, 4.0, 6.0, 8.0])
untransform(LastScale(3), [1.5, 1.5], state)   # [9.0, 9.0]
```
"""
function untransform end

"""
    NoTransform()

The identity transform, and the default: the model learns the target as given.

# Example
```julia
Forecaster(model; features=FeatureSet(Lag(1)), freq=Day(1), transform=NoTransform())
```
"""
struct NoTransform <: TargetTransform end

fit_transform(::NoTransform, y::AbstractVector{Float64}) = (y, nothing)
untransform(::NoTransform, ẑ::AbstractVector{Float64}, ::Nothing) = collect(ẑ)

function _scale_state(t, m::Float64, what::AbstractString)
    isfinite(m) && !iszero(m) || throw(ArgumentError(
        "$t cannot scale a series whose $what is $m. Use a transform that does " *
        "not divide by it, e.g. NoTransform() or Difference(1)."))
    return m
end

"""
    MeanScale()

Divide each series by the mean of its whole training history, and multiply the
forecasts back. Puts series of different levels on one scale for a global model.

# Example
```julia
Forecaster(model; features=FeatureSet(Lag(1)), freq=Day(1), id=:unique_id,
           transform=MeanScale())
```
"""
struct MeanScale <: TargetTransform end

function fit_transform(t::MeanScale, y::AbstractVector{Float64})
    m = _scale_state(t, Statistics.mean(y), "training mean")
    return y ./ m, m
end

"""
    LastScale(k)

Divide each series by the mean of its last `k` training values (fewer if the
series is shorter), and multiply the forecasts back. Unlike [`MeanScale`](@ref)
it scales by the recent level, so a trending series starts its forecast near 1,
inside the range a tree model has seen. Requires `k ≥ 1`.

# Example
```julia
LastScale(12)   # monthly data: scale by the last year's mean
```
"""
struct LastScale <: TargetTransform
    k::Int
    function LastScale(k::Integer)
        k ≥ 1 || throw(ArgumentError(
            "LastScale needs k ≥ 1 values to average, got $k. Pass e.g. LastScale(12)."))
        return new(Int(k))
    end
end

function fit_transform(t::LastScale, y::AbstractVector{Float64})
    m = _scale_state(t, Statistics.mean(y[max(1, end - t.k + 1):end]),
                     "mean of its last $(t.k) training values")
    return y ./ m, m
end

untransform(::Union{MeanScale,LastScale}, ẑ::AbstractVector{Float64}, m::Float64) =
    ẑ .* m

"""
    Deseasonalize(period)

Classical multiplicative seasonal adjustment, applied only to series that pass
the M4 competition's seasonality test: the autocorrelation at lag `period` must
exceed its 90% bound, which needs at least `3 * period` values. A seasonal
series is divided by seasonal indices: centred moving-average ratios averaged
by position in the season and normalised to mean 1. The forecasts are
multiplied by the index of their own position. Needs positive values when a
series is adjusted, and `period ≥ 2`.

# Example
```julia
Deseasonalize(12)   # monthly data
```
"""
struct Deseasonalize <: TargetTransform
    period::Int
    function Deseasonalize(period::Integer)
        period ≥ 2 || throw(ArgumentError(
            "Deseasonalize needs a period of at least 2, got $period. Pass the " *
            "season length in steps, e.g. Deseasonalize(12) for monthly data."))
        return new(Int(period))
    end
end

_acf(x, k) = (d = x .- Statistics.mean(x);
              sum(d[(k + 1):end] .* d[1:(end - k)]) / sum(abs2, d))

function _isseasonal(x, p)
    n = length(x)
    n < 3p && return false
    bound = 1.645 * sqrt((1 + 2 * sum(_acf(x, k)^2 for k in 1:(p - 1))) / n)
    return abs(_acf(x, p)) > bound
end

function _seasonal_index(x, p)
    _isseasonal(x, p) || return ones(p)
    h = p ÷ 2
    t = (h + 1):(length(x) - h)
    # An even window needs half weights to stay centred.
    ma = iseven(p) ?
        [(x[i - h] / 2 + sum(x[(i - h + 1):(i + h - 1)]) + x[i + h] / 2) / p for i in t] :
        [sum(x[(i - h):(i + h)]) / p for i in t]
    ratio = x[t] ./ ma
    s = [Statistics.mean(ratio[mod1.(t, p) .== q]) for q in 1:p]
    return s ./ Statistics.mean(s)
end

function fit_transform(t::Deseasonalize, y::AbstractVector{Float64})
    index = _seasonal_index(y, t.period)
    if !all(isone, index)
        i = findfirst(≤(0), y)
        i === nothing || throw(ArgumentError(
            "Deseasonalize($(t.period)) adjusts multiplicatively, so it needs " *
            "positive values, but the series has $(y[i]) at position $i. Shift the " *
            "series to positive values or use Difference($(t.period)) instead."))
    end
    return y ./ index[mod1.(eachindex(y), t.period)], (index=index, n=length(y))
end

untransform(t::Deseasonalize, ẑ::AbstractVector{Float64}, state) =
    ẑ .* state.index[mod1.(state.n .+ eachindex(ẑ), t.period)]

"""
    LogRatio(window)

Model each value as the log of its ratio to the mean of the `window` values
before it, `z_t = log(x_t / mean(x[t-window:t-1]))`, which is scale-free and
keeps trending series stationary. The first `window` values have no ratio and
are dropped. Forecasts are rebuilt step by step from the last `window` training
values. Needs positive values and `window ≥ 1`.

# Example
```julia
LogRatio(12)   # monthly data: ratio to the previous year's mean
```
"""
struct LogRatio <: TargetTransform
    window::Int
    function LogRatio(window::Integer)
        window ≥ 1 || throw(ArgumentError(
            "LogRatio needs window ≥ 1, got $window. Pass e.g. LogRatio(12)."))
        return new(Int(window))
    end
end

function fit_transform(t::LogRatio, y::AbstractVector{Float64})
    i = findfirst(≤(0), y)
    i === nothing || throw(ArgumentError(
        "LogRatio($(t.window)) takes logarithms of ratios, so it needs positive " *
        "values, but the series has $(y[i]) at position $i. Shift the series to " *
        "positive values or use Difference(1) instead."))
    w = t.window
    z = [log(y[k] / Statistics.mean(y[(k - w):(k - 1)])) for k in (w + 1):length(y)]
    return z, y[max(1, end - w + 1):end]
end

function untransform(t::LogRatio, ẑ::AbstractVector{Float64}, last)
    w = t.window
    length(last) == w || throw(ArgumentError(
        "LogRatio($w) needs $w training values to rebuild forecasts, but the " *
        "series had $(length(last)). Use a shorter window or a longer series."))
    x = copy(last)
    for z in ẑ
        push!(x, exp(z) * Statistics.mean(x[(end - w + 1):end]))
    end
    return x[(w + 1):end]
end

"""
    Difference(lag)

Model each value's change from `lag` steps before, `z_t = y_t - y_{t-lag}`, and
rebuild forecasts by adding the changes back. `Difference(1)` removes a trend,
`Difference(12)` a monthly season. The first `lag` values are dropped. Requires
`lag ≥ 1`.

# Example
```julia
TransformChain(Difference(1), Difference(12))   # mlforecast's Differences([1, 12])
```
"""
struct Difference <: TargetTransform
    lag::Int
    function Difference(lag::Integer)
        lag ≥ 1 || throw(ArgumentError(
            "Difference needs lag ≥ 1, got $lag. Pass e.g. Difference(1)."))
        return new(Int(lag))
    end
end

fit_transform(t::Difference, y::AbstractVector{Float64}) =
    (y[(t.lag + 1):end] .- y[1:(end - t.lag)], y[max(1, end - t.lag + 1):end])

function untransform(t::Difference, ẑ::AbstractVector{Float64}, last)
    length(last) == t.lag || throw(ArgumentError(
        "Difference($(t.lag)) needs $(t.lag) training values to rebuild forecasts, " *
        "but the series had $(length(last)). Use a smaller lag or a longer series."))
    y = copy(last)
    for z in ẑ
        push!(y, z + y[end - t.lag + 1])
    end
    return y[(t.lag + 1):end]
end

"""
    TransformChain(transforms...)

Apply several transforms in order, each to the output of the one before, and
undo them in reverse order. Needs at least one transform.

# Example
```julia
TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))
```
"""
struct TransformChain{T<:Tuple} <: TargetTransform
    transforms::T
    function TransformChain(ts::TargetTransform...)
        isempty(ts) && throw(ArgumentError(
            "TransformChain needs at least one transform, e.g. " *
            "TransformChain(LastScale(12), LogRatio(12))."))
        return new{typeof(ts)}(ts)
    end
end

function fit_transform(c::TransformChain, y::AbstractVector{Float64})
    states = Any[]
    z = y
    for t in c.transforms
        z, s = fit_transform(t, z)
        push!(states, s)
    end
    return z, states
end

function untransform(c::TransformChain, ẑ::AbstractVector{Float64}, states)
    y = collect(ẑ)
    for (t, s) in zip(reverse(c.transforms), reverse(states))
        y = untransform(t, y, s)
    end
    return y
end

Base.:(==)(a::TransformChain, b::TransformChain) = a.transforms == b.transforms
Base.hash(c::TransformChain, h::UInt) = hash(c.transforms, hash(:TransformChain, h))

# Drop the rows the transform consumed and put z in the target column.
function _transform_table(fc, tbl::NamedTuple, y::Vector{Float64})
    z, state = fit_transform(fc.transform, y)
    z === y && return tbl, z, state
    i = findfirst(!isfinite, z)
    i === nothing || throw(ArgumentError(
        "transform=$(fc.transform) turned the target into a non-finite value " *
        "($(z[i])) at row $(length(y) - length(z) + i). Check the series for zeros " *
        "or outliers, or choose another transform."))
    k = length(y) - length(z)
    sub = row_subset(tbl, (k + 1):nrows(tbl))
    return merge(sub, NamedTuple{(fc.target,)}((collect(Float64, z),))), z, state
end
