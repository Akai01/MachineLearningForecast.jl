"""
    MixtureWeight

How [`PoolIntervals`](@ref) weighs the seasonal pool against the residual pool at each
horizon: [`AdaptiveWeight`](@ref) (the paper's CSP-Adaptive) or [`FixedWeight`](@ref)
(CSP-Fixed). Public, not exported.
"""
abstract type MixtureWeight end

"""
    FixedWeight(w=0.5)

A constant seasonal-pool weight `w ∈ [0, 1]` at every horizon. `FixedWeight(0.5)` is the
paper's CSP-Fixed; `FixedWeight(0.0)` draws only residuals around the centre and
`FixedWeight(1.0)` only past values from the same season (residuals when a horizon's
season has no history).

# Example
```julia
ConformalSeasonalPool(12; freq=Month(1), weight=FixedWeight(0.5))
```
"""
struct FixedWeight <: MixtureWeight
    w::Float64
    function FixedWeight(w=0.5)
        fix = "Pass e.g. FixedWeight(0.5), the paper's CSP-Fixed."
        _nobool(w, "FixedWeight weight", fix; noun="number in [0, 1]")
        (w isa Real && 0 ≤ w ≤ 1) || throw(ArgumentError(
            "FixedWeight needs a seasonal-pool weight in [0, 1], got $(repr(w)). $fix"))
        return new(Float64(w))
    end
end

"""
    AdaptiveWeight()

The paper's CSP-Adaptive rule for the seasonal-pool weight: 0 without seasonality
(`period = 1`), 0.3 when one or two same-season values exist, 0.5 from three on. With
no same-season value the weight is 0 (see [`PoolIntervals`](@ref)). The default of
[`ConformalSeasonalPool`](@ref).

# Example
```julia
ConformalSeasonalPool(12; freq=Month(1), weight=AdaptiveWeight())
```
"""
struct AdaptiveWeight <: MixtureWeight end

Base.show(io::IO, w::FixedWeight) = print(io, "FixedWeight(", w.w, ")")
Base.show(io::IO, ::AdaptiveWeight) = print(io, "AdaptiveWeight()")

_mixture_weight(w::FixedWeight, m::Int, npool::Int) = w.w
_mixture_weight(::AdaptiveWeight, m::Int, npool::Int) =
    m ≤ 1 ? 0.0 : npool < 3 ? 0.3 : 0.5

"""
    PoolIntervals(period; weight=AdaptiveWeight(), calibration=0.5, recency=0.01,
                  samples=100, rng=Random.Xoshiro(42))

The interval method of Conformal Seasonal Pools (CSP; Manokhin 2026, arXiv:2605.03789),
the value of the `intervals` field of a [`Forecaster`](@ref). For each series and horizon
it draws `samples` values from a mixture: with probability `w_h` (`weight`), a past value
from the same position in the season, recent ones more likely; otherwise the point
forecast plus a signed seasonal difference `y[t] - y[t - period]` drawn uniformly from the
last `calibration` share of the series. The defaults are the paper's. `period` is the
season length in steps of `freq` (1 for none). Build the paired forecaster with
[`ConformalSeasonalPool`](@ref); a hand-built `Forecaster` accepts this method only with a
[`SeasonalNaive`](@ref)`(period)` model, `Lag(period)`, `Recursive()` and no transform.

The seasonal pool weighs the value at time `t` by `exp(-recency * (T - t))`, with `T - t`
in time steps of `freq` (so the weights fall by `exp(-recency * period)` per season).
The paper writes the rate in "normalized time-index units" (App. A, p. 13), which is
ambiguous; per step is our reading. `recency=0` weighs the pool equally. When a horizon's
season has no history, `w_h` is 0 and only residuals are drawn (our choice; the paper does
not define it).

`rng` is copied once per series at every call, so repeated calls give the same draws on
the same Julia version (use `StableRNGs.StableRNG` for draws that also reproduce across
versions) and the method's own `rng` is never advanced.

The method has no coverage guarantee: its residual pool is motivated by split-conformal
calibration (Papadopoulos et al. 2002), whose finite-sample guarantee needs exchangeable
data that time series are not, and the paper reports coverage as an empirical property
(mean 0.89 at the 0.95 level on its benchmark). Measure it with [`Coverage`](@ref) in a
[`backtest`](@ref). Horizons are drawn independently, so a sample number does not trace a
path; the residual pool is one pool for all horizons and is not scaled by the series; the
paper did not test other values of `samples`, `recency` or `calibration`.

# Example
```julia
PoolIntervals(12; calibration=0.5, recency=0.01, samples=100)
```
"""
struct PoolIntervals{W<:MixtureWeight,G<:Random.AbstractRNG}
    period::Int
    weight::W
    calibration::Float64
    recency::Float64
    samples::Int
    rng::G
    function PoolIntervals(period; weight=AdaptiveWeight(), calibration=0.5,
                           recency=0.01, samples=100,
                           rng::Random.AbstractRNG=Random.Xoshiro(42))
        _whole(period, "PoolIntervals period", "Pass the season length in steps, " *
               "e.g. 12 for monthly data, or 1 for no seasonality.")
        period ≥ 1 || throw(ArgumentError(
            "PoolIntervals needs a seasonal period ≥ 1 in steps of freq, got $period. " *
            "Pass e.g. 12 for monthly data, or 1 for no seasonality."))
        weight isa MixtureWeight || throw(ArgumentError(
            "weight must be an AdaptiveWeight() or a FixedWeight(w), got " *
            "$(repr(weight)). Pass e.g. weight=FixedWeight(0.5)."))
        cfix = "Pass e.g. calibration=0.5."
        _nobool(calibration, "calibration", cfix; noun="number in (0, 1]")
        (calibration isa Real && 0 < calibration ≤ 1) || throw(ArgumentError(
            "calibration must be in (0, 1], the share of each series' history whose " *
            "seasonal differences form the residual pool, got $(repr(calibration)). $cfix"))
        rfix = "Pass e.g. recency=0.01, or 0 to weigh the pool equally."
        _nobool(recency, "recency", rfix; noun="number ≥ 0")
        (recency isa Real && isfinite(recency) && recency ≥ 0) || throw(ArgumentError(
            "recency must be a finite rate ≥ 0 (the decay per step of the " *
            "seasonal-pool weights), got $(repr(recency)). $rfix"))
        _whole(samples, "samples", "Pass the number of draws per horizon, e.g. " *
               "samples=100.")
        samples ≥ 1 || throw(ArgumentError(
            "samples must be a whole number ≥ 1, got $samples. Pass e.g. samples=100."))
        return new{typeof(weight),typeof(rng)}(Int(period), weight, Float64(calibration),
                                               Float64(recency), Int(samples), rng)
    end
end

Base.show(io::IO, iv::PoolIntervals) =
    print(io, "PoolIntervals(", iv.period, ", ", iv.weight, ")")

# The pool R of Algorithm 1.
function _seasonal_differences(y::AbstractVector{Float64}, m::Int, ρ::Float64)
    T = length(y)
    ncal = floor(Int, ρ * T)
    return [y[t] - y[t - m] for t in max(T - ncal + 1, m + 1):T]
end

# Newest weight is 1, so large rates cannot underflow to 0.
function _seasonal_pool(y::AbstractVector{Float64}, j::Integer, m::Integer, λ::Float64)
    T = length(y)
    ts = mod1(T + j, m):m:T
    return y[ts], cumsum([exp(-λ * (last(ts) - t)) for t in ts])
end

_draw(rng::Random.AbstractRNG, cw::Vector{Float64}) =
    searchsortedlast(cw, rand(rng) * cw[end]) + 1

# Algorithm 1: one row per horizon, one column per draw.
function _pool_samples(rng::Random.AbstractRNG, y::AbstractVector{Float64},
                       centre::AbstractVector{Float64}, R::AbstractVector{Float64},
                       cp::PoolIntervals)
    m, B = cp.period, cp.samples
    S = Matrix{Float64}(undef, length(centre), B)
    for j in eachindex(centre)
        vals, cw = _seasonal_pool(y, j, m, cp.recency)
        w = isempty(vals) ? 0.0 : _mixture_weight(cp.weight, m, length(vals))
        w < 1 && isempty(R) && throw(ArgumentError(
            "the residual pool for step $j is empty: the last calibration=" *
            "$(cp.calibration) of the $(length(y)) training values hold no seasonal " *
            "difference y[t] - y[t - $m]. Raise calibration or provide a longer series."))
        for b in 1:B
            S[j, b] = rand(rng) < w ? vals[_draw(rng, cw)] :
                      centre[j] + R[rand(rng, 1:length(R))]
        end
    end
    return S
end
