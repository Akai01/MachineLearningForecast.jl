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
using MachineLearningForecast: FixedWeight
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
[`SeasonalNaive`](@ref)`(period)` model, exactly `FeatureSet(Lag(period))`, `Recursive()`
and no transform.

The seasonal pool weighs the value at time `t` by `exp(-recency * (T - t))`, with `T - t`
in time steps of `freq` (so the weights fall by `exp(-recency * period)` per season).
The paper writes the rate in "normalized time-index units" (App. A, p. 13), which is
ambiguous; per step is our reading. `recency=0` weighs the pool equally. When a horizon's
season has no history, `w_h` is 0 and only residuals are drawn (our choice; the paper does
not define it).

`rng` is copied when the method is built and once per series at every call, so repeated
calls give the same draws on the same Julia version (use `StableRNGs.StableRNG` for draws
that also reproduce across versions), and neither the method's `rng` nor the one passed in
is advanced.

The method has no coverage guarantee: its residual pool is motivated by split-conformal
calibration (Papadopoulos et al. 2002), whose finite-sample guarantee needs exchangeable
data that time series are not, and the paper reports coverage as an empirical property
(mean 0.89 at the 0.95 level on its benchmark). Measure it with [`Coverage`](@ref) in a
[`backtest`](@ref). Horizons are drawn independently, so a sample number does not trace a
path; the residual pool is one pool for all horizons and is not scaled by the series; the
paper did not test other values of `samples`, `recency` or `calibration`.

# Example
```julia
using MachineLearningForecast: PoolIntervals
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
                           rng=Random.Xoshiro(42))
        _whole(period, "PoolIntervals period", "Pass the season length in steps, " *
               "e.g. 12 for monthly data, or 1 for no seasonality.")
        period ≥ 1 || throw(ArgumentError(
            "PoolIntervals needs a seasonal period ≥ 1 in steps of freq, got $period. " *
            "Pass e.g. 12 for monthly data, or 1 for no seasonality."))
        rng isa Random.AbstractRNG || throw(ArgumentError(
            "rng must be a random number generator, got $(repr(rng)). Pass e.g. " *
            "rng=StableRNG(42) or rng=Random.Xoshiro(42)."))
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
        r = copy(rng)
        return new{typeof(weight),typeof(r)}(Int(period), weight, Float64(calibration),
                                             Float64(recency), Int(samples), r)
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
            # Draw u even at w = 0 or 1, so the edges match nearby w.
            S[j, b] = rand(rng) < w ? vals[_draw(rng, cw)] :
                      centre[j] + R[rand(rng, 1:length(R))]
        end
    end
    return S
end

_csp_fix(p::Integer) = "Use ConformalSeasonalPool($p; freq=...) to build the paired " *
                       "forecaster, or Forecaster(SeasonalNaive($p); " *
                       "features=FeatureSet(Lag($p)), strategy=Recursive(), freq=..., " *
                       "intervals=PoolIntervals($p))."

_check_pairing(model, features, strategy, transform, ::Nothing) = nothing

_check_pairing(model, features, strategy, transform, x) = throw(ArgumentError(
    "intervals must be nothing, a SplitConformal(holdout=c) or a PoolIntervals(m), got " *
    "$(repr(x)). Pass intervals=SplitConformal(holdout=60) for any model, or " *
    "intervals=PoolIntervals(m) for the seasonal naive one; ConformalSeasonalPool(m; " *
    "freq) builds a whole Forecaster."))

function _check_pairing(model, features, strategy, transform, iv::PoolIntervals)
    _check_centre(model, iv)
    outputnames(features) == [Symbol("y_lag_", iv.period)] || throw(ArgumentError(
        "intervals=PoolIntervals($(iv.period)) needs the feature set to be exactly " *
        "FeatureSet(Lag($(iv.period))), which feeds SeasonalNaive($(iv.period)), but " *
        "the feature set produces $(outputnames(features)). " * _csp_fix(iv.period)))
    _check_recursive(strategy, iv)
    _check_untransformed(transform, iv)
    return nothing
end

_check_centre(m::SeasonalNaive, iv::PoolIntervals) = m.period == iv.period ||
    throw(ArgumentError(
        "intervals=PoolIntervals($(iv.period)) needs SeasonalNaive($(iv.period)), got " *
        "SeasonalNaive($(m.period)): the pools are built around the seasonal naive " *
        "forecast of that period. " * _csp_fix(iv.period)))

_check_centre(m, iv::PoolIntervals) = throw(ArgumentError(
    "intervals=PoolIntervals($(iv.period)) is the paper's Conformal Seasonal Pools, " *
    "defined around the seasonal naive forecast, but the model is " *
    "$(nameof(typeof(m))). For any other model pass " *
    "intervals=SplitConformal(holdout=60). " * _csp_fix(iv.period)))

_check_recursive(::Recursive, iv::PoolIntervals) = nothing
_check_recursive(s::ForecastStrategy, iv::PoolIntervals) = throw(ArgumentError(
    "intervals=PoolIntervals($(iv.period)) needs strategy=Recursive(), got $s: under " *
    "Direct, Lag($(iv.period)) is not in phase with the target at every step, so the " *
    "centre is not the seasonal naive forecast. " * _csp_fix(iv.period)))

_check_untransformed(::NoTransform, iv::PoolIntervals) = nothing
_check_untransformed(t::TargetTransform, iv::PoolIntervals) = throw(ArgumentError(
    "intervals=PoolIntervals($(iv.period)) needs no target transform, got $t: the " *
    "paper's pools and centre are on the original scale. " * _csp_fix(iv.period)))

"""
    ConformalSeasonalPool(period; freq, weight=AdaptiveWeight(), calibration=0.5,
                          recency=0.01, samples=100, rng=Random.Xoshiro(42),
                          target=:y, time=:ds, id=nothing) -> Forecaster

The forecaster of Conformal Seasonal Pools (CSP; Manokhin 2026, arXiv:2605.03789): a
training-free probabilistic forecast around the seasonal naive forecast with season
length `period` (in steps of `freq`; 1 for none). It returns a [`Forecaster`](@ref) with
`SeasonalNaive(period)`, `FeatureSet(Lag(period))`, `Recursive()`, no transform and
`intervals=PoolIntervals(period; weight, calibration, recency, samples, rng)`, the only
setup the method is defined for (a hand-built `Forecaster` with a mismatched setup, or
with features beyond `FeatureSet(Lag(period))`, is refused). With `id` set, each series
of the panel has its own pools. `weight` is `AdaptiveWeight()` (the paper's
CSP-Adaptive, the default) or `FixedWeight(w)` (`FixedWeight(0.5)` is CSP-Fixed); both,
like `PoolIntervals`, are public and not exported:
`using MachineLearningForecast: PoolIntervals, FixedWeight`. `calibration`,
`recency`, `samples` and `rng` are described under `PoolIntervals`.

The method has no coverage guarantee: the paper reports empirical coverage only (mean
0.89 at the 0.95 level on its benchmark), so measure it on your data with
[`Coverage`](@ref) in a [`backtest`](@ref). Horizons are sampled independently; the
samples are not paths.

# Example
```julia
using MachineLearningForecast, Dates
csp = ConformalSeasonalPool(12; freq=Month(1))
fitted = fit(csp, df)                      # trains nothing; keeps each series' history
forecast(fitted, 18; levels=(80, 95))      # adds lo_80, hi_80, lo_95, hi_95
forecast_samples(fitted, 18)               # the draws, long format
backtest(csp, df; horizon=18, windows=3, levels=(95,), metrics=(crps, Coverage(95)))
```
"""
function ConformalSeasonalPool(period; freq, weight=AdaptiveWeight(), calibration=0.5,
                               recency=0.01, samples=100,
                               rng=Random.Xoshiro(42), target::Symbol=:y, time::Symbol=:ds,
                               id::Union{Nothing,Symbol}=nothing)
    fix = "Pass the season length in steps, e.g. 12 for monthly data, or 1 for no " *
          "seasonality."
    _whole(period, "ConformalSeasonalPool period", fix)
    period ≥ 1 || throw(ArgumentError(
        "ConformalSeasonalPool needs a seasonal period ≥ 1 in steps of freq, got " *
        "$period. $fix"))
    iv = PoolIntervals(period; weight, calibration, recency, samples, rng)
    return Forecaster(SeasonalNaive(iv.period); features=FeatureSet(Lag(iv.period)),
                      freq, target, time, id, intervals=iv)
end

_require_intervals(fc::Forecaster, who) = _require_intervals(fc.intervals, who)
_require_intervals(::Nothing, who) = throw(ArgumentError(
    "$who needs an interval method, but this forecaster has none. Pass " *
    "intervals=SplitConformal(holdout=60) for any model, or build the seasonal naive " *
    "one with ConformalSeasonalPool(period; freq=...)."))
_require_intervals(::PoolIntervals, who) = nothing

_check_levels(::Nothing) = Float64[]
function _check_levels(levels)
    ok = levels isa Real || applicable(iterate, levels)
    ls = ok ? collect(levels isa Real ? (levels,) : levels) : Any[]
    (ok && all(L -> L isa Real, ls)) || throw(ArgumentError(
        "levels must be percentages, got $(repr(levels)). Pass e.g. levels=(80, 95)."))
    out = Float64[_check_level(L) for L in ls]
    isempty(out) && throw(ArgumentError(
        "levels is empty; pass at least one level, e.g. levels=(95,)."))
    allunique(out) || throw(ArgumentError(
        "levels repeats a level, got $(repr(levels)); pass each level once, e.g. " *
        "levels=(80, 95)."))
    return out
end

# One matrix per series, each from its own rng copy.
function _sample_matrices(iv::PoolIntervals, f::FittedForecaster, preds)
    states = getfield(f, :series)
    out = Vector{Matrix{Float64}}(undef, length(states))
    for k in eachindex(states)
        st = states[k]
        out[k] = _in_series(f.spec, st.id) do
            R = _seasonal_differences(st.y_history, iv.period, iv.calibration)
            _pool_samples(copy(iv.rng), st.y_history, preds[k], R, iv)
        end
    end
    return out
end

function _forecast_samples(f::FittedForecaster, hs::Vector{Int}, future)
    states, grids, preds = _forecast_core(f, hs, future)
    out = _output_table(f.spec, states, grids, preds)
    return out, _sample_matrices(f.spec.intervals, f, preds)
end

# Columns lo_1, hi_1, lo_2, ... in the levels' order.
function _interval_columns(S::AbstractMatrix, levels::Vector{Float64})
    los, his = _interval_bounds(S, levels)
    return [c for l in eachindex(levels) for c in (los[l], his[l])]
end

function _level_table(taken, levels, cols)
    names = Tuple(Symbol(p, _level_label(L)) for L in levels for p in ("lo_", "hi_"))
    clash = [n for n in names if n in taken]
    isempty(clash) || throw(ArgumentError(
        "the interval column$(length(clash) == 1 ? "" : "s") " *
        "$(join(":" .* string.(clash), ", ")) would overwrite a column of the " *
        "result. Rename your time or id column."))
    return NamedTuple{names}(Tuple(cols))
end

"""
    forecast_samples(f::FittedForecaster, h::Integer; future=nothing) -> NamedTuple

The predictive samples behind `forecast(f, h; levels=...)`, as a long columntable: the id
column (for a panel), the time column, `:sample` (1 to the method's `samples`) and
`:y_hat`, one row per sample, `h * samples` rows per series. Needs a forecaster with an
interval method ([`ConformalSeasonalPool`](@ref)). The draws are the ones the intervals
are read from, and repeated calls give the same ones on the same Julia version. Each
series draws from its own copy of the method's `rng`, so every series starts from the same
random numbers: draws of different series are dependent (and with the default `Xoshiro`
they almost always stay aligned), so do not combine them by sample number (e.g. to sum a
panel). Horizons are drawn independently, so a sample number does not trace a path across
steps.

# Example
```julia
draws = forecast_samples(fitted, 18)
draws.y_hat[draws.sample .== 1]   # the first draw at each step
```
"""
function forecast_samples(f::FittedForecaster, h::Integer; future=nothing)
    _check_h(h)
    _require_intervals(f.spec, "forecast_samples")
    _check_support(f.spec.intervals, f, nothing, Int(h), "forecast_samples horizon")
    out, draws = _forecast_samples(f, fill(Int(h), nseries(f)), future)
    S = reduce(vcat, draws)
    n, B = size(S)
    rows = repeat(1:n; inner=B)
    lead = Base.front(keys(out))
    base = NamedTuple{lead}(map(c -> out[c][rows], lead))
    return merge(base, (sample=repeat(1:B; outer=n), y_hat=vec(permutedims(S))))
end
