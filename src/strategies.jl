"""
    ForecastStrategy

Abstract supertype of forecasting strategies. Shipped strategies:
[`Recursive`](@ref) and [`Direct`](@ref).
"""
abstract type ForecastStrategy end

"""
    Recursive()

One model; forecasts are fed back as pseudo-history to compute lag features.
The model is trained once on the one-step-ahead frame and iterated over the
horizon, so any horizon can be forecast. Neither strategy is better in general:
`Recursive` estimates one model efficiently but is biased when the one-step model is
misspecified, and the errors it feeds back accumulate (references under
[`Direct`](@ref)).

# Example
```julia
using Dates
Forecaster(model; features=FeatureSet(Lag(1)), strategy=Recursive(), freq=Day(1))
```
"""
struct Recursive <: ForecastStrategy end

"""
    Direct(max_horizon)

One model per horizon step `1..max_horizon`. `max_horizon` fixes the number of
models at fit time; forecasting beyond it errors (refit with a larger horizon
or use [`Recursive`](@ref)).

Neither strategy is better in general. `Recursive` estimates one model efficiently but
is biased when the one-step model is misspecified, and the errors it feeds back
accumulate. `Direct` is more robust to such misspecification and to non-stationarity
such as unit roots, and feeds nothing back, but estimates each horizon separately, which
costs variance, and its forecasts at neighbouring steps need not be coherent (Marcellino,
Stock & Watson 2006; Chevillon 2007; Ben Taieb et al. 2012). Empirically, iterated
(`Recursive`) forecasts are often at least as accurate on stationary data (Marcellino,
Stock & Watson 2006). Compare both with [`backtest`](@ref).

At forecast time, the direct models condition their target-history features on
the history up to the forecast origin, while exogenous and time features vary per
step.

Under `Direct`, `Lag(k)` counts back from the forecast origin, so at horizon `h`
it is `y[t0-k+1]` (`t0` is the index of the last observed value), not `y[target-k]`;
`Lag(m)` is in phase with the target only when `h = 1 (mod m)`; use `Lag.(1:m)`, which
holds an in-phase lag at every `h`, or [`Recursive`](@ref).

Model `i` predicts the target `i` steps ahead. At fit time its `AtOrigin`
features (lags, rolling statistics, differences, custom) come from the feature
row `j`, and its `AtTarget` features (time and exogenous) come from the target
row `j+i-1`, which is what `forecast` feeds at step `i`; see [`anchor`](@ref).
Taking both from row `j` would be a train/serve skew that silently degrades
every step after the first.

# Example
```julia
using Dates
Forecaster(model; features=FeatureSet(Lag(1)), strategy=Direct(28), freq=Day(1))
```
"""
struct Direct <: ForecastStrategy
    max_horizon::Int
    function Direct(h::Integer)
        _nobool(h, "Direct max_horizon", "Pass the number of steps to forecast, " *
                "e.g. Direct(28).")
        h ≥ 1 || throw(ArgumentError("Direct max_horizon must be ≥ 1, got $h. Pass " *
                                     "the number of steps to forecast, e.g. Direct(28)."))
        return new(Int(h))
    end
end

Direct() = throw(ArgumentError(
    "Direct() requires a max_horizon, e.g. Direct(28): the Direct strategy fits " *
    "one model per horizon step, so the number of models must be fixed at fit " *
    "time. Use Recursive() if you want a horizon-agnostic forecaster."))

"Number of models a strategy fits."
nmodels(::Recursive) = 1
nmodels(s::Direct) = s.max_horizon

_shifted(::AtOrigin, col, i, n) = col[1:(n - i + 1)]
_shifted(::AtTarget, col, i, n) = col[i:n]
_gather_index(::AtOrigin, from, to) = from
_gather_index(::AtTarget, from, to) = to
