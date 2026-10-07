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
horizon, so any horizon can be forecast — at the price of compounding
prediction error through the fed-back history.

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

v1 simplification: at forecast time, the direct models condition their
target-history features on the training-end history, while exogenous and time
features vary per step.

Model `i` predicts the target `i` steps ahead. At fit time its target-history
features (lags, rolling statistics, differences, custom) come from the feature
row `j`, and its time and exogenous features come from the target row `j+i-1`,
which is what `forecast` feeds at step `i`. Taking both from row `j` would be a
train/serve skew that silently degrades every step after the first.

# Example
```julia
using Dates
Forecaster(model; features=FeatureSet(Lag(1)), strategy=Direct(28), freq=Day(1))
```
"""
struct Direct <: ForecastStrategy
    max_horizon::Int
    Direct(h::Integer) = h ≥ 1 ? new(Int(h)) :
        throw(ArgumentError("Direct max_horizon must be ≥ 1, got $h. Pass the " *
                            "number of steps to forecast, e.g. Direct(28)."))
end

Direct() = throw(ArgumentError(
    "Direct() requires a max_horizon, e.g. Direct(28): the Direct strategy fits " *
    "one model per horizon step, so the number of models must be fixed at fit " *
    "time. Use Recursive() if you want a horizon-agnostic forecaster."))

"Number of machines a strategy fits."
nmachines(::Recursive) = 1
nmachines(s::Direct) = s.max_horizon
