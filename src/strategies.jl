"""
    ForecastStrategy

Abstract supertype of forecasting strategies. Shipped strategies:
[`Recursive`](@ref) and [`Direct`](@ref). It is the type of the `strategy` field
of a [`Forecaster`](@ref), and tuning grids and dispatch can use it.

# Example
```julia
strategies = [Recursive(), Direct(28)]
all(s -> s isa ForecastStrategy, strategies)   # true
```
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
or use [`Recursive`](@ref)). On a panel, a series with fewer than `max_horizon`
training rows trains only the step models it has rows for, and `fit` warns.

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

horizon_limit(::Recursive) = typemax(Int)
horizon_limit(s::Direct) = s.max_horizon
_feeds_back(::Recursive) = true
_feeds_back(::Direct) = false
_model_at(models, ::Recursive, s) = only(models)
_model_at(models, ::Direct, s) = models[s]

# Direct reads the fitted history, it never grows it.
_histories(::Recursive, states, hs) =
    [sizehint!(copy(st.z_history), length(st.z_history) + hs[k])
     for (k, st) in enumerate(states)]
_histories(::Direct, states, hs) = [st.z_history for st in states]

_gather_index(::AtOrigin, from, to) = from
_gather_index(::AtTarget, from, to) = to

function _pair_vectors(lens::Vector{Int}, i)
    m = sum(nX -> max(nX - i + 1, 0), lens)
    from, to = Vector{Int}(undef, m), Vector{Int}(undef, m)
    k = o = 0
    for nX in lens
        for j in 1:(nX - i + 1)
            k += 1
            from[k], to[k] = o + j, o + j + i - 1
        end
        o += nX
    end
    return from, to
end
