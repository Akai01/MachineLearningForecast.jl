# Lower is better; inputs are assumed missing-free.

function _validate_metric_args(y, ŷ)
    length(y) == length(ŷ) || throw(ArgumentError(
        "metric inputs must have equal length, got length(y)=$(length(y)) and " *
        "length(ŷ)=$(length(ŷ)). Pass one forecast per actual, aligned element by " *
        "element."))
    isempty(y) && throw(ArgumentError(
        "metric inputs are empty: got length(y)=0 and length(ŷ)=0. Pass at least " *
        "one actual and its forecast."))
    return nothing
end

"""
    mae(y, ŷ) -> Float64

Mean absolute error, `mean(|y - ŷ|)`.

# Example
```julia
mae([1.0, 2.0], [2.0, 4.0])   # 1.5
```
"""
function mae(y::AbstractVector, ŷ::AbstractVector)
    _validate_metric_args(y, ŷ)
    return Statistics.mean(abs(a - b) for (a, b) in zip(y, ŷ))
end

"""
    rmse(y, ŷ) -> Float64

Root mean squared error, `sqrt(mean((y - ŷ)²))`.

# Example
```julia
rmse([1.0, 2.0], [2.0, 4.0])   # ≈ 1.581
```
"""
function rmse(y::AbstractVector, ŷ::AbstractVector)
    _validate_metric_args(y, ŷ)
    return sqrt(Statistics.mean(abs2(a - b) for (a, b) in zip(y, ŷ)))
end

"""
    mape(y, ŷ) -> Float64

Mean absolute percentage error, `mean(|y - ŷ| / |y|)`, as a fraction (0.10 =
10%). If any `y` is zero, MAPE is undefined: a warning is emitted and `Inf` is
returned — prefer [`smape`](@ref) or [`mase`](@ref) for data with zeros.

# Example
```julia
mape([10.0, 20.0], [11.0, 18.0])   # 0.1
```
"""
function mape(y::AbstractVector, ŷ::AbstractVector)
    _validate_metric_args(y, ŷ)
    if any(iszero, y)
        @warn "mape is undefined when y contains zeros; returning Inf. " *
              "Consider smape or mase instead."
        return Inf
    end
    return Statistics.mean(abs((a - b) / a) for (a, b) in zip(y, ŷ))
end

"""
    smape(y, ŷ) -> Float64

Symmetric mean absolute percentage error,
`mean(2|y - ŷ| / (|y| + |ŷ|))`, as a fraction in `[0, 2]`. A term with
`y = ŷ = 0` contributes 0.

# Example
```julia
smape([10.0, 20.0], [11.0, 18.0])   # ≈ 0.1003
```
"""
function smape(y::AbstractVector, ŷ::AbstractVector)
    _validate_metric_args(y, ŷ)
    return Statistics.mean(begin
        d = abs(a) + abs(b)
        iszero(d) ? 0.0 : 2 * abs(b - a) / d
    end for (a, b) in zip(y, ŷ))
end

"""
    mase(y, ŷ; y_train, m=1) -> Float64

Mean absolute scaled error: `mae(y, ŷ)` divided by the in-sample MAE of the
(seasonal) naive forecast with seasonality `m` on `y_train`
(`mean(|y_train[t] - y_train[t-m]|)`). A value below 1 means the errors are smaller
on average than the in-sample one-step (lag-`m`) naive errors (Hyndman & Koehler 2006);
multi-step errors are naturally larger, so compare with a naive baseline backtested at
the same horizon. On intermittent series (many zeros), MAE-based measures, including
MASE, can favour forecasts close to zero; the M5 competition used a squared-error
scaled measure (RMSSE) for this reason (Makridakis, Spiliotis & Assimakopoulos 2022);
use [`rmsse`](@ref) for such series. If the naive error is zero (constant training
series), a warning is emitted and `Inf` is returned. One series with a constant
training window makes that fold's score, and so the overall score, `Inf`.

# Example
```julia
mase([10.0, 12.0], [11.0, 11.0]; y_train=[1.0, 3.0, 1.0, 3.0])   # 0.5
```
"""
function mase(y::AbstractVector, ŷ::AbstractVector; y_train::AbstractVector, m::Integer=1)
    _validate_metric_args(y, ŷ)
    _nobool(m, "mase seasonality m", "Pass the season length in steps, e.g. m=7 for " *
            "weekly seasonality on daily data.")
    m ≥ 1 || throw(ArgumentError(
        "mase seasonality m must be ≥ 1, got $m. Pass the season length in steps, " *
        "e.g. m=7 for weekly seasonality on daily data, or m=1 (the default)."))
    length(y_train) > m || throw(ArgumentError(
        "mase needs length(y_train) > m; got length(y_train)=$(length(y_train)) " *
        "with m=$m. Pass a longer y_train or a smaller m."))
    denom = Statistics.mean(abs(y_train[i] - y_train[i - m])
                            for i in (m + 1):length(y_train))
    if iszero(denom)
        @warn "mase is undefined for a constant training series (naive error is " *
              "zero); returning Inf."
        return Inf
    end
    return mae(y, ŷ) / denom
end

"""
    rmsse(y, ŷ; y_train, m=1) -> Float64

Root mean squared scaled error: `rmse(y, ŷ)` divided by the root of the in-sample
mean squared error of the (seasonal) naive forecast with seasonality `m` on `y_train`,

    rmse(y, ŷ) / sqrt(mean((y_train[t] - y_train[t-m])²))

the scale being taken over the whole training window. It is the RMSSE of the M5
competition (Makridakis, Spiliotis & Assimakopoulos 2022), except that M5 computes the
scale only from the first non-zero value on, and its seasonal form for any `m`
(Hyndman & Athanasopoulos 2021, §5.8); Hyndman & Koehler (2006) suggest the in-sample
RMSE of the naive method as the denominator of a scaled RMSE. A value below 1 means the
errors are smaller than the in-sample one-step (lag-`m`) naive errors; multi-step
errors are naturally larger, so compare with a naive baseline backtested at the same
horizon.

On a single series, RMSSE ranks methods exactly as [`rmse`](@ref) does, since the scale
is the same for all of them (Koutsandreas et al. 2022); the scale matters when
comparing series. Squared errors suit intermittent series (many zeros), where absolute
errors favour forecasts close to zero (Makridakis, Spiliotis & Assimakopoulos 2022;
Hewamalage, Ackermann & Bergmeir 2023). The price is that RMSSE is more sensitive to
outliers than [`mase`](@ref) (Hyndman & Koehler 2006; Hewamalage, Ackermann &
Bergmeir 2023). If the naive error is zero (constant training series), a warning is
emitted and `Inf` is returned.

In [`backtest`](@ref) it is computed per series, aggregated across series weighted by
scored points, then averaged over folds. This is not the dollar-weighted WRMSSE of M5.
One series with a constant training window makes that fold's score, and so the overall
score, `Inf`.

# Example
```julia
rmsse([10.0, 14.0], [11.0, 11.0]; y_train=[1.0, 2.0, 4.0, 7.0])   # ≈ 1.035
```
"""
function rmsse(y::AbstractVector, ŷ::AbstractVector; y_train::AbstractVector,
               m::Integer=1)
    _validate_metric_args(y, ŷ)
    _nobool(m, "rmsse seasonality m", "Pass the season length in steps, e.g. m=7 for " *
            "weekly seasonality on daily data.")
    m ≥ 1 || throw(ArgumentError(
        "rmsse seasonality m must be ≥ 1, got $m. Pass the season length in steps, " *
        "e.g. m=7 for weekly seasonality on daily data, or m=1 (the default)."))
    length(y_train) > m || throw(ArgumentError(
        "rmsse needs length(y_train) > m; got length(y_train)=$(length(y_train)) " *
        "with m=$m. Pass a longer y_train or a smaller m."))
    denom = Statistics.mean(abs2(y_train[i] - y_train[i - m])
                            for i in (m + 1):length(y_train))
    if iszero(denom)
        @warn "rmsse is undefined for a constant training series (naive error is " *
              "zero); returning Inf."
        return Inf
    end
    return rmse(y, ŷ) / sqrt(denom)
end

"""
    needs_ytrain(metric) -> Bool

Trait declaring that `metric` needs the fold's *training* target in addition to
the actuals and forecasts. [`backtest`](@ref) calls such metrics as
`metric(y, ŷ; y_train=...)` and ordinary ones as `metric(y, ŷ)`. On a panel it
calls them once per series, with that series' own training values.

Defaults to `false`; `true` for [`mase`](@ref) and [`rmsse`](@ref). Define it for
your own scaled metrics so they can be passed to `backtest` / [`tune`](@ref):

```julia
my_mase(y, ŷ; y_train) = mae(y, ŷ) / mae(y_train[2:end], y_train[1:end-1])
MachineLearningForecast.needs_ytrain(::typeof(my_mase)) = true
```
"""
needs_ytrain(::Any) = false
needs_ytrain(::typeof(mase)) = true
needs_ytrain(::typeof(rmsse)) = true

_apply_metric(metric, y, ŷ, y_train) =
    needs_ytrain(metric) ? metric(y, ŷ; y_train=y_train) : metric(y, ŷ)

_iscallable(m) = !isempty(methods(m))
