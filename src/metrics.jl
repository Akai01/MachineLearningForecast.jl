# Forecast accuracy metrics. All take (y, ŷ) vectors of equal length and
# return a Float64; lower is better. Inputs are assumed missing-free.

function _validate_metric_args(y, ŷ)
    length(y) == length(ŷ) || throw(ArgumentError(
        "metric inputs must have equal length, got length(y)=$(length(y)) and " *
        "length(ŷ)=$(length(ŷ))."))
    isempty(y) && throw(ArgumentError("metric inputs are empty."))
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
smape([10.0, 20.0], [11.0, 18.0])   # ≈ 0.1002
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
seasonal-naive forecast with seasonality `m` on `y_train`
(`mean(|y_train[t] - y_train[t-m]|)`). Values below 1 beat the naive baseline.
If the naive error is zero (constant training series), a warning is emitted
and `Inf` is returned.

# Example
```julia
mase([10.0, 12.0], [11.0, 11.0]; y_train=[1.0, 3.0, 1.0, 3.0])   # 0.5
```
"""
function mase(y::AbstractVector, ŷ::AbstractVector; y_train::AbstractVector, m::Integer=1)
    _validate_metric_args(y, ŷ)
    m ≥ 1 || throw(ArgumentError("mase seasonality m must be ≥ 1, got $m."))
    length(y_train) > m || throw(ArgumentError(
        "mase needs length(y_train) > m; got length(y_train)=$(length(y_train)) " *
        "with m=$m."))
    denom = Statistics.mean(abs(y_train[i] - y_train[i - m]) for i in (m + 1):length(y_train))
    if iszero(denom)
        @warn "mase is undefined for a constant training series (naive error is " *
              "zero); returning Inf."
        return Inf
    end
    return mae(y, ŷ) / denom
end

"""
    needs_ytrain(metric) -> Bool

Trait declaring that `metric` needs the fold's *training* target in addition to
the actuals and forecasts. [`backtest`](@ref) calls such metrics as
`metric(y, ŷ; y_train=...)` and ordinary ones as `metric(y, ŷ)`.

Defaults to `false`; `true` for [`mase`](@ref). Define it for your own scaled
metrics so they can be passed to `backtest` / [`tune`](@ref):

```julia
my_mase(y, ŷ; y_train) = mae(y, ŷ) / mae(y_train[2:end], y_train[1:end-1])
MachineLearningForecast.needs_ytrain(::typeof(my_mase)) = true
```
"""
needs_ytrain(::Any) = false
needs_ytrain(::typeof(mase)) = true

# Call a metric under its own convention.
_apply_metric(metric, y, ŷ, y_train) =
    needs_ytrain(metric) ? metric(y, ŷ; y_train=y_train) : metric(y, ŷ)
