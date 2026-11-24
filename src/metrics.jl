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
    _check_season("mase", m, "m=7", "m=1 (the default)")
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
    _check_season("rmsse", m, "m=7", "m=1 (the default)")
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

function _check_season(who, m, ex, dflt)
    _nobool(m, "$who seasonality m", "Pass the season length in steps, e.g. $ex for " *
            "weekly seasonality on daily data.")
    m ≥ 1 || throw(ArgumentError(
        "$who seasonality m must be ≥ 1, got $m. Pass the season length in steps, " *
        "e.g. $ex for weekly seasonality on daily data, or $dflt."))
    return nothing
end

function _season(who, m)
    _check_season(who, m, "$who(7)", "$who() (the default)")
    m ≤ typemax(Int) || throw(ArgumentError(
        "$who seasonality m must be at most $(typemax(Int)), got $m. Pass the season " *
        "length in steps, e.g. $who(7) for weekly seasonality on daily data."))
    return Int(m)
end

"""
    Mase(m=1)

Callable form of [`mase`](@ref) with season length `m`, for [`backtest`](@ref) and
[`tune`](@ref): `Mase(m)(y, ŷ; y_train)` equals `mase(y, ŷ; y_train, m)`. It is scored
per series with that series' own training window, and named `:mase` for `m=1` and
`:mase_12` for `m=12` in the `metrics` table.

# Example
```julia
Mase()([10.0, 12.0], [11.0, 11.0]; y_train=[1.0, 3.0, 1.0, 3.0, 1.0])   # 0.5
backtest(fc, df; horizon=12, windows=3, metrics=(mae, Mase(12)))
```
"""
struct Mase
    m::Int
    Mase(m::Integer=1) = new(_season("Mase", m))
end

(s::Mase)(y::AbstractVector, ŷ::AbstractVector; y_train::AbstractVector) =
    mase(y, ŷ; y_train, m=s.m)

"""
    Rmsse(m=1)

Callable form of [`rmsse`](@ref) with season length `m`, for [`backtest`](@ref) and
[`tune`](@ref): `Rmsse(m)(y, ŷ; y_train)` equals `rmsse(y, ŷ; y_train, m)`. It is scored
per series with that series' own training window, and named `:rmsse` for `m=1` and
`:rmsse_7` for `m=7` in the `metrics` table.

# Example
```julia
Rmsse()([10.0, 14.0], [11.0, 11.0]; y_train=[1.0, 2.0, 4.0, 7.0])   # ≈ 1.035
backtest(fc, df; horizon=7, windows=3, metrics=(rmse, Rmsse(7)))
```
"""
struct Rmsse
    m::Int
    Rmsse(m::Integer=1) = new(_season("Rmsse", m))
end

(s::Rmsse)(y::AbstractVector, ŷ::AbstractVector; y_train::AbstractVector) =
    rmsse(y, ŷ; y_train, m=s.m)

"""
    MetricInput

Supertype of the kinds of inputs a metric takes, returned by [`metric_inputs`](@ref):
[`Point`](@ref) and [`Scaled`](@ref). Public, not exported.

# Example
```julia
import MachineLearningForecast: metric_inputs, MetricInput
metric_inputs(mae) isa MetricInput   # true
```
"""
abstract type MetricInput end

"""
    Point()

The [`metric_inputs`](@ref) kind of a metric called as `metric(y, ŷ)`, such as
[`mae`](@ref). [`backtest`](@ref) pools the fold's rows of all series. The default.
Public, not exported.

# Example
```julia
import MachineLearningForecast: metric_inputs, Point
metric_inputs(mae) === Point()   # true
```
"""
struct Point <: MetricInput end

"""
    Scaled()

The [`metric_inputs`](@ref) kind of a metric called as `metric(y, ŷ; y_train)`, such
as [`mase`](@ref). [`backtest`](@ref) calls it once per series with that series' own
training values, and weights the series by their scored points. Public, not exported.

# Example
```julia
import MachineLearningForecast: metric_inputs, Scaled
metric_inputs(mase) === Scaled()   # true
```
"""
struct Scaled <: MetricInput end

"""
    metric_inputs(metric) -> MetricInput

What `metric` is called with, so [`backtest`](@ref) and [`tune`](@ref) know how to score
it: [`Point`](@ref) (the default) for `metric(y, ŷ)`, [`Scaled`](@ref) for
`metric(y, ŷ; y_train)`, which also needs the fold's training target. `Scaled` for
[`mase`](@ref), [`rmsse`](@ref), [`Mase`](@ref) and [`Rmsse`](@ref). Define it for
your own scaled metrics. Public, not exported.

# Example
```julia
using MachineLearningForecast
import MachineLearningForecast: metric_inputs, Scaled
my_mase(y, ŷ; y_train) = mae(y, ŷ) / mae(y_train[2:end], y_train[1:end-1])
metric_inputs(::typeof(my_mase)) = Scaled()
backtest(fc, df; horizon=12, windows=3, metrics=(mae, my_mase))
```
"""
metric_inputs(::Any) = Point()
metric_inputs(::typeof(mase)) = Scaled()
metric_inputs(::typeof(rmsse)) = Scaled()
metric_inputs(::Mase) = Scaled()
metric_inputs(::Rmsse) = Scaled()

"""
    metric_name(metric) -> Symbol

The name of `metric` in the `metrics` table of a [`BacktestResult`](@ref): `Symbol(metric)`
by default (`:mae` for [`mae`](@ref)), `:mase` or `:mase_12` for [`Mase`](@ref). Two
metrics of one `backtest` need different names. Define it for a metric object of your
own, or name a metric in place with a NamedTuple key: `metrics=(error=mae,)`. Public, not
exported.

# Example
```julia
using MachineLearningForecast
import MachineLearningForecast: metric_inputs, metric_name, Scaled
struct MyScaled
    k::Int
end
(s::MyScaled)(y, ŷ; y_train) = mae(y, ŷ) / mae(y_train[2:end], y_train[1:end-1])
metric_inputs(::MyScaled) = Scaled()
metric_name(s::MyScaled) = Symbol("my_scaled_", s.k)
metric_name(MyScaled(2))   # :my_scaled_2
```
"""
metric_name(m) = Symbol(m)
metric_name(m::Mase) = m.m == 1 ? :mase : Symbol("mase_", m.m)
metric_name(m::Rmsse) = m.m == 1 ? :rmsse : Symbol("rmsse_", m.m)

_iscallable(m) = !isempty(methods(m))

function _check_metric(who, m)
    m isa Type && throw(ArgumentError(
        "$who: got the type $(nameof(m)); pass " *
        (m <: Union{Mase,Rmsse} ? "$(nameof(m))() or $(nameof(m))(12)" :
         "an instance of it") * "."))
    metric_inputs(m) isa Union{Point,Scaled} || throw(ArgumentError(
        "$who: metric_inputs returned $(repr(metric_inputs(m))), which is not Point() " *
        "or Scaled(). Define metric_inputs for your metric, e.g. " *
        "MachineLearningForecast.metric_inputs(::typeof(f)) = " *
        "MachineLearningForecast.Scaled()."))
    return nothing
end

function _check_distinct(names)
    allunique(names) && return nothing
    name = first(l for l in names if count(==(l), names) > 1)
    n = count(==(name), names)
    throw(ArgumentError(
        "backtest metrics must have distinct names, but $n metrics are named " *
        "$(repr(name)), so their per-fold rows would be merged into one wrong " *
        "overall score. Remove the repeated metric, or name each one with a " *
        "NamedTuple key (closures from one factory share a name), e.g. " *
        "metrics=(a=mae, b=Mase(12))."))
end

# A one-shot iterator is read once.
function _named_metrics(metrics)
    ms = applicable(iterate, metrics) ? collect(Any, metrics) : nothing
    ms !== nothing && all(_iscallable, ms) || throw(ArgumentError(
        "backtest metrics must be a tuple or vector of metric functions, got " *
        "$(repr(metrics)). Pass e.g. metrics=(mae, rmse), or metrics=(mae,) for one " *
        "metric. Name a metric with a NamedTuple key, e.g. " *
        "metrics=(error=mae, scaled=Mase(12)), not with a Pair."))
    names = metrics isa NamedTuple ? collect(Any, keys(metrics)) :
            Any[metric_name(m) for m in ms]
    for (name, m) in zip(names, ms)
        name isa Symbol || throw(ArgumentError(
            "backtest metric $(repr(m)): metric_name returned $(repr(name)), which is " *
            "not a Symbol. Define metric_name for your metric to return one, e.g. " *
            "MachineLearningForecast.metric_name(::MyMetric) = :my_metric."))
        _check_metric("backtest metric $(repr(name))", m)
    end
    _check_distinct(names)
    return Pair{Symbol,Any}[n => m for (n, m) in zip(names, ms)]
end
