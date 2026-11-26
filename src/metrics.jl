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
training window makes that fold's score, and so its mean over folds in
[`summarize`](@ref), `Inf`.

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
One series with a constant training window makes that fold's score, and so its mean
over folds in [`summarize`](@ref), `Inf`.

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
`:mase_12` for `m=12` in the `scores` table.

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
`:rmsse_7` for `m=7` in the `scores` table.

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
[`Point`](@ref), [`Scaled`](@ref) and [`Samples`](@ref). Public, not exported.

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
`metric(y, ŷ; y_train)`, which also needs the fold's training target, and
[`Samples`](@ref) for `metric(y, S)`, which scores sampled forecasts (`tune` refuses
it). `Scaled` for
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

The name of `metric` in the `scores` table of a [`BacktestResult`](@ref): `Symbol(metric)`
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
    metric_inputs(m) isa Union{Point,Scaled,Samples} || throw(ArgumentError(
        "$who: metric_inputs returned $(repr(metric_inputs(m))), which is not Point(), " *
        "Scaled() or Samples(). Define metric_inputs for your metric, e.g. " *
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
        "$(repr(name)), so summarize would merge their per-fold rows into one " *
        "wrong mean. Remove the repeated metric, or name each one with a " *
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

"""
    Samples()

The [`metric_inputs`](@ref) kind of a metric called as `metric(y, S)`, where row `i` of
the matrix `S` holds the sampled forecasts for `y[i]`, such as [`crps`](@ref).
[`backtest`](@ref) scores it with the fold's samples, and [`tune`](@ref) refuses it.
Public, not exported.

# Example
```julia
import MachineLearningForecast: metric_inputs, Samples
metric_inputs(crps) === Samples()   # true
```
"""
struct Samples <: MetricInput end

function _check_sample_args(y::AbstractVector, S::AbstractMatrix)
    length(y) == size(S, 1) || throw(ArgumentError(
        "y has $(length(y)) values but S has $(size(S, 1)) rows; pass one row of " *
        "samples per actual value."))
    isempty(y) && throw(ArgumentError(
        "metric inputs are empty: pass at least one actual and its samples."))
    size(S, 2) ≥ 1 || throw(ArgumentError(
        "S has no samples (0 columns); pass at least one sample per actual."))
    any(ismissing, y) && throw(ArgumentError(
        "y contains missing values; drop or impute them before scoring."))
    any(ismissing, S) && throw(ArgumentError(
        "S contains missing values; check the model that produced the forecasts."))
    any(isnan, y) && throw(ArgumentError(
        "y contains NaN (first at index $(findfirst(isnan, y))); drop or impute it " *
        "before scoring."))
    any(isnan, S) && throw(ArgumentError(
        "S contains NaN (first at $(Tuple(findfirst(isnan, S)))); a NaN sample has no " *
        "rank. Check the model that produced the forecasts."))
    any(isinf, y) && throw(ArgumentError(
        "y contains Inf or -Inf (first at index $(findfirst(isinf, y))); drop or " *
        "impute it before scoring."))
    any(isinf, S) && throw(ArgumentError(
        "S contains Inf or -Inf (first at $(Tuple(findfirst(isinf, S)))); an infinite " *
        "sample makes every score infinite. Check the model that produced the " *
        "forecasts."))
    return nothing
end

function _check_level(level)
    level isa Real && 1 < level < 100 || throw(ArgumentError(
        "an interval level is a percentage in (1, 100), got $(repr(level)). Pass e.g. " *
        "95 for a 95% interval, not 0.95."))
    return Float64(level)
end

_level_label(L::Float64) = isinteger(L) ? string(Int(L)) : replace(string(L), "." => "_")

# Hyndman-Fan type 6, the one quantile rule here.
_sample_quantile(x::AbstractVector, p::Real) =
    Statistics.quantile(x, p; sorted=true, alpha=0, beta=0)

# One sort per row serves every level.
function _interval_bounds(S::AbstractMatrix, levels::AbstractVector{Float64})
    n = size(S, 1)
    los = [Vector{Float64}(undef, n) for _ in levels]
    his = [Vector{Float64}(undef, n) for _ in levels]
    for i in 1:n
        x = sort!(S[i, :])
        for (l, L) in enumerate(levels)
            los[l][i] = _sample_quantile(x, (100 - L) / 200)
            his[l][i] = _sample_quantile(x, (100 + L) / 200)
        end
    end
    return los, his
end

function _central(y::AbstractVector, S::AbstractMatrix, level::Float64)
    _check_sample_args(y, S)
    los, his = _interval_bounds(S, [level])
    return only(los), only(his)
end

# `pairs` is the denominator of the spread term.
function _crps(y::AbstractVector, S::AbstractMatrix, pairs::Real)
    B = size(S, 2)
    total = 0.0
    for i in 1:length(y)
        x = sort!(S[i, :])
        for k in 1:B
            total += abs(x[k] - y[i]) / B - (2k - B - 1) * x[k] / pairs
        end
    end
    return total / length(y)
end

"""
    crps(y, S) -> Float64

Continuous ranked probability score of sampled forecasts, averaged over the rows: for the
actual `y[i]` and the samples `x` in row `i` of `S`, the CRPS of their empirical
distribution, `mean(|x - y|) - mean(|x - x'|) / 2` over all `B^2` pairs (Gneiting and
Raftery 2007). This plug-in estimator scores the `B` samples as issued (our choice of
default; Zamo and Naveau 2018 recommend it for verifying the issued forecast); to score
the distribution they are drawn from, or to compare different `B`, use
[`fair_crps`](@ref). In the units of `y`, lower is better; one
sample gives the absolute error. The order of the samples does not matter.

# Example
```julia
crps([1.0], [0.0 2.0])   # 0.5
```
"""
function crps(y::AbstractVector, S::AbstractMatrix)
    _check_sample_args(y, S)
    return _crps(y, S, size(S, 2)^2)
end

"""
    fair_crps(y, S) -> Float64

The fair (ensemble-size adjusted) CRPS of sampled forecasts, averaged over the rows: like
[`crps`](@ref) but with the spread term `mean(|x - x'|) / 2` taken over the `B (B - 1)`
pairs of distinct samples (Ferro, Richardson and Weigel 2008, eq. 12 with `M = ∞`).
For samples drawn independently from the forecast distribution it is unbiased for that
distribution's CRPS, so it compares forecasts with different numbers of samples; `crps`
is higher by about `E|X - X'| / (2B)`. It needs at least 2 samples per row.

# Example
```julia
fair_crps([1.0], [0.0 2.0])   # 0.0; crps gives 0.5
```
"""
function fair_crps(y::AbstractVector, S::AbstractMatrix)
    _check_sample_args(y, S)
    B = size(S, 2)
    B ≥ 2 || throw(ArgumentError(
        "fair_crps needs at least 2 samples per row, got $B column. Use crps, which " *
        "accepts one sample, or pass more samples."))
    return _crps(y, S, B * (B - 1))
end

"""
    pinball_loss(y, S; quantiles=0.1:0.1:0.9) -> Float64

Mean pinball (quantile) loss of sampled forecasts: for every row and every `τ` in
`quantiles`, `u * (τ - (u < 0))` with `u = y[i] - q`, where `q` is the `τ`-quantile of
row `i` of `S` (Hyndman and Fan type 6, our choice: a new draw falls below the `k`-th
smallest sample with probability `k / (B + 1)`). It is the plain mean over rows and
levels, not normalised (our choice). In the units of `y`, lower is better.

# Example
```julia
pinball_loss([1.0], zeros(1, 10))   # 0.5
```
"""
function pinball_loss(y::AbstractVector, S::AbstractMatrix; quantiles=0.1:0.1:0.9)
    _check_sample_args(y, S)
    τs = collect(Float64, quantiles)
    (isempty(τs) || !all(τ -> 0 < τ < 1, τs)) && throw(ArgumentError(
        "pinball_loss quantiles must be a non-empty collection of levels in (0, 1), got " *
        "$(repr(quantiles)). Pass e.g. quantiles=(0.1, 0.5, 0.9)."))
    total = 0.0
    for i in 1:length(y)
        x = sort!(S[i, :])
        for τ in τs
            u = y[i] - _sample_quantile(x, τ)
            total += u * (τ - (u < 0))
        end
    end
    return total / (length(y) * length(τs))
end

"""
    Coverage(level=95)

Metric: the share of actuals inside the central `level` % interval of their samples, the
interval between the sample quantiles at `(100 - level) / 200` and `(100 + level) / 200`
(Hyndman and Fan type 6; a level above `100 (B - 1) / (B + 1)` clamps to the sample range).
Callable as `Coverage(95)(y, S)`; named `:coverage_95` in the `scores` table of a
[`backtest`](@ref). A calibrated forecast scores close to `level / 100`; higher is not
better, so it is not a score to minimise. `level` is a percentage in (1, 100), so
`Coverage(0.95)` is an error.

# Example
```julia
Coverage(95)([50.0], reshape(collect(1.0:100.0), 1, 100))   # 1.0
```
"""
struct Coverage
    level::Float64
    Coverage(level=95) = new(_check_level(level))
end

function (c::Coverage)(y::AbstractVector, S::AbstractMatrix)
    lo, hi = _central(y, S, c.level)
    return Statistics.mean(lo[i] ≤ y[i] ≤ hi[i] for i in 1:length(y))
end

"""
    IntervalWidth(level=95)

Metric: the mean width of the central `level` % interval of the samples (the same
interval as [`Coverage`](@ref)), a measure of sharpness. Callable as
`IntervalWidth(95)(y, S)`; named `:interval_width_95`. Compare widths only between
forecasts with similar `Coverage`: a narrow interval that misses is not sharp, it is
wrong.

# Example
```julia
IntervalWidth(80)([1.0], [0.0 1.0 2.0])   # 2.0
```
"""
struct IntervalWidth
    level::Float64
    IntervalWidth(level=95) = new(_check_level(level))
end

function (w::IntervalWidth)(y::AbstractVector, S::AbstractMatrix)
    lo, hi = _central(y, S, w.level)
    return Statistics.mean(hi .- lo)
end

metric_inputs(::typeof(crps)) = Samples()
metric_inputs(::typeof(fair_crps)) = Samples()
metric_inputs(::typeof(pinball_loss)) = Samples()
metric_inputs(::Coverage) = Samples()
metric_inputs(::IntervalWidth) = Samples()
metric_name(c::Coverage) = Symbol("coverage_", _level_label(c.level))
metric_name(w::IntervalWidth) = Symbol("interval_width_", _level_label(w.level))
