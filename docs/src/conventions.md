# Leakage and conventions

This page states what the package guarantees against leakage, the conventions
you need to know to read a forecast or a backtest score correctly, how to
choose between [`Recursive`](@ref) and [`Direct`](@ref), and recipes for custom
features. The code blocks are plain Julia: they show the pattern and are not
run when the documentation is built.

## What the package guarantees

Every target-history feature
([`TargetFeature`](@ref MachineLearningForecast.TargetFeature)) uses only the
target values strictly before the current row's target, so rolling windows and
differences require `lag ≥ 1` and [`Lag`](@ref) requires `k ≥ 1`. A test
compares every built-in feature's training columns with its forecast-time
values and rewrites the target from a row onward to check that earlier rows do
not change. `backtest` fits each fold, including its target transform, on the
rows up to the fold's origin only.

One case is yours to keep: a [`CustomFeature`](@ref) function must read only
the history it is given. A function that reads a captured series or
`parent(h)` bypasses the guarantee. When the forecaster has a target
transform, the history is the transformed target.

## Conventions

**`Lag` under `Direct`.** Under [`Direct`](@ref), `Lag(k)` counts back from the
forecast origin, so at horizon `h` it is `y[t0-k+1]` (`t0` is the index of the
last observed value), not `y[target-k]`; `Lag(m)` is in phase with the target
only when `h = 1 (mod m)`; use `Lag.(1:m)`, which holds an in-phase lag at every
`h`, or [`Recursive`](@ref).

**`Exogenous` and `backtest`.** A backtest feeds the realised values of the
covariates, so its errors are a best case (too low) unless the column holds the
values that were actually known at the forecast origin. The recipes below build
such columns.

**Panel `backtest`.** A fold does not score a series that has no rows up to the
fold's origin, too few of them to train on (it is skipped with a warning), or
no actuals after it, so the set of scored series can change from fold to fold.

**Gaps and missing values.** The package refuses a time column with gaps,
duplicates or missing timestamps, and a target with `missing` or non-finite
values. Fill them causally (forward fill, or the mean of past values) before
fitting, because a fill that uses later values, such as interpolation, leaks.

**Aggregation of metrics.** Unscaled metrics pool the fold's rows; scaled
metrics are per series, weighted by scored points; the overall row is the
unweighted mean over folds. Scaled metrics are those that
[`needs_ytrain`](@ref MachineLearningForecast.needs_ytrain) marks, such as
[`mase`](@ref).

**Tuning base models with MLJ.** If you tune the base model with MLJ's
`TunedModel`, use `TimeSeriesCV` and never `CV()`, shuffled or not: every fold
but the last trains on rows that come after its test rows, so the model sees
the future of the block it is scored on. [Tuning the models](tuning.md) avoids
the question by scoring on a validation window that comes after the training
window.

## Choosing a strategy

[`Recursive`](@ref) fits one model on one-step-ahead rows, where the lag
features are the true past values. At forecast time it runs that model on its
own earlier predictions, so its errors compound over the horizon, and any
horizon can be forecast.

[`Direct`](@ref) fits one model per step `1..max_horizon`. Model `i` predicts
the target `i` steps ahead from the history at the forecast origin, so no
prediction is fed back and errors do not compound. It needs `max_horizon`
models, cannot forecast past `max_horizon`, and every step sees the same target
history, which is why the `Lag` convention above matters.

Neither is better in general. Compare them with [`backtest`](@ref), or let
[`tune`](@ref) search over `strategy`.

## Recipes

Each recipe assumes `df` is a daily series with the time column `:ds` and the
target `:y`, `panel` the same as a long table with an id column `:unique_id`, and
`model` an MLJ regressor. The recipes were run on small synthetic series before
they were written down.

### Rolling median, exponentially weighted mean, zeros

[`CustomFeature`](@ref) turns a function of the history into a feature. The
history `h` is a vector in forecasting and a view in training: read it, do not
keep it. In training the function is called once per row, so keep it cheap; the
window below bounds the cost of the exponentially weighted mean.

```julia
using MachineLearningForecast, Dates, Statistics

function ewm(h, α; window = 30)
    s = h[max(1, end - window + 1)]
    for v in @view h[max(1, end - window + 2):end]
        s = α * v + (1 - α) * s
    end
    return s
end

features = FeatureSet(
    Lag(1),
    CustomFeature(:roll_median_7, h -> median(@view h[end-6:end]), 7),
    CustomFeature(:ewm_03, h -> ewm(h, 0.3), 1),
    # Share of zeros in the last 28 values (fewer at the start).
    CustomFeature(:zero_share_28,
                  h -> count(iszero, @view h[max(1, end - 27):end]) / min(28, length(h)),
                  1),
    # Steps since the last non-zero value; 0 when the last value is non-zero.
    CustomFeature(:since_nonzero,
                  h -> (i = findlast(!iszero, h);
                        i === nothing ? length(h) : length(h) - i),
                  1),
)
fc = Forecaster(model; features, freq = Day(1))
```

`minhistory` is the number of history values the function needs: 7 for the
median, which reads the last 7. All four work under `Recursive` and `Direct`,
and in `backtest`.

### A holiday feature

A feature that depends on the date only subtypes
[`TimeFeature`](@ref MachineLearningForecast.TimeFeature). It needs
`outputnames` and `featurevalues!`; `minhistory` is 0 for every time feature,
and the default `materialize!` fills the training column by calling
`featurevalues!` on every row.

```julia
using Dates
using MachineLearningForecast: TimeFeature
import MachineLearningForecast: outputnames, featurevalues!

struct Holiday <: TimeFeature
    days::Set{Date}
end
outputnames(::Holiday) = [:is_holiday]
featurevalues!(out, f::Holiday, ctx) = (out[1] = Float64(Date(ctx.t) in f.days); nothing)

holidays = Holiday(Set([Date(2023, 1, 1), Date(2023, 5, 1), Date(2023, 12, 25)]))
fc = Forecaster(model; features = FeatureSet(Lag(1), holidays), freq = Day(1))
```

`ctx.t` is the timestamp of the row's target, so the feature needs no
`new_data`. See [`featurevalues!`](@ref MachineLearningForecast.featurevalues!)
and [`anchor`](@ref MachineLearningForecast.anchor) for the contract.

### A parent aggregate as a covariate

The total over all series (the parent in a hierarchy) is not known for the
forecast period, but its value `H` steps ago is, as long as `H` is at least the
forecast horizon. Shift it by `H`, then pass it through [`Exogenous`](@ref). The recipe assumes
that every series ends on the same date and that the dates are contiguous.
The column holds only values from before the forecast origin in `fit`, in
`forecast` and in `backtest`, so the backtest scores are not too low.

```julia
H = 7                                   # at least the horizon
dates = sort(unique(panel.ds))
total = [sum(panel.y[panel.ds .== d]) for d in dates]
# The parent's value H steps before each date.
lagged = Dict(d + H * Day(1) => v for (d, v) in zip(dates, total))

# The first H dates have no lagged value: leave them out of the training data.
keep = [haskey(lagged, d) for d in panel.ds]
train = (unique_id = panel.unique_id[keep], ds = panel.ds[keep], y = panel.y[keep],
         parent_lag = [lagged[d] for d in panel.ds[keep]])

panel_fc = Forecaster(model; features = FeatureSet(Lag(1), Exogenous(:parent_lag)),
                      freq = Day(1), id = :unique_id)
panel_fitted = fit(panel_fc, train)

stores = unique(panel.unique_id)
future = collect(dates[end] + Day(1):Day(1):dates[end] + Day(H))
future_data = (unique_id = repeat(stores, inner = H), ds = repeat(future, length(stores)),
               parent_lag = [lagged[d] for _ in stores for d in future])
forecast(panel_fitted, H; new_data = future_data)
backtest(panel_fc, train; horizon = H, windows = 2)
```

With a horizon above `H`, the lagged column would hold values from after the
origin, and the backtest would leak them.

### A covariate shifted by at least the horizon

The same rule works for any covariate: shift it by `k ≥ horizon` and the column
is known at the origin. For a single series with a `price` vector:

```julia
k = 14                                  # at least the horizon
h = 7
single = (ds = df.ds[(k + 1):end], y = df.y[(k + 1):end],
          price_lag14 = price[1:(end - k)])
fc = Forecaster(model; features = FeatureSet(Lag(1), Exogenous(:price_lag14)),
                freq = Day(1))
fitted = fit(fc, single)

n = length(df.y)
new_data = (ds = df.ds[end] .+ Day.(1:h), price_lag14 = price[(n - k + 1):(n - k + h)])
forecast(fitted, h; new_data)
```

### Scores per horizon step and per series

`BacktestResult.folds` has one row per fold, step and series with `:step`,
`:y` and `:y_hat` (and the id column for a panel), so any score is a few lines.
This one scores `panel_fc`, `train` and `H` from the parent aggregate recipe:

```julia
using Statistics
r = backtest(panel_fc, train; horizon = H, windows = 3)
f = r.folds
err = abs.(f.y .- f.y_hat)
by_step = [mean(err[f.step .== s]) for s in sort(unique(f.step))]
by_series = Dict(id => mean(err[f.unique_id .== id]) for id in unique(f.unique_id))
```

### A naive forecaster as a baseline

A model that returns `y_lag_1` is a naive forecaster: under `Recursive` every
step repeats the last observed value. Backtesting it gives the score to beat.

`NaiveLast` is an MLJ model, so it needs MLJModelInterface (for `MMI`) and
Tables.jl in your own environment.

```julia
using Tables
import MLJModelInterface as MMI

mutable struct NaiveLast <: MMI.Deterministic end
MMI.fit(::NaiveLast, verbosity, X, y) = (nothing, nothing, NamedTuple())
MMI.predict(::NaiveLast, fitresult, Xnew) =
    collect(Float64, Tables.getcolumn(Xnew, :y_lag_1))
MMI.input_scitype(::Type{NaiveLast}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{NaiveLast}) = AbstractVector{MMI.Continuous}

naive = Forecaster(NaiveLast(); features = FeatureSet(Lag(1)), freq = Day(1))
backtest(naive, df; horizon = 7, windows = 3, metrics = (mae, smape)).metrics
```
