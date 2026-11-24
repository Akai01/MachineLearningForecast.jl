# MachineLearningForecast.jl

```@raw html
<p style="text-align: center;">
  <img class="docs-light-only" src="assets/logo-wide-tagline.svg"
       alt="MachineLearningForecast.jl logo" width="520"/>
  <img class="docs-dark-only" src="assets/logo-wide-tagline-dark.svg"
       alt="MachineLearningForecast.jl logo" width="520"/>
</p>
```

Generic, composable time-series forecasting built on the MLJ model interface:
any MLJ `Deterministic` regressor is a valid base model.

```julia
using MachineLearningForecast, EvoTrees, Dates

df = (ds = collect(Date(2020,1,1):Day(1):Date(2023,12,31)),
      y  = randn(1461) .+ 10)

fc = Forecaster(EvoTreeRegressor(nrounds=200, eta=0.05);
                features = FeatureSet(Lag(1), Lag(7), RollingMean(7),
                                      Calendar(:dayofweek), Fourier(365.25, 3)),
                strategy = Recursive(),
                freq     = Day(1))

fitted = fit(fc, df)
fcast  = forecast(fitted, 28)
```

MachineLearningForecast is Tables.jl-native: any Tables.jl-compatible source is accepted
(NamedTuple of vectors, `DataFrame`, `CSV.File`, ...), and all tabular results
are columntables (NamedTuples of vectors).

See the [README](https://github.com/Akai01/MachineLearningForecast.jl#readme)
for the full tour: feature catalogue, strategies, backtesting, and the
two-level tuning story — including the **leakage warning** about preferring
`TimeSeriesCV` to MLJ's `CV()` when tuning base models with MLJ's `TunedModel`.

[Leakage and conventions](conventions.md) states what the package guarantees against
leakage, the conventions behind forecasts and backtest scores, how to choose a
strategy, and recipes for custom features.
[Using different models](models.md) compares six MLJ learners on the M3 monthly series.
[Tuning the models](tuning.md) tunes their hyperparameters on a validation window.

## API reference

```@docs
MachineLearningForecast
```

### Forecasting

```@docs
Forecaster
FittedForecaster
MachineLearningForecast.TrainedModel
SeriesState
nseries
fit
forecast
Recursive
Direct
ForecastStrategy
```

### Features

```@docs
FeatureSet
MachineLearningForecast.outputnames
MachineLearningForecast.minhistory
MachineLearningForecast.inputcolumns
MachineLearningForecast.FeatureContext
MachineLearningForecast.featurevalues!
MachineLearningForecast.materialize!
MachineLearningForecast.TrainingSeries
MachineLearningForecast.Anchor
MachineLearningForecast.AtOrigin
MachineLearningForecast.AtTarget
MachineLearningForecast.anchor
MachineLearningForecast.anchors
Lag
RollingMean
RollingStd
RollingMin
RollingMax
Diff
Calendar
Fourier
Exogenous
CustomFeature
MachineLearningForecast.AbstractFeature
TargetFeature
TimeFeature
ExogenousFeature
```

### Target transforms

```@docs
TargetTransform
MachineLearningForecast.fit_transform
MachineLearningForecast.untransform
MachineLearningForecast.dropped_rows
NoTransform
MeanScale
LastScale
Deseasonalize
LogRatio
Difference
TransformChain
```

### Backtesting and metrics

```@docs
backtest
BacktestResult
mae
rmse
mape
smape
mase
rmsse
Mase
Rmsse
MachineLearningForecast.metric_inputs
MachineLearningForecast.metric_name
MachineLearningForecast.MetricInput
MachineLearningForecast.Point
MachineLearningForecast.Scaled
```

### Tuning

```@docs
tune
TuneResult
TuningStrategy
GridSearch
RandomSearch
MachineLearningForecast.ask!
MachineLearningForecast.tell!
```
