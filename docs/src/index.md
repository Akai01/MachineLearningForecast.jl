# MachineLearningForecast.jl

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

See the README for the full tour:
feature catalogue, strategies, backtesting, and the two-level tuning story —
including the **leakage warning** about always using `TimeSeriesCV` (never
shuffled `CV()`) when tuning base models with MLJ's `TunedModel`.

## API reference

```@docs
MachineLearningForecast
```

### Forecasting

```@docs
Forecaster
FittedForecaster
fit
forecast
Recursive
Direct
MachineLearningForecast.ForecastStrategy
```

### Features

```@docs
FeatureSet
MachineLearningForecast.outputnames
MachineLearningForecast.minhistory
MachineLearningForecast.targetcolumnmask
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
MachineLearningForecast.TargetFeature
MachineLearningForecast.TimeFeature
MachineLearningForecast.ExogenousFeature
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
MachineLearningForecast.needs_ytrain
```

### Tuning

```@docs
tune
TuneResult
TuningStrategy
GridSearch
RandomSearch
ask
tell!
```
