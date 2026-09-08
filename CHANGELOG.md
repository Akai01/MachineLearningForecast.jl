# Changelog

All notable changes to MachineLearningForecast.jl are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Panel (multi-series) forecasting.** `Forecaster` takes an `id` keyword; with
  it set, `fit` groups a long-format table by series, materialises features
  within each series, and pools the rows to train **one global model**.
  `forecast` returns an id column and forecasts each series from its own last
  timestamp, so ragged panels are supported. Exogenous covariates join on
  `(id, time)`; `backtest` cuts folds on the global timestamp grid and scores by
  joining on `(id, time)`; `tune` works on panels unchanged. Adds `nseries` and
  `SeriesState`, and `FittedForecaster` now exposes per-series state as
  `.series`.

### Changed

- `Forecaster` gained a third type parameter (`Nothing` or `Symbol`) so panel
  and single-series fitting are selected by dispatch rather than a runtime flag.
- `FittedForecaster` stores `series::Vector{SeriesState}`. For a single series
  `f.y_history`, `f.t_last`, `f.t_start` and `f.n_train` still work; on a panel
  they raise an error pointing at `f.series`.

## [0.1.0]

Initial release.

### Added

- `Forecaster` / `FittedForecaster` with a functional `fit` / `forecast` pair;
  any MLJ `Deterministic` regressor works as the base model.
- Declarative features: `Lag`, `RollingMean`, `RollingStd`, `RollingMin`,
  `RollingMax`, `Diff`, `Calendar`, `Fourier`, `Exogenous`, `CustomFeature`,
  composed with `FeatureSet`.
- Forecasting strategies `Recursive` and `Direct`, selected by dispatch.
- Future-known exogenous covariates, joined to the forecast horizon by
  timestamp.
- `backtest` for expanding-window time-series cross-validation, and the metrics
  `mae`, `rmse`, `mape`, `smape` and `mase` (the latter via the `needs_ytrain`
  trait so scaled metrics receive each fold's training window).
- `tune` for pipeline-level tuning on backtest score, with `GridSearch` and
  `RandomSearch`, and a documented public ask/tell `TuningStrategy` interface so
  search strategies can be implemented outside the package.
- Tables.jl-native I/O throughout: any Tables.jl source is accepted and all
  tabular results are columntables. No DataFrames.jl dependency.
