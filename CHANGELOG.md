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
- `Lag(1.5)`, `Lag("a")` and `Lag(missing)` throw an `ArgumentError` with an
  example instead of an `InexactError`, `MethodError` or `TypeError`
  respectively; a whole float such as `Lag(7.0)` is still accepted.
- `backtest` checks `metrics`, and `tune` checks `metric`, before fitting
  anything. A bare function (`metrics=mae`), symbols or strings throw an
  `ArgumentError` with an example (`metrics=(mae, rmse)`, `metric=smape`)
  instead of a `MethodError` after the first fold's fit, or in `tune` an "all
  candidates failed" error. `metrics` still accepts any iterable of metric
  functions: a tuple, vector, `NamedTuple`, `Set` or generator.
- `tune` checks the `:strategy` of each candidate it will evaluate (every grid
  candidate, the first `max_evals`, or the `RandomSearch` draws) before any
  fit. A value that is not a strategy (`strategy=[:recursive]`) throws an
  `ArgumentError` naming the candidate instead of a `MethodError`. A
  `:features` value that is not a `FeatureSet` (`features=[Lag(1)]`) still
  fails only its own candidate, but its `:error` entry is now an
  `ArgumentError` saying to wrap it in `FeatureSet(...)` instead of a
  `MethodError`.
- `tune` on a panel whose data lacks the time column throws the same
  `ArgumentError` as `backtest` instead of a `FieldError` (an `ErrorException`
  on Julia 1.10).
- A `CustomFeature` whose function indexes past its history (its `minhistory`
  is too small) or returns something that is not a number throws an
  `ArgumentError` naming the feature, the timestamp, the history length and
  the fix, instead of a bare `BoundsError` or `MethodError`. Any other error
  from the function, such as a `BoundsError` on a vector it captured, passes
  through unchanged. Valid custom features produce the same values as before.
- A third-party `ExogenousFeature` without a `cols` field or property throws
  an `ArgumentError` at `forecast` and `backtest` saying it must store its
  input columns in `cols::Vector{Symbol}`, instead of a `FieldError` (an
  `ErrorException` on Julia 1.10). A `cols` property defined through
  `getproperty` still works. `CONTRIBUTING.md` now documents that requirement.
- Panel errors about one series' data name the series and give the row of the
  table passed to `fit` or `backtest`. A missing or non-finite target and a
  missing exogenous value were reported without the series and at a row
  counted within it, and a missing or off-grid timestamp at a row of the
  series after sorting it by time.
- More `ArgumentError` messages name the offending value and the fix: those of
  the feature constructors (`Lag`, the rolling statistics, `Diff`, `Fourier`,
  `CustomFeature`), `Direct`, `Forecaster`, `forecast`, `backtest`,
  `RandomSearch`, `tune` and the metrics, of a missing time, target, id or
  `Exogenous` column, of a non-numeric target, and of panel data with a
  missing id, no rows, or a missing exogenous value in `new_data`. Passing a
  model type instead of an instance (`Forecaster(DecisionTreeRegressor; ...)`)
  names the type and the instance to pass, instead of reporting
  `got DataType`. A panel `backtest` fold with nothing to score says that no
  series trained up to its origin has data after it, instead of suggesting a
  `freq` grid mismatch.
- Forecasting allocates less per step: each feature fills the forecast row
  through a function barrier, and `Calendar` returns a concretely typed tuple.
  `Exogenous` converts its columns in a type-stable loop, so building the
  training frame from a table with many column types (for example a `String`
  id, a `Date`, and `Bool` and `Int` covariates) is several times faster.
  Forecasts are unchanged.

### Fixed

- `Recursive` fits, single-series and panel, now copy the model prototype
  before training, as `Direct` already did. A model holding an RNG object (e.g.
  `DecisionTreeRegressor(rng=StableRNG(1))`) no longer has its RNG advanced by
  `fit`, so refits, `backtest` folds and `tune` candidates are reproducible.
- Displaying a forecaster fitted on a panel (REPL echo, `show`, `repr`) no
  longer throws; it reads "trained on N series". A panel `Forecaster` now shows
  its `id` column.
- `tune` no longer picks a candidate whose mean score is `NaN` as the best (for
  example a model that predicts `NaN`). A non-finite mean score now marks the
  candidate as failed, with the reason in the `:error` column, and excludes it
  from ranking.
- `FeatureSet`, `Calendar` and `Exogenous` store a copy of the vector passed
  in. Mutating that vector after construction could get around their checks,
  including the `Forecaster` guard against a feature that emits the target
  column.
- A panel `backtest` with an `Exogenous` feature no longer throws when a
  series' rows end before or inside a fold's forecast window, or when series
  sit on different phases of the `freq` grid (e.g. weekly on Mondays and on
  Thursdays). Each series now takes exogenous values from its own rows and is
  forecast only over the steps it has data for, so a fold scores the same
  `(id, time)` rows as it would without the exogenous feature.
- A time column typed `Union{Missing,Date}` or `Any` (for example after
  `allowmissing`) whose values are all `Date`s now works with `Exogenous`
  features. `fit` accepted it, but `backtest` and `tune` rejected the
  `new_data` they slice from it, single-series and panel, and `forecast`
  rejected such a `new_data`. The check now reads the values: a missing
  timestamp in `new_data` throws an `ArgumentError` naming the row, and a
  `Date` vs `DateTime` mismatch is still rejected.
- `Calendar(:hour)` and `Calendar(:minute)` check the time values instead of
  the element type, so a `DateTime` column typed `Union{Missing,DateTime}` or
  `Any` is accepted. A `Date` column is still rejected with the same message.
- The `FittedForecaster`, `BacktestResult`, `ask`, `tell!` and `TuneResult`
  docstrings have examples. `FittedForecaster`'s describes `fitted.series`,
  and `BacktestResult`'s lists the id column a panel backtest adds to `folds`.
- The `AbstractFeature`, `Direct` and `TuningStrategy` docstrings now state the
  four-method feature contract, `Direct`'s column anchoring and the ask/tell
  loop.
- The `forecast` docstring describes a panel forecast: the id column in the
  result, and the id column and per-series timestamps `new_data` needs. The
  `Forecaster` signature shows `id=nothing`, and the `TuningStrategy` and `ask`
  docstrings list `:id` among the candidate keys.
- The `fit` docstring said the time column must be sorted. That holds for a
  single series; a panel's rows may come in any order, since `fit` sorts each
  series by time.
- Docstring examples: `smape([10.0, 20.0], [11.0, 18.0])` is ≈ 0.1003, not
  0.1002, and the examples load what they use (`Statistics` for `mean`,
  `Random` for `Xoshiro` in place of the test-only `StableRNG`, `Dates` for
  `Day`, `EvoTrees` for `EvoTreeRegressor`).
- The `tune` docstring and the README say that `tune` throws an
  `ErrorException` listing the errors when every candidate fails, and the
  README that a non-finite mean score counts as a failure.
- The docs site's leakage warning now says never to use `CV()`, shuffled or
  not, as the README does, and links the README. The README's `SeriesState`
  link is fixed, and it calls Tables.jl the only table dependency rather than
  the only dependency.

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
