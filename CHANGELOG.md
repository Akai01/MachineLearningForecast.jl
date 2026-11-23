# Changelog

All notable changes to MachineLearningForecast.jl are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `rmsse(y, ŷ; y_train, m=1)`: the root mean squared scaled error, the RMSE
  divided by the root of the in-sample mean squared (seasonal) naive error
  (M5's RMSSE, except that M5 computes the scale only from the first non-zero
  value on, while `rmsse` uses the whole training window). It declares `needs_ytrain`, so
  `backtest` and `tune` score it per series like `mase`; zero scale warns and
  returns `Inf`.
- A documentation page, "Leakage and conventions": what the package guarantees
  against leakage; the conventions behind forecasts and backtest scores (`Lag`
  under `Direct`, realised covariates in a backtest, the series scored in a
  panel backtest, gaps and missing values, how metrics aggregate, MLJ's `CV()`
  on time series, intermittent series); how to choose between `Recursive` and
  `Direct`; a "References" section with the literature behind these
  conventions; and recipes
  for a rolling median, an exponentially weighted mean, the share of zeros and
  the time since the last non-zero value as `CustomFeature`s, a holiday
  `TimeFeature`, a parent aggregate or a covariate shifted by at least the
  horizon passed through `Exogenous`, per-step and per-series scores from
  `BacktestResult.folds`, naive and seasonal naive baselines, and `mase` and
  `rmsse` scaled from the first non-zero value (the M5 rule) by trimming
  `y_train`.
- `Probabilistic` MLJ models work as base models: the forecast is the mean of
  the predicted distribution (`predict_mean`), so a model such as EvoTrees'
  `EvoTreeGaussian` fits, forecasts, backtests and tunes like a
  `Deterministic` one; before, forecasting failed with a `MethodError`. Such a
  model no longer warns when the `Forecaster` is built, while one that is
  neither `Deterministic` nor `Probabilistic` still does.
- **Backtest folds counted from each series' end.** `backtest` and `tune` take
  `windows` as an alternative to `initial`: the last `windows` folds, `step`
  apart, with the final fold ending at the end of the data. On a panel each
  series is cut at its own end, as the M competitions hold out each series'
  last values, so panels whose series span different years can be backtested
  and tuned; `windows=1` is the M3 holdout.
- **Target transforms.** `Forecaster` takes a `transform` keyword: a
  `TargetTransform` applied to each series' target before features are built,
  with its state fitted on that series' own training history, refitted in every
  `backtest` fold, and searchable as a `tune` grid key. Forecasts are mapped
  back to the original scale. Built-ins: `NoTransform` (the default, unchanged
  results), `MeanScale`, `LastScale(k)`, `Deseasonalize(period)` (M4
  seasonality test, classical multiplicative indices), `LogRatio(window)`,
  `Difference(lag)` and `TransformChain`; new ones implement `fit_transform`
  and `untransform`. On the 1399 M3 monthly series (18-month holdout, default
  hyperparameters, lags 1-12, a 12-month rolling mean and the month),
  `TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))` lowered
  sMAPE from 16.26 to 13.67 for EvoTrees, 15.69 to 13.66 for a random forest
  and 15.74 to 13.71 for LightGBM against no transform. `SeriesState` gains
  `z_history` and `transform_state`; `y_history` stays the untransformed target.
- **Panel (multi-series) forecasting.** `Forecaster` takes an `id` keyword; with
  it set, `fit` groups a long-format table by series, materialises features
  within each series, and pools the rows to train **one global model**.
  `forecast` returns an id column and forecasts each series from its own last
  timestamp, so ragged panels are supported. Exogenous covariates join on
  `(id, time)`; `backtest` cuts folds on the global timestamp grid and scores by
  joining on `(id, time)`; `tune` works on panels unchanged. Adds `nseries` and
  `SeriesState`, and `FittedForecaster` now exposes per-series state as
  `.series`.
- The documentation site shows the package logo: the hex logo in the sidebar
  and the wide logo with its tagline on the home page, each in a light and a
  dark variant. The README opens with a banner, light or dark to match the
  reader's system theme; each has its own background, so it reads on either
  page colour. The logo and banner files live in `docs/src/assets/`.
- A documentation page, "Using different models", fits six MLJ learners
  (EvoTrees, a DecisionTree random forest, MLJLinearModels, XGBoost, LightGBM
  and NearestNeighborModels) to the M3 monthly series, each as one global
  model with the same features, strategy and default hyperparameters. Each
  series is seasonally adjusted (the M4 seasonality test and classical
  multiplicative indices) and modelled as the log ratio to its previous 12
  months, with the built-in
  `TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))`. The page
  scores the learners with `backtest(...; windows = 1)`, which holds out each
  series' last 18 months, against seasonal-naive and Naive2 baselines, and
  compares them with published M3 results.
  XGBoost is given `base_score = 0.0`, the centre of the log-ratio target:
  MLJ's default of 0.5 is an offset every XGBoost model would first have to
  unlearn, while LightGBM and EvoTrees start from the target's mean.
- A documentation page, "Tuning the models", continues "Using different
  models": it gives each of the six learners the same budget of 30
  hyperparameter configurations (its defaults and 29 drawn at random with a
  fixed seed, or the whole space where it has 30 points), picks the one with
  the lowest sMAPE on the 18 months before the holdout, scored with
  `backtest(...; windows = 1)` on the panel without its holdout, and scores it
  once on the M3 holdout next to the defaults. Its XGBoost candidates get
  `base_score = 0.0` too, and a short check shows what the default 0.5 costs
  candidates with a small learning rate and few rounds. An optional section,
  "How much tuning can gain here", scores every configuration of the four tree
  ensembles on the holdout and on a second, earlier validation window, for
  diagnosis only: it reports the gain the best configuration would have given
  in hindsight and how well each validation score ranks the configurations.
- A panel `Direct` fit warns when a series has fewer training rows than
  `max_horizon`; such a series is left out of the later step models, which
  before happened silently.

### Changed

- A single series is fitted through the same code as a panel of one series; fits
  are unchanged. A panel `Direct` fit that has too few rows to train every step
  model now throws before it trains any of them (the message is unchanged).
- A single series is forecast through the same code as a panel of one series;
  forecasts are unchanged. A model that returns the wrong number of predictions
  for a forecast step now fails with an `ArgumentError` on panels too.
- A base model must not keep the `Xnew` table after `predict` returns: forecasting
  reuses its columns between steps.
- **Breaking: the `forecast` keyword `new_data` is now `future`.**
  `forecast(fitted, h; future=table)` takes the future-known covariates;
  `new_data=` is no longer accepted (no alias, no deprecation). The
  `ArgumentError` and warning texts that named the keyword now say `future`.
  Forecast values are unchanged.
- **Breaking: when every `tune` candidate fails, `tune` throws an `ArgumentError`**
  (it was an `ErrorException`). The message lists "candidate i (overrides): error"
  for the first five candidates, with the first error in full and the first line
  of each other error, then "... and N more", and says to fix the first error
  or check `horizon`, `initial`, the grid and the tuner's candidates. Partial
  failures are unchanged: errors are recorded in the table's `:error` column.
- **Breaking: `ask` is now `ask!`, and four names are no longer exported.** The
  tuning-strategy function that returns the next candidate is `ask!`, since it
  advances the strategy's state; there is no `ask` any more. `ask!`, `tell!`,
  `fit_transform` and `untransform` stay public API but are no longer exported:
  extend them with `import MachineLearningForecast: ask!, tell!` (tuning
  strategies) or `import MachineLearningForecast: fit_transform, untransform`
  (target transforms), so that a method you define extends the package's
  function instead of creating a new one. `TargetFeature`, `TimeFeature` and
  `ExogenousFeature`, the abstract types to subtype for a new feature, and
  `ForecastStrategy`, for dispatch and `isa` checks, are now exported and have
  docstrings with examples.
  The error `tune` raises when a strategy's `ask!` returns something other than
  a `NamedTuple` or `nothing`, or nothing at its first call, now says `ask!`.
- Documentation follows the forecasting literature (references on the
  conventions page). The README and the docs no longer call `CV()` unsafe in
  every case, and say when `TimeSeriesCV` helps, including its limit on a
  panel. The `Exogenous` and `backtest` docstrings and the README say that a
  backtest scores ex-post forecasts. The `mase` docstring says that its scale is
  the in-sample one-step (lag-`m`) naive error, and that MAE-based measures can
  favour near-zero forecasts on intermittent series. The `backtest` docstring
  and the README explain how scaled and unscaled metrics aggregate over a panel
  and the cross-series cutoff caveat of `windows`. The `Recursive` and `Direct`
  docstrings and the README compare the two strategies in both directions.
- **`Forecaster` has one constructor, `Forecaster(model; features, freq, ...)`.**
  The eight-argument positional constructor is gone. `features` that is not a
  `FeatureSet`, and a `freq` that is not a positive `Dates.Period`, now throw an
  `ArgumentError` when the `Forecaster` is built (before, a wrong `features`
  or `freq` type was a `TypeError` and a non-positive `freq` failed only at
  `fit`). `tune` rebuilds candidates through the same constructor and accepts
  only the keys `:model`, `:features`, `:strategy` and `:transform`. The
  data-shape fields `:freq`, `:target`, `:time` and `:id` are no longer valid
  grid keys or candidate overrides: a grid with one of them throws, and an
  `ask!` candidate with one fails that candidate, with a message saying the
  field describes the data and cannot be tuned (a name that is not a field is
  still reported as unknown). The `FeatureSet` error text for a bad
  `:features` candidate no longer starts with "tune candidate key".
- Documentation wording. The `Direct` docstring and the README no longer call
  the origin-anchored target history a "v1 simplification", and say what it
  means for `Lag` (`y[t0-k+1]`, not `y[target-k]`, at horizon `h`). The
  `Exogenous` and `backtest` docstrings say that a backtest feeds realised
  covariates, so it scores ex-post forecasts; `backtest` also says how metrics
  aggregate, and that a fold does not score a series that has no rows up to the
  fold's origin, too few of them to train on (it is skipped with a warning), or
  no actuals after it, so the set of scored series can change from fold to fold.
  The module docstring lists the target transforms, the `forecast`
  docstring gives `h::Integer`, as the method does, the `Difference` example no
  longer names another package, and the README roadmap promises prediction
  intervals without naming a method.
- The error for a feature that produces the time column no longer suggests a
  `CustomFeature` for a trend index, which only sees the length of a series'
  history and not a time index shared by all series; it points to `Calendar`
  and `Fourier`. The error for a non-numeric `Exogenous` value no longer says
  "In v1".
- The package precompiles a workload with PrecompileTools, a new dependency:
  single-series and panel `fit`, `forecast` and `backtest` with both
  strategies, with and without a transform. In a fresh session, loading the
  package and EvoTrees, then a first panel fit and forecast, a panel backtest
  and a single-series `Direct` fit and forecast took 9.0 s instead of 14.8 s
  on Julia 1.12.5 and 7.1 s instead of 10.2 s on Julia 1.10.12. Precompiling
  the package, once per install or update, takes 13 s instead of 2 s on Julia
  1.12.5 and 8 s instead of 2 s on Julia 1.10.12.
- A panel `Direct` fit builds each step's training table with one gather per
  column from the stacked series, instead of slicing every series and stacking
  the slices again. On the 1399 M3 monthly series, the package's own share of
  a `Direct(18)` fit allocates 429 MiB instead of 707 MiB, with identical
  results.
- On a panel, `Fourier` counts steps from the earliest series' first training
  timestamp, not from each series' own, so series that start on different
  dates have the same Fourier values on the same date and one global model
  learns one season instead of one per start phase. A single series is
  unchanged. On five panels of 100 daily series with a weekly season and
  start dates spread over 300 days, the mean absolute error of a 14-day
  forecast fell from 2.24 to 1.42 with `Fourier(7, 2)` alone and from 0.98
  to 0.94 with lags 1 to 7 added, for EvoTrees and the random forest alike.
  `SeriesState` gains `step0`, the Fourier step index of its first value.
- `BacktestResult.folds` gained a first column `:fold`, the fold number of each
  row, and the fold count shown by `show` now comes from it. A fold's `:origin`
  in `metrics` is `missing` when its series start their forecasts at different
  timestamps, as with `windows` on a ragged panel.
- **Models are trained without MLJ machines.** They are trained and queried
  through `MLJModelInterface` directly (`clean!`, `reformat`, `fit`,
  `predict`), which is what MLJBase's machines did for this package; forecasts
  are identical. MLJBase stays a dependency only because loading it gives
  models MLJModelInterface's full data interface: XGBoost, LightGBM,
  MLJLinearModels and NearestNeighborModels call `MLJModelInterface.matrix`,
  which fails without it. Skipping the machine layer shortened a fresh
  session's load plus first fit and forecast with `XGBoostRegressor` from about
  10.8 s to 8.0 s. Any MLJ model works, including `TunedModel` and other MLJ
  composites when their packages are loaded.
- `FittedForecaster.machines` (MLJ machines) is now `FittedForecaster.models`,
  a vector of `TrainedModel`s, each holding the trained copy of the model as
  `.model` and its MLJ fitresult as `.fitresult`; pass both to MLJ's interface
  functions, e.g. `fitted_params(t.model, t.fitresult)`, to inspect them.
- A model whose declared `input_scitype` does not cover a table of `Continuous`
  columns now warns when the `Forecaster` is built, instead of when MLJBase's
  machine was built at every fit. Models declaring no input scitype are not
  checked, as before.
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
  functions: a tuple, vector, `NamedTuple`, `Set` or generator. A one-shot
  iterator such as `Iterators.Stateful` is collected once, so every fold is
  scored; it used to score only the first fold.
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
- **Feature contract.** A feature computes one row through
  `featurevalues!(out, f, ctx)`, where `ctx` is a `FeatureContext` (the history
  before the target row, its time, its step index and its exogenous values); the
  training columns are derived from it, so by default (unless a feature
  overrides `materialize!`) the batch and the single-row path cannot disagree.
  The old `featurevalues(f, y_hist, t_next, exog_row)` and the vectorised
  `materialize!(out, f, y, t, data)` are gone (a feature that still defines them
  throws an `ArgumentError` saying what to implement); `materialize!(out, f,
  s::TrainingSeries)` is an optional fast path. `Fourier` no longer needs
  special handling anywhere. The user columns a feature declares through
  `inputcolumns` are read from the training data too, so they must be present
  there: `fit` throws otherwise. Forecasts and training matrices are unchanged.
- Features declare the user columns they read through `inputcolumns(f)`
  instead of a `cols` field or property: `Exogenous` returns its columns,
  every other built-in returns none, and a third-party `ExogenousFeature`
  without an `inputcolumns` method throws an `ArgumentError` at `fit`, `forecast`
  and `backtest` saying what to define. A `cols` property no longer works.
- Which row a feature's columns belong to under `Direct` is a trait,
  `anchor(f)`, instead of an `isa TargetFeature` test: `TargetFeature` defaults
  to `AtOrigin`, `TimeFeature` and `ExogenousFeature` to `AtTarget`, a feature
  of any other kind must define it or `Direct` throws an `ArgumentError`.
  `targetcolumnmask` is replaced by `anchors`. Results are unchanged.
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
  missing id, no rows, or a missing exogenous value in `future`. Passing a
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
- `CustomFeature` is parametric in its function (`CustomFeature{F}`), so the
  call to it is statically dispatched; behaviour, equality and the printed form
  are unchanged.

### Fixed

- `fitted.y_history`, `.t_last`, `.t_start`, `.n_train`, `.z_history` and
  `.transform_state` throw on every panel `FittedForecaster`, including a panel
  fitted with `id=` on a single series, as the `FittedForecaster` docstring says;
  read `fitted.series`.
- A missing exogenous value in a single series with a row-consuming transform
  (`Difference`, `LogRatio`, chains) is reported at the right row of your
  table, not shifted by the rows the transform consumed.
- Displaying a `TuneResult` no longer throws for a model with a `missing`
  hyperparameter, and a `NaN` hyperparameter left at its default is no longer
  shown as changed.
- `Lag(true)` built `Lag(1)` (a `Bool` is an integer to Julia). `Lag` now
  refuses a `Bool` with the whole-number `ArgumentError`.
  The other step counts and orders did the same and now refuse a `Bool` too:
  the `Diff` and rolling-feature arguments, `Fourier`'s period and order,
  `CustomFeature`'s `minhistory`, `Direct`, the target transforms,
  `RandomSearch`, `forecast`'s `h`, `mase`'s `m`, and the `horizon`,
  `initial`, `windows` and `step` of `backtest` and `tune` (and `max_evals`).
- `backtest` merged metrics with the same name (for example closures from
  one factory, or `(mae, mae)`): their per-fold rows were indistinguishable
  and every overall score came out as the mean of the metrics sharing the
  name. `backtest`, on a single series or a panel, now throws an
  `ArgumentError` naming the duplicated metric and asking you to remove the
  repeat or give each metric its own named function. `tune` takes one metric
  and is unaffected.
- `Fourier` values drifted as the step index grew: computing
  `sin(2πkn/period)` directly rounds more as `n` grows, so a phase took a
  slightly different value in every period (at phase 0, -2.4e-16 after one
  period and -2.7e-14 after 25). Tree learners split on these differences as
  if they were a time index, and histogram learners such as EvoTrees could put
  a forecast row in another phase's bin: on a year of daily data with a pure
  weekly season, EvoTrees missed by up to 3.6 on an amplitude of 5. `Fourier`
  now reduces `k * n` modulo the period first, so a phase has the same value in
  every period. On 200 noisy daily series with a weekly season, forecasting 14
  days with `Fourier(7, 2)` alone, the mean absolute error fell from 1.22 to
  1.08 for EvoTrees and the random forest; with lags 1 to 7 added it changed
  by less than 0.03.
- Panel `backtest` scored `mase`, and any metric declaring `needs_ytrain`, with
  one training vector pooled from every series in table order, so its scale
  mixed series and changed with row order. Two series climbing 1 and 10 a day
  under a naive forecast scored 0.014 instead of 3.0. Such metrics are now
  computed per series, with each series' own training values, and averaged
  with each series weighted by its scored points; other metrics are unchanged.
- Fitting a panel took time and memory quadratic in the number of series,
  because the series' training rows were stacked one pair at a time. They are
  now stacked in one pass, with identical results. On the 1399 M3 monthly
  series, the package's own share of a `Recursive` fit fell from about 1.2 s
  and 11 GB allocated to 0.06 s and 107 MB, and of a `Direct(18)` fit from
  16 s and 169 GB to 0.3 s and 711 MB; `backtest` and `tune` on panels gain
  the same way.
- Panel `backtest` found each forecast row's `step` by rescanning every earlier
  row of the fold, quadratic in the rows per fold. It now counts in one pass,
  with identical results. Three folds over the 1399 M3 monthly series on one
  shared calendar fell from 3.0 s to 0.3 s of the package's own time.
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
  held-out exogenous rows they slice from it, single-series and panel, and
  `forecast` rejected such a `future`. The check now reads the values: a missing
  timestamp in `future` throws an `ArgumentError` naming the row, and a
  `Date` vs `DateTime` mismatch is still rejected.
- `Calendar(:hour)` and `Calendar(:minute)` check the time values instead of
  the element type, so a `DateTime` column typed `Union{Missing,DateTime}` or
  `Any` is accepted. A `Date` column is still rejected with the same message.
- The `FittedForecaster`, `BacktestResult`, `ask!`, `tell!` and `TuneResult`
  docstrings have examples. `FittedForecaster`'s describes `fitted.series`,
  and `BacktestResult`'s lists the id column a panel backtest adds to `folds`.
- The `AbstractFeature`, `Direct` and `TuningStrategy` docstrings now state the
  four-method feature contract, `Direct`'s column anchoring and the ask!/tell!
  loop.
- The `forecast` docstring describes a panel forecast: the id column in the
  result, and the id column and per-series timestamps `future` needs. The
  `Forecaster` signature shows `id=nothing`, and the `TuningStrategy` and `ask!`
  docstrings list `:id` among the candidate keys.
- The `fit` docstring said the time column must be sorted. That holds for a
  single series; a panel's rows may come in any order, since `fit` sorts each
  series by time.
- Docstring examples: `smape([10.0, 20.0], [11.0, 18.0])` is ≈ 0.1003, not
  0.1002, and the examples load what they use (`Statistics` for `mean`,
  `Random` for `Xoshiro` in place of the test-only `StableRNG`, `Dates` for
  `Day`, `EvoTrees` for `EvoTreeRegressor`).
- The README says that a non-finite mean score counts as a tuning failure.
- The docs site's leakage warning now matches the README's and links the
  README. The README's `SeriesState`
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
