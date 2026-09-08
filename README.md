# MachineLearningForecast.jl

**Generic, composable time-series forecasting for Julia, built on the MLJ model
interface.** Any MLJ `Deterministic` regressor is a valid base model — random
forests, [EvoTrees](https://github.com/Evovest/EvoTrees.jl) gradient boosting,
XGBoost, LightGBM, CatBoost, linear models — MachineLearningForecast never hard-codes a
learner.

- **Declarative feature engineering**: lags, rolling statistics, differences,
  calendar features, Fourier seasonal terms, future-known exogenous covariates,
  and custom features, composed as first-class Julia objects.
- **Pluggable forecasting strategies**: `Recursive` (one model, iterated) and
  `Direct` (one model per horizon step), chosen by dispatch — extensible by
  design.
- **Backtesting**: expanding-window time-series cross-validation with pluggable
  metrics.
- **Pipeline tuning**: grid/random search over models, feature sets, and
  strategies, scored on backtest error, with a documented ask/tell interface
  for plugging in your own search strategy (e.g. a Bayesian optimizer).
- **Tables.jl-native**: accepts *any* Tables.jl-compatible source (a NamedTuple
  of vectors, `CSV.File`, a `DataFrame`, ...) and returns plain columntables
  (NamedTuples of vectors) that any table sink understands. MachineLearningForecast itself
  depends only on Tables.jl — bring whatever table type you like.

## Quick start

```julia
using MachineLearningForecast, EvoTrees, Dates

n  = 1461
ds = Date(2020, 1, 1):Day(1):Date(2023, 12, 31)
df = (ds    = collect(ds),
      y     = 10 .+ 2 .* sin.(2π .* (1:n) ./ 7) .+ 0.5 .* randn(n),
      promo = rand(Bool, n))                     # exogenous, known in future

model = EvoTreeRegressor(nrounds=200, eta=0.05)  # any MLJ Deterministic regressor

fc = Forecaster(
    model;
    features = FeatureSet(
        Lag(1), Lag(7), Lag(14),
        RollingMean(7; lag=1), RollingStd(28; lag=1),
        Calendar(:dayofweek, :month, :weekofyear),
        Fourier(365.25, 3),
        Exogenous(:promo),
    ),
    strategy = Recursive(),          # or Direct(28)
    freq     = Day(1),
    target   = :y,
    time     = :ds,
)

fitted = fit(fc, df)

future_exog = (ds    = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 28)),
               promo = rand(Bool, 28))

fcast = forecast(fitted, 28; new_data = future_exog)
fcast.y_hat            # the 28 point forecasts
fcast.ds               # the future timestamps

# Backtesting (expanding window)
results = backtest(fc, df; horizon=28, initial=730, step=28,
                   metrics=(mae, rmse, smape))
results.metrics        # per-fold and overall scores
```

Everything above works identically with a `DataFrame`, a `CSV.File`, or any
other Tables.jl source in place of the NamedTuple — and the returned
columntables convert straight back: `DataFrame(fcast)`.

## Features

| Feature | Kind | Description |
|---|---|---|
| `Lag(k)` | target | `y_{t-k}` |
| `RollingMean(w; lag=1)` | target | mean of the `w` values ending `lag` steps back |
| `RollingStd(w; lag=1)` | target | std of the same window |
| `RollingMin(w; lag=1)` / `RollingMax(w; lag=1)` | target | min / max of the same window |
| `Diff(k; lag=1)` | target | `y_{t-lag} - y_{t-lag-k}` |
| `Calendar(parts...)` | time | `:year`, `:quarter`, `:month`, `:weekofyear`, `:dayofweek`, `:dayofmonth`, `:dayofyear`, `:hour`, `:minute` |
| `Fourier(period, K)` | time | `sin`/`cos` harmonics on a step index continuing seamlessly into the future |
| `Exogenous(cols...)` | exogenous | future-known covariates, supplied via `new_data` at forecast time |
| `CustomFeature(name, f, minhistory)` | target | `f(history)` — any function of the target history strictly before the current row |

**No-leakage guarantee**: every target-history feature uses only information
strictly *before* the current row's target (rolling windows and differences
require `lag ≥ 1`), and the test suite asserts it.

## Strategies

- `Recursive()` — one model trained one-step-ahead; forecasts are fed back as
  pseudo-history to compute lag features for later steps. Any horizon.
- `Direct(max_horizon)` — one model per step `1..max_horizon`; forecasting
  beyond `max_horizon` errors. v1 simplification: at forecast time the direct
  models condition target-history features on the training-end history, while
  time and exogenous features vary per step.

## Hyperparameter tuning

### Level 1 — tune the base model with MLJ (no MachineLearningForecast code involved)

Any MLJ model is accepted, so MLJ's own tuning composes for free — wrap your
regressor in a `TunedModel` and pass it to `Forecaster`:

```julia
using MLJ
tuned = TunedModel(model = EvoTreeRegressor(),
                   resampling = TimeSeriesCV(nfolds=5),   # time-ordered folds!
                   tuning = Grid(),
                   range = [range(EvoTreeRegressor(), :eta, values=[0.01, 0.05, 0.1]),
                            range(EvoTreeRegressor(), :max_depth, values=[4, 6, 8])],
                   measure = rms)
fc = Forecaster(tuned; features=..., strategy=Recursive(), freq=Day(1))
```

> **⚠️ Leakage warning.** Never use MLJ's default `CV()` or
> `Holdout(shuffle=true)` resampling here — shuffled folds put future rows in
> the training folds, leaking future lag information. Use **`TimeSeriesCV`
> only**.
>
> Also note the caveat: this tunes *one-step-ahead tabular* error, which is a
> proxy — recursion error compounds over the horizon, and pipeline-level
> choices (lags, Fourier order, strategy) are invisible to it. For those, use
> `tune` below.

For Bayesian optimization of the base model, the same pattern works with
Tree-structured Parzen estimators from
[TreeParzen.jl](https://github.com/IQVIA-ML/TreeParzen.jl):

```julia
using MLJ, TreeParzen
space = Dict(:eta       => HP.LogUniform(:eta, log(0.005), log(0.3)),
             :max_depth => HP.QuantUniform(:max_depth, 3.0, 10.0, 1.0))
tuned = TunedModel(model = EvoTreeRegressor(),
                   tuning = MLJTreeParzenTuning(),
                   resampling = TimeSeriesCV(nfolds=5),   # TimeSeriesCV only!
                   range = space, n = 50, measure = rms)
fc = Forecaster(tuned; features=..., strategy=Recursive(), freq=Day(1))
```

The same one-step-ahead-proxy caveat applies.

### Level 2 — tune the whole pipeline on backtest score

`tune` searches over `Forecaster` fields — model, feature set, strategy — and
scores each candidate with a full backtest, so recursion effects and feature
choices are measured for real:

```julia
result = tune(fc, df;
    grid = (
        model    = [EvoTreeRegressor(eta=0.05), EvoTreeRegressor(eta=0.1)],
        features = [FeatureSet(Lag(1), Lag(7)),
                    FeatureSet(Lag(1), Lag(7), Fourier(365.25, 3))],
        strategy = [Recursive(), Direct(28)],
    ),
    horizon=28, initial=730, step=28, metric=smape)

result.table         # scores for every candidate (columntable)
result.best          # the winning Forecaster spec
result.best_fitted   # the winner refit on all data, ready to forecast
```

`tuner=GridSearch()` (default) takes the Cartesian product;
`tuner=RandomSearch(n; rng=...)` draws `n` candidates. Candidate failures are
caught, recorded in the table's `:error` column, and excluded from ranking —
they never abort the search.

**Bring your own strategy.** The search loop is a documented ask/tell
protocol: subtype `TuningStrategy`, implement `ask(s)` (return the next
candidate as a NamedTuple of `Forecaster` field overrides, or `nothing` to
stop) and `tell!(s, candidate, score)` (receive the backtest score, or
`missing` on failure), and pass it as `tuner=`. That is the exact interface a
sequential Bayesian optimizer needs — see the `TuningStrategy` docstring.

## Metrics

`mae`, `rmse`, `mape`, `smape`, `mase` — all `(y, ŷ) -> Float64`, lower is
better. `mape` and `smape` return fractions (0.10 = 10%);
`mase(y, ŷ; y_train, m=1)` scales by the in-sample seasonal-naive error.

## Error messages

Validation failures throw `ArgumentError` with what was wrong, the offending
values, and the fix, e.g.:

```
time column :ds has 3 gaps for freq=1 day; first gap after 2021-03-04.
Reindex your data or resample before fitting.
```

## Design notes

- **Immutable specs, explicit fitted state**: `Forecaster` is immutable; `fit`
  returns a `FittedForecaster`. Your data and model prototypes are never
  mutated.
- **`fit`/`forecast` are MachineLearningForecast's own functions** (re-exported), not
  extensions of `MLJBase.fit` or `StatsAPI.fit` — the signatures differ enough
  that sharing a generic function would mislead. (`fit` is not exported by
  `MLJ` itself, so `using MLJ, MachineLearningForecast` does not make it ambiguous.)
- **Name collisions with `MLJ`.** Four MachineLearningForecast exports share a name with an
  `MLJ` export: `mae`, `rmse`, `mape` (MLJ's measures, via
  StatisticalMeasures.jl) and `RandomSearch` (MLJ's tuning strategy, via
  MLJTuning.jl). Under `using MLJ, MachineLearningForecast` Julia leaves those four names
  undefined rather than picking a winner, so you get an `UndefVarError` (with a
  hint naming the conflict) the moment you *use* one. Pick what you mean
  explicitly:

  ```julia
  using MLJ, MachineLearningForecast
  import MachineLearningForecast: mae, rmse, mape          # MachineLearningForecast's (y, ŷ) -> Float64 metrics
  backtest(fc, df; horizon=28, initial=730, metrics=(mae, rmse))

  tune(fc, df; grid=..., tuner=MachineLearningForecast.RandomSearch(20),  # not MLJ's RandomSearch
       horizon=28, initial=730)
  ```

  The other 28 exports are collision-free. Note that MachineLearningForecast's `mae`/`rmse`
  are plain functions of two vectors, whereas MLJ's are measure objects — they
  are not interchangeable, which is exactly why the choice must be explicit.
- **Multiple dispatch over configuration flags**: strategies, features, and
  tuning strategies are types; behavior is chosen by dispatch.

## Contributing

Bug reports, documentation fixes and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, the design
principles the code follows, and the dispatch contract for adding a new feature
type or tuning strategy. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).

## Roadmap (not in v1)

Probabilistic forecasts (conformal residuals from `backtest`), panel/
multi-series fitting, `DirRec` strategy, target transforms, continuous search
spaces for tuning.
