# MachineLearningForecast.jl

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="docs/src/assets/readme-banner-dark.svg">
    <img src="docs/src/assets/readme-banner.svg"
         alt="MachineLearningForecast.jl: time-series forecasting with any MLJ model"
         width="640">
  </picture>
</p>

**Generic, composable time-series forecasting for Julia, built on the MLJ model
interface.** Any MLJ `Deterministic` regressor is a valid base model, and a
`Probabilistic` one forecasts its predicted mean — random forests,
[EvoTrees](https://github.com/Evovest/EvoTrees.jl) gradient boosting, XGBoost,
LightGBM, CatBoost, linear models — MachineLearningForecast never hard-codes a
learner.

- **Declarative feature engineering**: lags, rolling statistics, differences,
  calendar features, Fourier seasonal terms, future-known exogenous covariates,
  and custom features, composed as first-class Julia objects.
- **Pluggable forecasting strategies**: `Recursive` (one model, iterated) and
  `Direct` (one model per horizon step), chosen by dispatch — extensible by
  design.
- **Panel (multi-series) forecasting**: set `id=` and MachineLearningForecast fits *one
  global model* across every series, so short or noisy series borrow strength
  from the rest. Series may be ragged, and features never reach across a series
  boundary.
- **Backtesting**: expanding-window time-series cross-validation with pluggable
  metrics.
- **Pipeline tuning**: grid/random search over models, feature sets, and
  strategies, scored on backtest error, with a documented ask!/tell! interface
  for plugging in your own search strategy (e.g. a Bayesian optimizer).
- **Tables.jl-native**: accepts *any* Tables.jl-compatible source (a NamedTuple
  of vectors, `CSV.File`, a `DataFrame`, ...) and returns plain columntables
  (NamedTuples of vectors) that any table sink understands. MachineLearningForecast's
  only table dependency is Tables.jl — bring whatever table type you like.

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

# Or hold out the last 3 folds, counted back from the end
results = backtest(fc, df; horizon=28, windows=3)
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
| `Fourier(period, K)` | time | `sin`/`cos` harmonics on a step index continuing seamlessly into the future; on a panel, the same date has the same values in every series |
| `Exogenous(cols...)` | exogenous | future-known covariates, supplied via `new_data` at forecast time |
| `CustomFeature(name, f, minhistory)` | target | `f(history)` — any function of the target history strictly before the current row |

**No-leakage guarantee**: every target-history feature uses only information
strictly *before* the current row's target (rolling windows and differences
require `lag ≥ 1`), and the test suite asserts it. A backtest uses the realised values
of the covariates, so it scores ex-post forecasts (Hyndman & Athanasopoulos 2021, §7.6).
For covariates not known at the origin (a competitor's price, the weather) this
typically understates the error of the real, ex-ante forecast, which also carries the
error of forecasting the covariate. For covariates known in advance (calendar effects,
planned promotions, prices you set) ex-ante and ex-post agree. To measure ex-ante
error, supply in the held-out rows the covariate values that were available at the
origin (a forecast, or a lagged column).

## Target transforms

A transform reshapes each series' target before features are built, and maps
the forecasts back. Its state is fitted on each series' own training history,
and `backtest` refits it in every fold, so it never sees the forecast window.
Tree models cannot forecast beyond the levels they were trained on, so for
trending series a transform often matters more than the hyperparameters.

```julia
fc = Forecaster(EvoTreeRegressor(); freq=Month(1), id=:unique_id,
                features=FeatureSet(Lag.(1:12)..., RollingMean(12), Calendar(:month)),
                transform=TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12)))
```

| Transform | Effect |
|---|---|
| `NoTransform()` | the default: the target as given |
| `MeanScale()` / `LastScale(k)` | divide by the mean of the whole history / of its last `k` values |
| `Deseasonalize(period)` | classical multiplicative adjustment, for series passing the M4 seasonality test |
| `LogRatio(window)` | `log(y_t / mean(y[t-window:t-1]))`: scale-free and stationary |
| `Difference(lag)` | `y_t - y_{t-lag}` |
| `TransformChain(t...)` | apply in order, undo in reverse |

Your own transform subtypes `TargetTransform` and implements
`fit_transform(t, y) -> (z, state)` and `untransform(t, ẑ, state)`. Both are public
but not exported: `import MachineLearningForecast: fit_transform, untransform`.

## Strategies

- `Recursive()` — one model trained one-step-ahead; forecasts are fed back as
  pseudo-history to compute lag features for later steps. Any horizon.
- `Direct(max_horizon)` — one model per step `1..max_horizon`; forecasting
  beyond `max_horizon` errors. At forecast time, the direct models condition their
  target-history features on the history up to the forecast origin, while exogenous and
  time features vary per step. Under `Direct`, `Lag(k)` counts back from the forecast
  origin, so at horizon `h` it is `y[t0-k+1]` (`t0` is the index of the last observed
  value), not `y[target-k]`; `Lag(m)` is in phase with the target only when
  `h = 1 (mod m)`; use `Lag.(1:m)`, which holds an in-phase lag at every `h`, or
  `Recursive`.

Neither strategy is better in general. `Recursive` estimates one model efficiently but
is biased when the one-step model is misspecified, and the errors it feeds back
accumulate. `Direct` is more robust to such misspecification and to non-stationarity
such as unit roots, and feeds nothing back, but estimates each horizon separately, which
costs variance, and its forecasts at neighbouring steps need not be coherent
(Marcellino, Stock & Watson 2006; Chevillon 2007; Ben Taieb et al. 2012). Empirically,
iterated (`Recursive`) forecasts are often at least as accurate on stationary data
(Marcellino, Stock & Watson 2006). Compare both with `backtest`.

[Leakage and conventions](docs/src/conventions.md) compares the two strategies and lists
the rules each feature, backtest and metric follows, with recipes for custom features.
Its "References" section lists the sources cited in this README.

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

> **Leakage warning.** If you tune the base model with MLJ's `TunedModel`, prefer
> `TimeSeriesCV` to `CV()`. Plain K-fold CV trains on rows that come after some of its
> test rows. For a stationary series, pure lag features and a model whose one-step
> residuals are not autocorrelated, this does not bias the estimate (Bergmeir, Hyndman
> & Koo 2018); it does for a trending or otherwise non-stationary series (Cerqueira,
> Torgo & Mozetič 2020) or when the model underfits and its residuals are
> autocorrelated. The result does not cover covariates or multi-step targets.
> `TimeSeriesCV` never trains on rows after its test rows on a single series, so it
> avoids these problems there. On a panel the pooled frame is ordered by series, not by
> time, so score on a validation window or use `tune` instead.
>
> Also note the caveat: this tunes *one-step-ahead tabular* error, which is a
> proxy — recursion error accumulates over the horizon, and pipeline-level
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
                   resampling = TimeSeriesCV(nfolds=5),   # time-ordered folds!
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
they never abort the search. A non-finite mean score (`NaN` or `Inf`) counts as
a failure. If every candidate fails, `tune` throws an `ErrorException` listing
the distinct error messages.

**Bring your own strategy.** The search loop is a documented ask!/tell!
protocol: subtype `TuningStrategy`, implement `ask!(s)` (return the next
candidate as a NamedTuple of `Forecaster` field overrides, or `nothing` to
stop) and `tell!(s, candidate, score)` (receive the backtest score, or
`missing` on failure), and pass it as `tuner=`. Both functions are public but not
exported: `import MachineLearningForecast: ask!, tell!`. That is the exact interface a
sequential Bayesian optimizer needs — see the `TuningStrategy` docstring.

## Panel (multi-series) forecasting

Pass `id=` and give MachineLearningForecast a long-format table — one row per (series,
timestamp). Every feature is materialised **within** each series, then the rows
of all series are pooled to fit a single global model:

```julia
panel = (unique_id = repeat(["store_a", "store_b", "store_c"], inner = 365),
         ds        = repeat(collect(Date(2023,1,1):Day(1):Date(2023,12,31)), 3),
         y         = vcat(sales_a, sales_b, sales_c))

fc = Forecaster(EvoTreeRegressor(nrounds = 200);
                features = FeatureSet(Lag(1), Lag(7), RollingMean(7),
                                      Calendar(:dayofweek)),
                strategy = Recursive(),
                freq     = Day(1),
                target   = :y,
                time     = :ds,
                id       = :unique_id)      # <- the only change

fitted = fit(fc, panel)
nseries(fitted)                              # 3
forecast(fitted, 28)                         # (unique_id, ds, y_hat), 3 x 28 rows
```

Notes:

- **One global model.** `fitted.models` holds a single trained model (or
  `max_horizon` of them under `Direct`), trained on the pooled rows — not one
  model per series.
- **No cross-series leakage.** `Lag(1)` at the first row of a series is
  undefined, never the previous series' last value. Each series drops its own
  `minhistory` rows.
- **Ragged panels are fine.** Series need not share a start date, an end date
  or a length, and each forecasts forward from *its own* last timestamp. Series
  shorter than the feature set needs are skipped with a warning rather than
  aborting the fit.
- **Exogenous covariates** join on `(id, time)`, so `new_data` must carry the id
  column too.
- **Backtesting** with `initial` cuts folds on the *global timestamp grid* — `initial`,
  `step` and `horizon` count distinct timestamps, not rows. With `windows` it instead
  counts each series' folds back from *its own* end, as the M competitions hold out each
  series' last values. If the series are not aligned in calendar time, the training set
  of a fold can contain values from after another series' cutoff, a known pitfall of
  global models (Hewamalage, Ackermann & Bergmeir 2023); with `initial`, all series are
  cut at the same timestamp, which avoids it. Either way it scores by joining forecasts
  to actuals on `(id, time)`, and scaled metrics such as `mase` use each series' own
  training values. `tune` takes `initial` or `windows` the same way. A fold does not
  score a series that has no rows up to the fold's origin, too few of them to train on
  (it is skipped with a warning), or no actuals after it, so the set of scored series
  can change from fold to fold.
- Per-series state is available as `fitted.series`, a vector of
  `SeriesState`; for a single series the familiar `fitted.y_history`,
  `fitted.t_last` and `fitted.n_train` still work.

## Metrics

`mae`, `rmse`, `mape`, `smape`, `mase`, `rmsse` — all `(y, ŷ) -> Float64`, lower is
better. `mape` and `smape` return fractions (0.10 = 10%);
`mase(y, ŷ; y_train, m=1)` scales by the in-sample seasonal-naive error;
`rmsse(y, ŷ; y_train, m=1)` is its squared-error counterpart, the RMSE divided by the
root of the in-sample mean squared seasonal-naive error (M5's RMSSE, except that M5
starts the scale at the first non-zero value, while `rmsse` uses the whole training
window).
In `backtest`, unscaled metrics pool the fold's rows; scaled metrics are per series,
weighted by scored points; the overall row is the unweighted mean over folds.
Scaled metrics follow Hyndman & Koehler (2006): each series' errors are divided by that
series' own training-window naive error, then averaged over series (here weighted by
scored points, our choice). Unscaled metrics are pooled across series, so on a panel
they are dominated by the large-scale series; use a scaled metric to compare series of
different scale. On intermittent series (many zeros), MAE-based measures, including
MASE, can favour forecasts close to zero; the M5 competition used a squared-error
scaled measure (RMSSE) for this reason (Makridakis, Spiliotis & Assimakopoulos 2022);
use `rmsse` for such series.

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

  The other 40 exports are collision-free. Note that MachineLearningForecast's `mae`/`rmse`
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

## Roadmap

- **Box-Cox** and other power transforms as built-in `TargetTransform`s.
- **Static features**: per-series constants given once, not repeated in
  `new_data`, and per-series encodings (a series-id feature).
- **More history features**: expanding mean, exponentially weighted mean,
  seasonal rolling statistics, rolling quantiles, and lags of exogenous columns.
- **Prediction intervals** around the point forecasts.
- **Model reuse**: forecast series not seen in training, update a fitted model
  with new observations without refitting, access the feature matrix, fit
  several models in one call, and tested save/load.
- `DirRec` strategy and continuous search spaces for tuning.
