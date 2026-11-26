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
rows up to the fold's origin only. The pools and samples of
[`ConformalSeasonalPool`](@ref) are built from the fitted training window only, so a
`backtest` fold's intervals and sample scores never see its actuals; a test changes the
values after fold 1's test window and checks that nothing in fold 1 moves.

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

**`Exogenous` and `backtest`.** A backtest uses the realised values of the
covariates, so it scores ex-post forecasts (Hyndman & Athanasopoulos 2021,
§7.6). For covariates not known at the origin (a competitor's price, the
weather) this typically understates the error of the real, ex-ante forecast,
which also carries the error of forecasting the covariate. For covariates known
in advance (calendar effects, planned promotions, prices you set) ex-ante and
ex-post agree. To measure ex-ante error, supply in the held-out rows the
covariate values that were available at the origin (a forecast, or a lagged
column). The recipes below build lagged columns.

**Panel `backtest`.** A fold does not score a series that has no rows up to the
fold's origin, too few of them to train on (it is skipped with a warning), or
no actuals after it, so the set of scored series can change from fold to fold.
With `windows`, each series is cut at its own end, as in the M competitions. If
the series are not aligned in calendar time, the training set of a fold can
contain values from after another series' cutoff, a known pitfall of global
models (Hewamalage, Ackermann & Bergmeir 2023). With `initial`, all series are
cut at the same timestamp, which avoids it.

**Gaps and missing values.** The package refuses a time column with gaps,
duplicates or missing timestamps, and a target with `missing` or non-finite
values. Fill them causally (forward fill, or the mean of past values) before
fitting, because a fill that uses later values, such as interpolation, leaks.
`Exogenous` columns may hold `NaN`, which reaches the model as a missing marker,
but not `missing` or `±Inf`.

**Aggregation of metrics.** Unscaled metrics pool the fold's rows; scaled
metrics are per series, weighted by scored points; [`summarize`](@ref) gives the
unweighted mean over folds. Scaled metrics are those whose
[`metric_inputs`](@ref MachineLearningForecast.metric_inputs) is `Scaled()`, such as
[`mase`](@ref) and [`rmsse`](@ref), or their callable forms [`Mase`](@ref) and
[`Rmsse`](@ref). Scaled metrics follow Hyndman & Koehler
(2006): each series' errors are divided by that series' own training-window
naive error, then averaged over series (here weighted by scored points, our
choice). One series with a constant training window has a zero scale, which
makes that fold's score, and so its mean over folds in [`summarize`](@ref), `Inf`.
Unscaled metrics are pooled across series, so on a panel they are dominated by the
large-scale series; use a scaled metric to compare series of different scale.
On intermittent series (many zeros), MAE-based measures, including MASE, can
favour forecasts close to zero; the M5 competition used a squared-error scaled
measure (RMSSE) for this reason (Makridakis, Spiliotis & Assimakopoulos 2022);
use [`rmsse`](@ref) for such series. Both take the scale from the whole training
window; for the M5 rule, which starts it at the first non-zero value, see the recipe
below.

**Tuning base models with MLJ.** If you tune the base model with MLJ's
`TunedModel`, prefer `TimeSeriesCV` to `CV()`. Plain K-fold CV trains on rows
that come after some of its test rows. For a stationary series, pure lag
features and a model whose one-step residuals are not autocorrelated, this does
not bias the estimate (Bergmeir, Hyndman & Koo 2018); it does for a trending or
otherwise non-stationary series (Cerqueira, Torgo & Mozetič 2020) or when the
model underfits and its residuals are autocorrelated. The result does not cover
covariates or multi-step targets. `TimeSeriesCV` never trains on rows after its
test rows on a single series, so it avoids these problems there. On a panel the
pooled frame is ordered by series, not by time, so score on a validation window
or use `tune` instead. [Tuning the models](tuning.md) scores on a validation
window that comes after the training window.

## Choosing a strategy

[`Recursive`](@ref) fits one model on one-step-ahead rows, where the lag
features are the true past values. At forecast time it runs that model on its
own earlier predictions, and any horizon can be forecast.

[`Direct`](@ref) fits one model per step `1..max_horizon`. Model `i` predicts
the target `i` steps ahead from the history at the forecast origin, so no
prediction is fed back. It needs `max_horizon` models, cannot forecast past
`max_horizon`, and every step sees the same target history, which is why the
`Lag` convention above matters. On a panel, a series with fewer than
`max_horizon` training rows trains only the step models it has rows for, and
`fit` warns.

Neither strategy is better in general. `Recursive` estimates one model
efficiently but is biased when the one-step model is misspecified, and the
errors it feeds back accumulate. `Direct` is more robust to such misspecification
and to non-stationarity such as unit roots, and feeds nothing back, but estimates
each horizon separately, which costs variance, and its forecasts at neighbouring
steps need not be coherent (Marcellino, Stock & Watson 2006; Chevillon 2007; Ben
Taieb et al. 2012). Empirically, iterated (`Recursive`) forecasts are often at
least as accurate on stationary data (Marcellino, Stock & Watson 2006). Compare both with
[`backtest`](@ref), or let [`tune`](@ref) search over `strategy`.

## Probabilistic forecasts

[`ConformalSeasonalPool`](@ref) and the `levels` keyword of [`forecast`](@ref) and
[`backtest`](@ref) give intervals and predictive samples. Levels are percentages in (1,
100), never fractions; `y_hat` stays the point forecast; the draws are reproducible on the
same Julia version, and every series (and fold) starts from a copy of the same generator;
sample scores pool the rows of a fold; the method has no coverage guarantee.
[Probabilistic forecasts](probabilistic.md) gives the details, a worked example and the
choices that are ours.

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
`future` table. See [`featurevalues!`](@ref MachineLearningForecast.featurevalues!)
and [`anchor`](@ref MachineLearningForecast.anchor) for the contract.

### A parent aggregate as a covariate

The total over all series (the parent in a hierarchy) is not known for the
forecast period, but its value `H` steps ago is, as long as `H` is at least the
forecast horizon. Shift it by `H`, then pass it through [`Exogenous`](@ref). The recipe assumes
that every series ends on the same date and that the dates are contiguous.
The column holds only values from before the forecast origin in `fit`, in
`forecast` and in `backtest`, so the backtest measures the ex-ante error.

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
future_dates = collect(dates[end] + Day(1):Day(1):dates[end] + Day(H))
future_data = (unique_id = repeat(stores, inner = H), ds = repeat(future_dates, length(stores)),
               parent_lag = [lagged[d] for _ in stores for d in future_dates])
forecast(panel_fitted, H; future = future_data)
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
future = (ds = df.ds[end] .+ Day.(1:h), price_lag14 = price[(n - k + 1):(n - k + h)])
forecast(fitted, h; future)
```

### Scaling from the first non-zero value (the M5 rule)

[`mase`](@ref) and [`rmsse`](@ref) take the scale from the whole training window
(Hyndman & Koehler 2006; Hyndman & Athanasopoulos 2021, §5.8). The M5 competition
computed it only from the first non-zero value on, because products there launch
late and the leading zeros mean "not yet launched" (Makridakis, Spiliotis &
Assimakopoulos 2022; M Open Forecasting Center 2020). To score that way, trim
`y_train` to start at its first non-zero value, that value included, and delegate:

```julia
using MachineLearningForecast
import MachineLearningForecast: metric_inputs, Scaled

function from_first_nonzero(y_train)
    i = findfirst(!iszero, y_train)
    return i === nothing ? y_train : y_train[i:end]
end

rmsse_m5(y, ŷ; y_train, m = 1) = rmsse(y, ŷ; y_train = from_first_nonzero(y_train), m)
mase_m5(y, ŷ; y_train, m = 1) = mase(y, ŷ; y_train = from_first_nonzero(y_train), m)
metric_inputs(::typeof(rmsse_m5)) = Scaled()
metric_inputs(::typeof(mase_m5)) = Scaled()

rmsse_m5([2.0, 3.0], [2.0, 2.0]; y_train = [0.0, 0.0, 0.0, 4.0, 6.0, 5.0, 7.0])
```

Pass `rmsse_m5` or `mase_m5` to [`backtest`](@ref) or [`tune`](@ref) like any
metric. A metric with a fixed `m`, such as `Mase(12)`, is a plain `const` (see
`mase12` in [Models](models.md)).

Trim only when leading zeros mean "not yet launched". Where they are genuine zero
demand, they belong to the series and trimming discards them. An all-zero window has
nothing to trim, so it is passed on unchanged: the scale is zero, and the metric warns
and returns `Inf`, as `rmsse` and `mase` do for any constant window; the same happens
when the values from the first non-zero on are constant. A window with `m` or fewer
values from its first non-zero value on is too short for the scale, so `rmsse` and
`mase` throw their usual `ArgumentError` (`length(y_train) > m`), and a `backtest`
over such a series fails.

### Scores per horizon step and per series

`BacktestResult.forecasts` has one row per fold, step and series with `:step`,
`:y` and `:y_hat` (and the id column for a panel), so any score is a few lines.
This one scores `panel_fc`, `train` and `H` from the parent aggregate recipe:

```julia
using Statistics
r = backtest(panel_fc, train; horizon = H, windows = 3)
f = r.forecasts
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
summarize(backtest(naive, df; horizon = 7, windows = 3, metrics = (mae, smape)))
```

### A seasonal naive forecaster as a baseline

Naive and seasonal naive forecasts are the standard benchmarks (Hyndman &
Athanasopoulos 2021, §5.2). When the data are seasonal with period `m`, backtest
[`SeasonalNaive`](@ref)`(m)`: a model that returns `y_lag_m`, with `Lag(m)` as the
only feature, under `Recursive`. For a horizon above `m` the fed-back
predictions equal the lagged values, so each step repeats the value one or more
seasons earlier, which is the seasonal naive forecast. Under `Direct` it does
not: `Lag(m)` is then `y[t0-m+1]` at every step, as explained above.

```julia
m = 7
snaive = Forecaster(SeasonalNaive(m); features = FeatureSet(Lag(m)), freq = Day(1))
summarize(backtest(snaive, df; horizon = 14, windows = 3, metrics = (mae, smape)))
```

## References

- Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A review and
  comparison of strategies for multi-step ahead time series forecasting based on
  the NN5 forecasting competition. *Expert Systems with Applications*, 39(8),
  7067–7083. <https://doi.org/10.1016/j.eswa.2012.01.039>
- Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of
  cross-validation for evaluating autoregressive time series prediction.
  *Computational Statistics & Data Analysis*, 120, 70–83.
  <https://doi.org/10.1016/j.csda.2017.11.003>
- Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating time series
  forecasting models: an empirical study on performance estimation methods.
  *Machine Learning*, 109(11), 1997–2028.
  <https://doi.org/10.1007/s10994-020-05910-7>
- Chevillon, G. (2007). Direct multi-step estimation and forecasting. *Journal of
  Economic Surveys*, 21(4), 746–785.
  <https://doi.org/10.1111/j.1467-6419.2007.00518.x>
- Ferro, C. A. T., Richardson, D. S., & Weigel, A. P. (2008). On the effect of ensemble
  size on the discrete and continuous ranked probability scores. *Meteorological
  Applications*, 15(1), 19–24. <https://doi.org/10.1002/met.45>
- Gneiting, T., & Katzfuss, M. (2014). Probabilistic forecasting. *Annual Review of
  Statistics and Its Application*, 1, 125–151.
  <https://doi.org/10.1146/annurev-statistics-062713-085831>
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
  <https://doi.org/10.1198/016214506000001437>
- Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation for
  data scientists: common pitfalls and best practices. *Data Mining and
  Knowledge Discovery*, 37(2), 788–832.
  <https://doi.org/10.1007/s10618-022-00894-5>
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and
  practice* (3rd ed.). OTexts. <https://otexts.com/fpp3/>
- Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. *The
  American Statistician*, 50(4), 361–365.
  <https://doi.org/10.1080/00031305.1996.10473566>
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast
  accuracy. *International Journal of Forecasting*, 22(4), 679–688.
  <https://doi.org/10.1016/j.ijforecast.2006.03.001>
- M Open Forecasting Center (2020). *The M5 competition: competitors' guide*.
  University of Nicosia.
  <https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf>
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
  competition: results, findings, and conclusions. *International Journal of
  Forecasting*, 38(4), 1346–1364.
  <https://doi.org/10.1016/j.ijforecast.2021.11.013>
- Manokhin, V. (2026). Training-free probabilistic time-series forecasting with conformal
  seasonal pools. arXiv:2605.03789. <https://arxiv.org/abs/2605.03789>
- Marcellino, M., Stock, J. H., & Watson, M. W. (2006). A comparison of direct and
  iterated multistep AR methods for forecasting macroeconomic time series.
  *Journal of Econometrics*, 135(1–2), 499–526.
  <https://doi.org/10.1016/j.jeconom.2005.07.020>
- Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive confidence
  machines for regression. In *Machine Learning: ECML 2002*, LNCS 2430, 345–356.
  Springer. <https://doi.org/10.1007/3-540-36755-1_29>
- Zamo, M., & Naveau, P. (2018). Estimation of the continuous ranked probability score
  with limited information and applications to ensemble weather forecasts. *Mathematical
  Geosciences*, 50(2), 209–234. <https://doi.org/10.1007/s11004-017-9709-7>
