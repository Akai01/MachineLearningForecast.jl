# Using different models

Any MLJ `Deterministic` regressor can be the base model of a
[`Forecaster`](@ref). To change the learner you swap the model and nothing
else: the features, the strategy and the data stay as they are. This page fits
six learners from six packages to the same panel of 1399 monthly series from
the M3 competition, each as one global model with the same specification and
its package's default hyperparameters, apart from one XGBoost setting explained
in [The models](@ref). It scores them on the M3 holdout next to two baselines
and published results.

The code below runs top to bottom as one script, and the outputs shown are
from that run. Besides MachineLearningForecast it uses the standard libraries
Downloads, Dates, Statistics and Printf, plus these packages, none of which is
a dependency of MachineLearningForecast: ZipArchives, EvoTrees,
MLJDecisionTreeInterface, MLJLinearModels, MLJXGBoostInterface, LightGBM and
NearestNeighborModels. Add them to your environment with `Pkg.add`.
MachineLearningForecast is not registered yet, so add it by URL:
`Pkg.add(url="https://github.com/Akai01/MachineLearningForecast.jl")`.

## The data

The series are the 1428 monthly series of the M3 competition (Makridakis, S.
and Hibon, M. (2000), "The M3-Competition: results, conclusions and
implications", *International Journal of Forecasting* 16(4), 451–476). The
code downloads them from the Monash Time Series Forecasting Archive (Godahewa,
R., Bergmeir, C., Webb, G. I., Hyndman, R. J. and Montero-Manso, P. (2021),
"Monash Time Series Forecasting Archive", *NeurIPS Track on Datasets and
Benchmarks*): the "M3 Monthly Dataset" on Zenodo,
[DOI 10.5281/zenodo.4656298](https://doi.org/10.5281/zenodo.4656298), licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The zip file holds one file in the archive's `.tsf` format: comment and
`@attribute` lines, then, after `@data`, one series per line as
`name:start:values`, for example `T1:1990-01-01 00-00-00:2640,2640,2160,...`.

```julia
using Downloads, ZipArchives, Dates

url = "https://zenodo.org/api/records/4656298/files/m3_monthly_dataset.zip/content"
archive = ZipReader(take!(Downloads.download(url, IOBuffer())))
tsf = zip_readentry(archive, "m3_monthly_dataset.tsf", String)

function parse_series(line)
    name, start, values = split(line, ':')
    return (id = String(name), start = Date(start[1:10]),
            y = parse.(Float64, split(values, ',')))
end

lines = eachline(IOBuffer(split(tsf, "@data")[2]))
series = [parse_series(l) for l in lines if !isempty(l)]
dated = filter(s -> s.start != Date(1900, 1, 1), series)
println(length(series), " series, ", length(dated), " with a start date")
```

```text
1428 series, 1399 with a start date
```

The file's header notes that the start dates of the last 29 series are
unknown and were set to 1900-01-01. The feature set below reads the month from
each timestamp, so those series would get made-up months: they are left out,
which leaves 1399 series.

## The holdout

The evaluation follows the M3 protocol: the last 18 months of each series are
held out, and everything before them is the series' training history. All the
steps below use the training history only; the held-out months are used only
to score the forecasts.

The M3 series span different calendar years (they end in 73 different months,
from 1864 to 2005), so no single cut date would hold out each series' own last
18 months. [`backtest`](@ref) with `windows = 1` cuts each series at its own
end instead, and with `horizon = 18` it holds out exactly the M3 months.
`train` and `test` split a series the same way for the two baselines below.

```julia
using Statistics

h = 18
train(s) = s.y[1:(end - h)]
test(s) = s.y[(end - h + 1):end]
```

## The transform

One global model fits all series with one loss, so the target has to mean the
same thing in every series. M3 series differ in level, many trend, and many
are seasonal. Tree models, such as boosted trees and random forests, predict
averages of the targets they were trained on, so they cannot extrapolate
beyond the levels they saw. Dividing each series by its whole-history mean
does not fix that: a series that trends up ends far from its old mean, and the
model pulls its forecasts back towards old levels. The transform below models
each month relative to the months just before it instead, and rebuilds the
forecasts from each series' own last year.

For each series, from its training history only, three built-in
[`TargetTransform`](@ref)s are applied in turn:

1. **Level,** [`LastScale`](@ref)`(12)`. Divide the series by ``m``, the mean
   of its last 12 values, so every series is near 1 at its forecast origin.
2. **Seasonal adjustment,** [`Deseasonalize`](@ref)`(12)`. The series is
   seasonal if its autocorrelation at lag 12, ``r_{12}``, passes the 90% test
   that the M4 competition's benchmarks use:
   ``|r_{12}| > 1.645 \sqrt{(1 + 2 \sum_{k=1}^{11} r_k^2) / n}``
   on ``n \ge 36`` points (every M3 training history has at least 48). A
   seasonal series gets the classical multiplicative indices: each value is
   divided by its centred 2×12 moving average, these ratios are averaged by
   position in the year, and the 12 averages are normalised to mean 1. A
   non-seasonal series gets indices of 1. Dividing by the indices gives the
   seasonally adjusted series ``x``.
3. **Log ratio,** [`LogRatio`](@ref)`(12)`. ``z_t = \log(x_t / \bar{x}_t)``,
   where ``\bar{x}_t`` is the mean of the 12 values before ``x_t``: each month
   relative to the year before it. A ratio has no unit, so ``z`` does not
   depend on the level of step 1 and is on the same scale in every series. It
   is 12 values shorter than the history.
4. **Inverse.** Forecasts of ``z`` are turned back one month at a time:
   ``\exp(\hat{z})`` times the mean of the previous 12 values of ``x``, rebuilt
   ones included, then times that month's seasonal index and ``m``.

[`TransformChain`](@ref) applies the three in order and undoes them in
reverse. Given as the `transform` of a [`Forecaster`](@ref), it is fitted on
each series' own training history, refitted in every `backtest` fold, and the
forecasts come back on the original scale.

Positions in the year are counted from each series' first month, so the
forecast for month ``n + j`` of a history of length ``n`` gets the index at
position `mod1(n + j, 12)`. The log needs positive values: every M3 training
history is positive, and `LogRatio` and `Deseasonalize` throw an error naming
the series and the value otherwise. A series with zeros or negative values
needs a shift first.

This setup was chosen on a validation window, the 18 months before the
holdout, among several target transforms and feature sets, and was then run
once on the holdout.

```julia
using MachineLearningForecast

chain = TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))
```

## The panel

`long` puts series into one long-format table with columns `unique_id`, `ds`
and `y`, each series whole, its held-out months included: `backtest` cuts them
off. The transform drops the first 12 months of each training history, which
have no year before them.

```julia
long(ss) = (unique_id = reduce(vcat, [fill(s.id, length(s.y)) for s in ss]),
            ds = reduce(vcat, [s.start .+ Month.(0:(length(s.y) - 1)) for s in ss]),
            y = reduce(vcat, [s.y for s in ss]))
panel = long(dated)
println(length(panel.y), " rows")
```

```text
165503 rows
```

## One specification for every learner

Every learner gets the same [`FeatureSet`](@ref) (lags 1 to 12 of ``z``, its
12-month rolling mean and the calendar month), the same [`Recursive`](@ref)
strategy and the transform, with `id = :unique_id` to fit one model across the
panel. The month comes from [`Calendar`](@ref). On a panel,
[`Fourier`](@ref) would mark the same month in every series as well, since it
counts steps from the earliest series' first month, although these series
start in different months (1155 in January, 198 in October, 46 in others).

```julia
features = FeatureSet(Lag.(1:12)..., RollingMean(12), Calendar(:month))
spec(model) = Forecaster(model; features, strategy = Recursive(), freq = Month(1),
                         id = :unique_id, transform = chain)
```

## Scoring

`evaluate` runs [`backtest`](@ref) with `windows = 1` on a panel: it fits one
model on every series' training history, forecasts 18 months for every series
(each from its own last training month), and scores the forecasts with
[`smape`](@ref) and [`mase`](@ref) on the original scale. The M3 benchmarks
scale MASE by the seasonal naive error, so `mase12` calls `mase` with
seasonality `m = 12`, and
[`needs_ytrain`](@ref MachineLearningForecast.needs_ytrain) tells `backtest`
to pass it each series' own training values. `backtest` reports MASE averaged
over the series and sMAPE over all forecast months, which is the same average
over the series, since each has 18. sMAPE is shown in percent.

!!! note "Why sMAPE"
    This page reports sMAPE only because the M3 competition did, so the
    results can be compared with published ones; it does not recommend it.
    Hyndman, R. J. and Koehler, A. B. (2006), "Another look at measures of
    forecast accuracy", *International Journal of Forecasting* 22(4),
    679–688, found the measures used in the M3 competition inadequate in
    commonly occurring situations, such as values close to zero, and proposed
    MASE as the standard for comparing accuracy across series. MASE is
    reported alongside for that reason.

There are two baselines, scored the same way series by series. The seasonal
naive forecast repeats each series' last 12 training months. Naive2, one of
the M3 competition's benchmarks, repeats the last seasonally adjusted value and
puts the season back. [`fit_transform`](@ref MachineLearningForecast.fit_transform)
with `Deseasonalize(12)` does the adjustment, with the same seasonal indices as
step 2 of the transform, since they do not depend on the level, and
[`untransform`](@ref MachineLearningForecast.untransform) the inverse. Neither is
exported, so the code imports them. A transform of your own that drops rows,
as `LogRatio` and `Difference` do, declares them with
[`dropped_rows`](@ref MachineLearningForecast.dropped_rows), so that `fit` skips
a series too short for it instead of failing on it.

```julia
import MachineLearningForecast: fit_transform, untransform

mase12(y, ŷ; y_train) = mase(y, ŷ; y_train, m = 12)
MachineLearningForecast.needs_ytrain(::typeof(mase12)) = true

function evaluate(model, data = panel)
    t0 = time()
    r = backtest(spec(model), data; horizon = h, windows = 1, metrics = (smape, mase12))
    seconds = time() - t0
    s, m = r.metrics.value[r.metrics.fold .== 1]
    return 100 * s, m, seconds
end

function scores(forecasts)
    smapes = [smape(test(s), forecasts[s.id]) for s in dated]
    mases = [mase12(test(s), forecasts[s.id]; y_train = train(s)) for s in dated]
    return 100 * mean(smapes), mean(mases)
end

function naive2(y)
    z, state = fit_transform(Deseasonalize(12), y)
    return untransform(Deseasonalize(12), fill(z[end], h), state)
end

snaive = Dict(s.id => repeat(train(s)[(end - 11):end], 2)[1:h] for s in dated)
naive2s = Dict(s.id => naive2(train(s)) for s in dated)
```

## The models

Each model is used with its package's default hyperparameters, except
XGBoost's `base_score`, set below. Where a model draws random numbers, its seed
is fixed, so the results reproduce.

| Model | Package | Kind |
|---|---|---|
| `EvoTreeRegressor` | EvoTrees | gradient-boosted trees |
| `RandomForestRegressor` | MLJDecisionTreeInterface (DecisionTree.jl) | random forest |
| `LinearRegressor` | MLJLinearModels | least squares |
| `XGBoostRegressor` | MLJXGBoostInterface (XGBoost.jl) | gradient-boosted trees |
| `LGBMRegressor` | LightGBM | gradient-boosted trees |
| `KNNRegressor` | NearestNeighborModels | k nearest neighbours |

The linear model is plain least squares, which has no hyperparameter. A
penalised one such as `RidgeRegressor` needs its penalty chosen for the scale
of the features, which would tune one learner and not the others. LightGBM
does not export its MLJ model, so it is reached as
`LightGBM.MLJInterface.LGBMRegressor`, and `verbosity = -1` silences its
training log.

Boosted trees start from a constant prediction and add trees to correct it.
LightGBM and EvoTrees start from the mean of the target, but MLJ's
`XGBoostRegressor` starts from its `base_score` hyperparameter, which defaults
to 0.5. The target ``z`` here is a log ratio centred near 0, so XGBoost is
given `base_score = 0.0`, its natural centre, which puts it on the same footing
as the other two. With the default 0.5, every XGBoost model first has to
unlearn an offset of 0.5: a small learning rate or few rounds then leave part
of it in every forecast. The setting was changed after a first run on the
holdout, in which XGBoost with the default 0.5 scored 13.85 sMAPE and 0.860
MASE. [Tuning the models](tuning.md) measures the effect on two such
configurations.

```julia
using EvoTrees, MLJDecisionTreeInterface, MLJLinearModels, MLJXGBoostInterface
using LightGBM, NearestNeighborModels

models = [
    EvoTreeRegressor(seed = 1),
    RandomForestRegressor(rng = 1),
    LinearRegressor(),
    XGBoostRegressor(seed = 1, base_score = 0.0),
    LightGBM.MLJInterface.LGBMRegressor(seed = 1, verbosity = -1),
    KNNRegressor(),
]
```

## Results

Each model is evaluated twice, and the table shows the second run: the first
call to a model includes Julia's compilation, which would otherwise dominate
the time.

```julia
using Printf

@printf("%-22s %8s %6s %8s\n", "model", "sMAPE %", "MASE", "seconds")
@printf("%-22s %8.2f %6.3f %8s\n", "seasonal naive", scores(snaive)..., "-")
@printf("%-22s %8.2f %6.3f %8s\n", "Naive2", scores(naive2s)..., "-")
for model in models
    evaluate(model)  # the first call compiles; time the second
    @printf("%-22s %8.2f %6.3f %8.1f\n", nameof(typeof(model)), evaluate(model)...)
end
```

```text
model                   sMAPE %   MASE  seconds
seasonal naive            17.26  1.146        -
Naive2                    16.93  1.046        -
EvoTreeRegressor          13.67  0.854      0.6
RandomForestRegressor     13.66  0.855     11.6
LinearRegressor           14.58  0.991      0.2
XGBoostRegressor          13.78  0.858      0.5
LGBMRegressor             13.71  0.856      0.5
KNNRegressor              14.32  0.898      2.4
```

All six learners beat both baselines on both metrics: their sMAPE is 2.3 to
3.3 points below Naive2's, and every MASE is below 1, while both baselines'
are above it. The four tree ensembles score 13.66 to 13.78, the
nearest-neighbour model 14.32 and the linear model 14.58. The random forest
took the longest to fit and forecast. Differences of a few hundredths between
learners mean little here: step 1 of the transform cancels in step 3, so
leaving it out changes ``z`` only by rounding, yet in a separate run it moved
the sMAPE of EvoTrees, the random forest and LightGBM from 13.67, 13.66 and
13.71 to 13.64, 13.62 and 13.74.

For calibration, the table below lists every method with a monthly result in
Table 14 of Oreshkin, B. N., Carpov, D., Chapados, N. and Bengio, Y. (2020),
"N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting", *International Conference on Learning Representations (ICLR
2020)*. Those numbers are on all 1428 series and this page uses 1399, so the
comparison is approximate. Naive2 scores 16.93 here against the published
16.91, so the scoring and the subset are close to the published setting.

| Method | sMAPE % |
|---|---|
| Naive2 | 16.91 |
| ARIMA (B-J automatic) | 14.81 |
| Comb S-H-D | 14.48 |
| ForecastPro | 13.86 |
| Theta | 13.85 |
| DOTM | 13.74 |
| BaggedETS.BC | 13.74 |
| EXP | 13.43 |
| N-BEATS-G | 13.19 |
| N-BEATS-I | 13.15 |
| N-BEATS-I+G | 13.11 |

The four tree ensembles land level with Theta, the winner of M3, ForecastPro,
DOTM and bagged ETS. EXP, a later statistical method, is 0.2 to 0.35 ahead
of them, and N-BEATS about 0.5 to 0.7. That does not show that the trees beat
Theta: the margin over Theta is at most 0.19, there is no significance test,
and the series differ. The nearest-neighbour and linear models are close to
Comb S-H-D and ahead of ARIMA.

[Tuning the models](tuning.md) tunes each learner's hyperparameters on a
validation window and scores the chosen configurations on this holdout.

**Tested with** Julia 1.12.5 and 1.10.12, each started with `--threads=auto`
(20 threads), which gave the same scores; the times above are from Julia
1.12.5 and depend on the machine and the thread count. Run alone, download
included, the whole script took about a minute on both versions and peaked at
about 2.8 GiB of memory on Julia 1.12.5 and 5 GiB on 1.10.12. Package
versions: MachineLearningForecast 0.1.0, EvoTrees 0.19.0,
MLJDecisionTreeInterface 0.5.0, MLJLinearModels 0.10.4, MLJXGBoostInterface
0.3.13, LightGBM 2.2.2, NearestNeighborModels 0.2.3 and ZipArchives 2.6.1.

## Next steps

These are default hyperparameters, `base_score` aside, on one feature set. To
tune the hyperparameters of these six learners on a validation window, continue
with [Tuning the models](tuning.md). To search over models, hyperparameters,
feature sets and strategies, scored on backtest error, see [`tune`](@ref).
