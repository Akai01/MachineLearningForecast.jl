# Using different models

Any MLJ `Deterministic` regressor can be the base model of a
[`Forecaster`](@ref). To change the learner you swap the model and nothing
else: the features, the strategy and the data stay as they are. This page fits
six learners from six packages to the same panel of 1399 monthly series from
the M3 competition, each as one global model with the same specification, and
scores them on the same holdout.

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

## The panel and the holdout

The evaluation follows the M3 protocol: the last 18 months of each series are
held out, and the model is trained on everything before them. The training
rows of all series go into one long-format table with columns `unique_id`,
`ds` and `y`.

The series' training means range from about 1,300 to 18,800, and one global
model fits all of them with one loss. So each series is divided by its
training mean before fitting, and its forecasts are multiplied back; the mean
uses training months only, so nothing from the holdout leaks in.

```julia
using Statistics

h = 18
train(s) = s.y[1:(end - h)]
test(s) = s.y[(end - h + 1):end]
level = Dict(s.id => mean(train(s)) for s in dated)

panel = (unique_id = reduce(vcat, [fill(s.id, length(train(s))) for s in dated]),
         ds = reduce(vcat, [s.start .+ Month.(0:(length(s.y) - h - 1)) for s in dated]),
         y = reduce(vcat, [train(s) ./ level[s.id] for s in dated]))
println(length(panel.y), " training rows")
```

```text
140321 training rows
```

## One specification for every learner

Every learner gets the same [`FeatureSet`](@ref) (lags 1 to 3 and 12, a
12-month rolling mean and the calendar month) and the same
[`Recursive`](@ref) strategy, with `id = :unique_id` to fit one model across
the panel. The month comes from [`Calendar`](@ref) and not from
[`Fourier`](@ref): `Fourier` counts steps from each series' first timestamp,
and these series start in different months (1155 in January, 198 in October,
46 in other months), so the same `Fourier` value would mean a different month
in different series.

```julia
using MachineLearningForecast

features = FeatureSet(Lag(1), Lag(2), Lag(3), Lag(12), RollingMean(12),
                      Calendar(:month))
spec(model) = Forecaster(model; features, strategy = Recursive(), freq = Month(1),
                         id = :unique_id)
```

## Scoring

`evaluate` fits one model on the panel, forecasts 18 months for every series
(each from its own last training month), undoes the scaling, and scores each
series with [`smape`](@ref) and [`mase`](@ref) on the original scale. `mase`
uses seasonality `m = 12` and that series' training values. Both are averaged
over the 1399 series, and sMAPE is shown in percent. MLJLinearModels also
exports a `fit`, so the code calls `MachineLearningForecast.fit`.

The seasonal-naive baseline repeats each series' last 12 training months.

```julia
function scores(forecasts)
    smapes = [smape(test(s), forecasts[s.id]) for s in dated]
    mases = [mase(test(s), forecasts[s.id]; y_train = train(s), m = 12) for s in dated]
    return 100 * mean(smapes), mean(mases)
end

function evaluate(model)
    t0 = time()
    fc = forecast(MachineLearningForecast.fit(spec(model), panel), h)
    seconds = time() - t0
    forecasts = Dict(s.id => fc.y_hat[fc.unique_id .== s.id] .* level[s.id]
                     for s in dated)
    return (scores(forecasts)..., seconds)
end

snaive = Dict(s.id => repeat(train(s)[(end - 11):end], 2)[1:h] for s in dated)
```

This holdout is built by hand and not with [`backtest`](@ref) because, on a
panel, `backtest` cuts its folds on one global timestamp grid. The M3 series
span different calendar years (they end in 73 different months, from 1864 to
2005), so a cut on a shared date would not hold out each series' own last 18
months. When your series share a calendar, `backtest` does this for you, and
[`tune`](@ref) builds on it.

## The models

Each model is used with its package's default hyperparameters. Where a model
draws random numbers, its seed is fixed, so the results reproduce.

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

```julia
using EvoTrees, MLJDecisionTreeInterface, MLJLinearModels, MLJXGBoostInterface
using LightGBM, NearestNeighborModels

models = [
    EvoTreeRegressor(seed = 1),
    RandomForestRegressor(rng = 1),
    LinearRegressor(),
    XGBoostRegressor(seed = 1),
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
for model in models
    evaluate(model)  # the first call compiles; time the second
    @printf("%-22s %8.2f %6.3f %8.1f\n", nameof(typeof(model)), evaluate(model)...)
end
```

```text
model                   sMAPE %   MASE  seconds
seasonal naive            17.26  1.146        -
EvoTreeRegressor          16.70  1.133      1.3
RandomForestRegressor     16.63  1.076      7.4
LinearRegressor           17.34  1.224      1.1
XGBoostRegressor          16.76  1.113      1.0
LGBMRegressor             16.66  1.125      1.2
KNNRegressor              16.95  1.105      1.0
```

In this run the five nonlinear learners beat the seasonal-naive baseline on
both metrics, by small margins: their mean sMAPE is 16.63 to 16.95 against the
baseline's 17.26, and their mean MASE 1.076 to 1.133 against 1.146. The linear
model is worse than the baseline on both, at 17.34 and 1.224. The order
depends on the metric: `RandomForestRegressor` has the lowest value of both,
while `KNNRegressor` is fifth by sMAPE and second by MASE. Every MASE is above
1, the baseline's included: on average, the holdout errors are larger than the
in-sample seasonal-naive errors that MASE divides by. The random forest took
the longest to fit and forecast. These numbers come from one holdout, one
feature set and default hyperparameters, so they describe this setup and not
the learners in general.

**Tested with** Julia 1.12.5 and 1.10.12, each started with `--threads=auto`
(20 threads), which gave the same scores; the times above are from Julia
1.12.5 and depend on the machine and the thread count. The whole script
peaked at about 3 GB of memory on Julia 1.12.5 and 6 GB on 1.10.12. Package
versions: MachineLearningForecast 0.1.0, EvoTrees 0.19.0,
MLJDecisionTreeInterface 0.5.0, MLJLinearModels 0.10.4, MLJXGBoostInterface
0.3.13, LightGBM 2.2.2, NearestNeighborModels 0.2.3 and ZipArchives 2.6.1.

## Next steps

These are default hyperparameters on one feature set. To search over models,
hyperparameters, feature sets and strategies, scored on backtest error, see
[`tune`](@ref).
