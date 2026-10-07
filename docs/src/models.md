# Using different models

Any MLJ `Deterministic` regressor can be the base model of a
[`Forecaster`](@ref). To change the learner you swap the model and nothing
else: the features, the strategy and the data stay as they are. This page fits
six learners from six packages to the same panel of 1399 monthly series from
the M3 competition, each as one global model with the same specification and
its package's default hyperparameters. It scores them on the M3 holdout next to
two baselines and published results.

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

For each series, from its training history only:

1. **Level.** Divide the series by ``m``, the mean of its last 12 values, so
   every series is near 1 at its forecast origin.
2. **Seasonal adjustment.** The series is seasonal if its autocorrelation at
   lag 12, ``r_{12}``, passes the 90% test that the M4 competition's
   benchmarks use: ``|r_{12}| > 1.645 \sqrt{(1 + 2 \sum_{k=1}^{11} r_k^2) / n}``
   on ``n \ge 36`` points (every M3 training history has at least 48). A
   seasonal series gets the classical multiplicative indices: each value is
   divided by its centred 2×12 moving average, these ratios are averaged by
   position in the year, and the 12 averages are normalised to mean 1. A
   non-seasonal series gets indices of 1. Dividing by the indices gives the
   seasonally adjusted series ``x``.
3. **Log ratio.** ``z_t = \log(x_t / \bar{x}_t)``, where ``\bar{x}_t`` is the
   mean of the 12 values before ``x_t``: each month relative to the year
   before it. A ratio has no unit, so ``z`` does not depend on the level of
   step 1 and is on the same scale in every series. It is 12 values shorter
   than the history.
4. **Inverse.** Forecasts of ``z`` are turned back one month at a time:
   ``\exp(\hat{z})`` times the mean of the previous 12 values of ``x``, rebuilt
   ones included, then times that month's seasonal index and ``m``.

Positions in the year are counted from each series' first month, so the
forecast for month ``n + j`` of a history of length ``n`` gets the index at
position `mod1(n + j, 12)`. The log needs positive values, and the code
assumes them: every M3 training history is positive. A series with zeros or
negative values needs a shift before the log.

This setup was chosen on a validation window, the 18 months before the
holdout, among several target transforms and feature sets, and was then run
once on the holdout.

```julia
acf(x, k) = (d = x .- mean(x); sum(d[(k + 1):end] .* d[1:(end - k)]) / sum(abs2, d))

function isseasonal(x)
    n = length(x)
    n < 36 && return false
    return abs(acf(x, 12)) > 1.645 * sqrt((1 + 2 * sum(acf(x, k)^2 for k in 1:11)) / n)
end

function seasonal_index(x)
    isseasonal(x) || return ones(12)
    t = 7:(length(x) - 6)
    # Centred 2x12 moving average: half weight at both ends.
    ma = [(x[i - 6] / 2 + sum(x[(i - 5):(i + 5)]) + x[i + 6] / 2) / 12 for i in t]
    ratio = x[t] ./ ma
    s = [mean(ratio[mod1.(t, 12) .== p]) for p in 1:12]
    return s ./ mean(s)
end

function transform(y)
    m = mean(y[(end - 11):end])
    index = seasonal_index(y ./ m)
    x = y ./ m ./ index[mod1.(eachindex(y), 12)]
    z = [log(x[t] / mean(x[(t - 12):(t - 1)])) for t in 13:length(x)]
    return (; z, m, index, n = length(x), last12 = x[(end - 11):end])
end

function untransform(zhat, state)
    x = copy(state.last12)
    for z in zhat
        push!(x, exp(z) * mean(x[(end - 11):end]))
    end
    return state.m .* x[13:end] .* state.index[mod1.(state.n .+ eachindex(zhat), 12)]
end
```

## The panel

`tr` holds each series' transform: its ``z`` and what `untransform` needs.
The ``z`` of all series go into one long-format table with columns
`unique_id`, `ds` and `y`. Since ``z`` starts at the 13th month of each
history, so does `ds`.

```julia
tr = Dict(s.id => transform(train(s)) for s in dated)

panel = (unique_id = reduce(vcat, [fill(s.id, length(tr[s.id].z)) for s in dated]),
         ds = reduce(vcat, [s.start .+ Month.(12:(length(train(s)) - 1)) for s in dated]),
         y = reduce(vcat, [tr[s.id].z for s in dated]))
println(length(panel.y), " training rows")
```

```text
123533 training rows
```

## One specification for every learner

Every learner gets the same [`FeatureSet`](@ref) (lags 1 to 12 of ``z``, its
12-month rolling mean and the calendar month) and the same
[`Recursive`](@ref) strategy, with `id = :unique_id` to fit one model across
the panel. The month comes from [`Calendar`](@ref) and not from
[`Fourier`](@ref): `Fourier` counts steps from each series' first timestamp,
and these series start in different months (1155 in January, 198 in October,
46 in other months), so the same `Fourier` value would mean a different month
in different series.

```julia
using MachineLearningForecast

features = FeatureSet(Lag.(1:12)..., RollingMean(12), Calendar(:month))
spec(model) = Forecaster(model; features, strategy = Recursive(), freq = Month(1),
                         id = :unique_id)
```

## Scoring

`evaluate` fits one model on the panel, forecasts 18 months of ``z`` for every
series (each from its own last training month), turns them back into values
with `untransform`, and scores each series with [`smape`](@ref) and
[`mase`](@ref) on the original scale. `mase` uses seasonality `m = 12` and that
series' training values. Both are averaged over the 1399 series, and sMAPE is
shown in percent. MLJLinearModels also exports a `fit`, so the code calls
`MachineLearningForecast.fit`.

There are two baselines. The seasonal naive forecast repeats each series' last
12 training months. Naive2, the benchmark of the M3 competition, repeats the
last seasonally adjusted value and puts the season back, with the seasonal
indices of the transform.

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
    forecasts = Dict(s.id => untransform(fc.y_hat[fc.unique_id .== s.id], tr[s.id])
                     for s in dated)
    return (scores(forecasts)..., seconds)
end

function naive2(y, index)
    n = length(y)
    return y[end] / index[mod1(n, 12)] .* index[mod1.(n .+ (1:h), 12)]
end

snaive = Dict(s.id => repeat(train(s)[(end - 11):end], 2)[1:h] for s in dated)
naive2s = Dict(s.id => naive2(train(s), tr[s.id].index) for s in dated)
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
EvoTreeRegressor          13.67  0.854      1.4
RandomForestRegressor     13.66  0.855     11.8
LinearRegressor           14.58  0.991      1.4
XGBoostRegressor          13.85  0.860      1.7
LGBMRegressor             13.71  0.856      1.8
KNNRegressor              14.32  0.898      3.5
```

All six learners beat both baselines on both metrics: their sMAPE is 2.3 to
3.3 points below Naive2's, and every MASE is below 1, while both baselines'
are above it. The four tree ensembles score 13.66 to 13.85, the
nearest-neighbour model 14.32 and the linear model 14.58. The random forest
took the longest to fit and forecast. Differences of a few hundredths between
learners mean little here: step 1 of the transform cancels in step 3, so
leaving it out changes ``z`` only by rounding, yet it moves the sMAPE of
EvoTrees, the random forest and LightGBM by 0.03 to 0.05.

For calibration, the table below lists published sMAPE on the M3 monthly
series from Oreshkin, B. N., Carpov, D., Chapados, N. and Bengio, Y. (2020),
"N-BEATS: Neural basis expansion analysis for interpretable time series
forecasting", *International Conference on Learning Representations (ICLR
2020)*, Table 14. Those numbers are on all 1428 series and this page uses
1399, so the comparison is approximate. Naive2 scores 16.93 here against the
published 16.91, so the scoring and the subset are close to the published
setting.

| Method | sMAPE % |
|---|---|
| Naive2 | 16.91 |
| ARIMA (B-J automatic) | 14.81 |
| Comb S-H-D | 14.48 |
| ForecastPro | 13.86 |
| Theta | 13.85 |
| N-BEATS-G | 13.19 |
| N-BEATS-I | 13.15 |

The four tree ensembles land level with Theta, the winner of M3, and
ForecastPro, and about 0.5 to 0.7 behind N-BEATS. That does not show that they
beat Theta: the margin over Theta is at most 0.19, there is no significance
test, and the series differ. The nearest-neighbour and linear models are close
to Comb S-H-D and ahead of ARIMA.

Tuning added little. In a separate experiment, each learner was given 30
hyperparameter configurations under this setup, its defaults among them, and
the one with the lowest sMAPE on the validation window was scored on the
holdout. Against the defaults, that changed holdout sMAPE by -0.60 for
`KNNRegressor`, -0.22 for `XGBoostRegressor`, -0.03 for `EvoTreeRegressor`,
0.00 for `LinearRegressor`, +0.02 for `LGBMRegressor` and +0.04 for
`RandomForestRegressor`. These numbers come from one holdout, one transform
and one feature set, so they describe this setup and not the learners in
general.

**Tested with** Julia 1.12.5 and 1.10.12, each started with `--threads=auto`
(20 threads), which gave the same scores; the times above are from Julia
1.12.5 and depend on the machine and the thread count. The whole script
peaked at about 3 GB of memory on Julia 1.12.5 and 5 GB on 1.10.12. Package
versions: MachineLearningForecast 0.1.0, EvoTrees 0.19.0,
MLJDecisionTreeInterface 0.5.0, MLJLinearModels 0.10.4, MLJXGBoostInterface
0.3.13, LightGBM 2.2.2, NearestNeighborModels 0.2.3 and ZipArchives 2.6.1.

## Next steps

These are default hyperparameters on one feature set. To search over models,
hyperparameters, feature sets and strategies, scored on backtest error, see
[`tune`](@ref).
