# Tuning the models

[Using different models](models.md) fits six learners with their packages'
default hyperparameters. This page tunes the hyperparameters of each learner on
a validation window, then scores the chosen configuration once on the same M3
holdout.

This page continues the session of [Using different models](models.md): run
that page's code first. The code below reuses its data, `train` and `test`,
`transform` and `untransform`, `spec` with its features, `evaluate` and the
list of `models`, and adds the standard library Random. It runs top to bottom
after that page's code, and the outputs shown are from that run.

## The validation window

The tuning must not see the holdout, so it scores on the window before it. For
each series, the validation history is `y[1:end-36]`, the training history
without its last 18 months, and the validation target is the 18 months that
follow it. Cutting the holdout off each series turns `train` and `test` into
exactly these. The transform is fitted on the validation history only, as it
was fitted on the training history before. 291 of the 1399 validation
histories are shorter than 36 months (the shortest has 30), too short for the
seasonality test, so the transform treats them as non-seasonal. `validate`
scores a model as `evaluate` does, on the validation target.

```julia
vdated = [(id = s.id, start = s.start, y = s.y[1:(end - h)]) for s in dated]
vtr = Dict(s.id => transform(train(s)) for s in vdated)

vpanel = (unique_id = reduce(vcat, [fill(s.id, length(vtr[s.id].z)) for s in vdated]),
          ds = reduce(vcat, [s.start .+ Month.(12:(length(train(s)) - 1)) for s in vdated]),
          y = reduce(vcat, [vtr[s.id].z for s in vdated]))
println(length(vpanel.y), " training rows")

function validate(model)
    fc = forecast(MachineLearningForecast.fit(spec(model), vpanel), h)
    forecasts = Dict(s.id => untransform(fc.y_hat[fc.unique_id .== s.id], vtr[s.id])
                     for s in vdated)
    smapes = [smape(test(s), forecasts[s.id]) for s in vdated]
    mases = [mase(test(s), forecasts[s.id]; y_train = train(s), m = 12) for s in vdated]
    return 100 * mean(smapes), mean(mases)
end
```

```text
98351 training rows
```

## The search spaces

Every learner gets the same budget of 30 configurations, and configuration 1 is
its model from the previous page, with its package's defaults. For the four
tree ensembles, `random_search` draws the other 29 from the space below: each
hyperparameter uniformly from its list, from a random number generator with a
fixed seed, and a draw that repeats an earlier configuration is drawn again.
The linear and nearest-neighbour spaces have exactly 30 points, so all of them
are tried, in order. The linear one is least squares, then ridge with 15
penalties from ``10^{-8}`` to ``10^{-1}`` and lasso with 14 from ``10^{-8}``
to ``10^{-1.5}``, in half-decade steps. The nearest-neighbour one is ten values
of `K` under three neighbour weightings. Each search pairs its configurations
with a function that builds a model from one.

```julia
using Random

# A repeat would spend budget on a known configuration.
function random_search(model, space)
    rng = MersenneTwister(2026)
    configs = [NamedTuple{keys(space)}(getproperty.(Ref(model), keys(space)))]
    while length(configs) < 30
        c = map(v -> rand(rng, v), space)
        c in configs || push!(configs, c)
    end
    return configs
end

evotrees = random_search(models[1],
    (nrounds = [100, 200, 400, 800], eta = [0.02, 0.05, 0.1, 0.2],
     max_depth = [3, 4, 5, 6, 7, 8], L2 = [0.0, 1.0, 10.0], lambda = [0.0, 0.1, 1.0],
     min_weight = [1.0, 10.0, 50.0], rowsample = [0.5, 0.8, 1.0],
     colsample = [0.5, 0.8, 1.0]))
forest = random_search(models[2],
    (n_trees = [100, 300, 500], max_depth = [-1, 8, 12, 16],
     min_samples_leaf = [1, 5, 20, 50], n_subfeatures = [-1, 0, 3, 4],
     sampling_fraction = [0.5, 0.7, 1.0]))
linear = [(model = LinearRegressor, lambda = 0.0);
          [(model = RidgeRegressor, lambda = l) for l in 10.0 .^ (-8:0.5:-1)];
          [(model = LassoRegressor, lambda = l) for l in 10.0 .^ (-8:0.5:-1.5)]]
xgboost = random_search(models[4],
    (num_round = [100, 200, 400, 800], eta = [0.02, 0.05, 0.1, 0.3],
     max_depth = [3, 4, 6, 8], subsample = [0.5, 0.8, 1.0],
     colsample_bytree = [0.5, 0.8, 1.0], min_child_weight = [1.0, 10.0, 50.0]))
lightgbm = random_search(models[5],
    (num_iterations = [100, 200, 400, 800], learning_rate = [0.02, 0.05, 0.1, 0.2],
     num_leaves = [7, 15, 31, 63, 127], min_data_in_leaf = [5, 20, 50, 100, 200],
     feature_fraction = [0.6, 0.8, 1.0], bagging_fraction = [0.6, 0.8, 1.0]))
knn = [(K = k, weights = w) for w in (Uniform(), Inverse(), Dudani())
       for k in (5, 10, 20, 30, 50, 75, 100, 150, 200, 300)]

searches = [
    evotrees => c -> EvoTreeRegressor(; seed = 1, c...),
    forest => c -> RandomForestRegressor(; rng = 1, c...),
    linear => c -> c.model === LinearRegressor ? LinearRegressor() :
                   c.model(lambda = c.lambda),
    xgboost => c -> XGBoostRegressor(; seed = 1, c...),
    # LightGBM bags only when bagging_freq is above 0.
    lightgbm => c -> LightGBM.MLJInterface.LGBMRegressor(; seed = 1, verbosity = -1,
                         bagging_freq = c.bagging_fraction < 1 ? 1 : 0, c...),
    knn => c -> KNNRegressor(; c...),
]
```

## Tuning on the validation window

Each of the 180 configurations is fitted on the validation panel and scored on
the validation target. Each learner keeps the configuration with the lowest
validation sMAPE; on a tie, the first. This step takes most of the runtime,
about a quarter of an hour on 20 threads (see the end of the page). For 12 of
the 14 lasso fits, those with ``\lambda \le 10^{-2.5}``, MLJLinearModels
warns that proximal gradient descent did not converge in 1000 iterations, and
it warns again when the chosen lasso is fitted for the holdout below. The
outputs leave those warnings out, and the fits are scored as they are.

```julia
chosen = []
println(" "^23, "sMAPE % on validation   MASE on validation")
@printf("%-22s %10s %10s %10s %10s %7s\n", "model", "default", "best",
        "default", "best", "config")
for (model, (configs, make)) in zip(models, searches)
    candidates = [model; make.(configs[2:end])]
    results = validate.(candidates)
    i = argmin(first.(results))
    push!(chosen, candidates[i])
    @printf("%-22s %10.3f %10.3f %10.4f %10.4f %7d\n", nameof(typeof(model)),
            results[1][1], results[i][1], results[1][2], results[i][2], i)
end
```

```text
                       sMAPE % on validation   MASE on validation
model                     default       best    default       best  config
EvoTreeRegressor           14.169     14.092     0.8985     0.8852      17
RandomForestRegressor      14.416     14.073     0.9279     0.8873       7
LinearRegressor            14.513     14.512     0.9310     0.9307      23
XGBoostRegressor           14.518     14.140     0.9234     0.8902       2
LGBMRegressor              14.177     14.125     0.9034     0.8880      22
KNNRegressor               14.738     14.079     0.9646     0.8872      26
```

The chosen configurations are `evotrees[17]`, `forest[7]`, `linear[23]`,
`xgboost[2]`, `lightgbm[22]` and `knn[26]`:

- `EvoTreeRegressor`: `nrounds = 800`, `eta = 0.02`, `max_depth = 5`,
  `L2 = 1.0`, `lambda = 1.0`, `min_weight = 1.0`, `rowsample = 0.5`,
  `colsample = 0.8`.
- `RandomForestRegressor`: `n_trees = 100`, `max_depth = 8`,
  `min_samples_leaf = 20`, `n_subfeatures = -1`, `sampling_fraction = 1.0`.
- `LinearRegressor`: a `LassoRegressor` with `lambda = 1.0e-5`, level with
  least squares on validation (14.512 against 14.513).
- `XGBoostRegressor`: `num_round = 800`, `eta = 0.02`, `max_depth = 8`,
  `subsample = 0.8`, `colsample_bytree = 1.0`, `min_child_weight = 50.0`.
- `LGBMRegressor`: `num_iterations = 800`, `learning_rate = 0.02`,
  `num_leaves = 15`, `min_data_in_leaf = 50`, `feature_fraction = 1.0`,
  `bagging_fraction = 0.6`, so it bags.
- `KNNRegressor`: `K = 75` with `weights = Dudani()`; the default is `K = 5`
  with uniform weights.

## Scoring on the holdout

Only now is the holdout used. `evaluate`, from the previous page, fits each
learner's chosen configuration once on the full training histories, with the
transform fitted on them, and scores it on the held-out 18 months. The default
column repeats the previous page's results, to more decimals.

```julia
println(" "^23, "sMAPE % on the holdout   MASE on the holdout")
@printf("%-22s %10s %10s %10s %10s\n", "model", "default", "tuned", "default", "tuned")
for (model, tuned) in zip(models, chosen)
    d, t = evaluate(model), evaluate(tuned)
    @printf("%-22s %10.3f %10.3f %10.4f %10.4f\n", nameof(typeof(model)),
            d[1], t[1], d[2], t[2])
end
```

```text
                       sMAPE % on the holdout   MASE on the holdout
model                     default      tuned    default      tuned
EvoTreeRegressor           13.674     13.646     0.8539     0.8563
RandomForestRegressor      13.663     13.707     0.8553     0.8673
LinearRegressor            14.583     14.587     0.9909     0.9914
XGBoostRegressor           13.854     13.636     0.8597     0.8532
LGBMRegressor              13.706     13.729     0.8562     0.8626
KNNRegressor               14.317     13.718     0.8978     0.8442
```

## What tuning changed

In this run, tuning moved holdout sMAPE little for four learners: by -0.03 for
`EvoTreeRegressor`, +0.04 for `RandomForestRegressor`, +0.02 for
`LGBMRegressor` and +0.00 for the linear learner, and the MASE of all four got
slightly worse. It helped the other two. `KNNRegressor` gained 0.60 sMAPE and
its MASE fell from 0.898 to 0.844: its default of 5 neighbours is too small
here, and the search picked 75. `XGBoostRegressor` gained 0.22, with 800
rounds at learning rate 0.02 in place of the default 100 at 0.3. After tuning,
five of the six learners score 13.64 to 13.73, while the linear learner stays
near 14.58.

One 18-month validation window is a noisy guide. On validation every learner
can only improve, since its default is one of the candidates, but the gain
need not carry over to the holdout. The random forest gained 0.34 sMAPE on
validation, about as much as XGBoost's 0.38, and lost 0.04 on the holdout;
EvoTrees and LightGBM gained 0.08 and 0.05 on validation for -0.03 and +0.02.
The gains of XGBoost and KNN did carry over. The two tables are shown in full
so that you can compare them yourself.

Which configurations are drawn matters as well. On Julia 1.10 the same code
draws other configurations for the four tree ensembles (see the end of the
page). There the tuned holdout sMAPE was 13.645 for EvoTrees, 13.573 for the
random forest, 13.735 for XGBoost and 13.587 for LightGBM, against 13.646,
13.707, 13.636 and 13.729 above: the random forest and LightGBM gained about
0.1 instead of losing a little, and XGBoost gained 0.12 instead of 0.22. For
the tree ensembles, then, the change from tuning, at most 0.22 here, is of the
same order as the change from drawing a different 29 configurations, up to
0.14. The linear and nearest-neighbour searches try their whole space and
chose the same configurations on both versions.

The search is done by hand, and not with [`tune`](@ref), for the reason the
holdout is built by hand on the previous page: on a panel, [`backtest`](@ref)
cuts its folds on one global timestamp grid, and the M3 series span different
calendar years, so a cut on a shared date would not give each series its own
validation window. When your series share a calendar, or for a single series,
`tune` runs this search for you and scores each candidate by its mean over
several backtest folds instead of one window: pass the candidate models in
`grid` with [`RandomSearch`](@ref) or [`GridSearch`](@ref), or write a
[`TuningStrategy`](@ref) for a search of your own.

The choice is made on sMAPE, the measure the M3 competition reported, so the
note "Why sMAPE" under [Scoring](@ref) on the previous page applies here too;
MASE is shown alongside for that reason.

**Tested with** Julia 1.12.5 and 1.10.12, each started with `--threads=auto`
(20 threads), running the previous page's code and this page's as one script.
The outputs above are from Julia 1.12.5, where the script took 16.5 minutes
and peaked at about 4.5 GB of memory; on 1.10.12 it took 14 minutes and peaked
at about 11.5 GB. Julia 1.11 changed how `MersenneTwister` turns an integer
seed into its state, so on Julia 1.10 `MersenneTwister(2026)` gives other
random numbers and the four random searches draw other configurations. The
defaults and the linear and nearest-neighbour searches score the same on both
versions. Package versions are those listed on the previous page.
