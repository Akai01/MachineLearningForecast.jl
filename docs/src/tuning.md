# Tuning the models

[Using different models](models.md) fits six learners with their packages'
default hyperparameters, XGBoost's `base_score` aside. This page tunes the
hyperparameters of each learner on a validation window, then scores the chosen
configuration once on the same M3 holdout.

This page continues the session of [Using different models](models.md): run
that page's code first. The code below reuses its data, `h`, `long`, `spec`
with its features and transform, `evaluate` and the list of `models`, and adds
the standard library Random. It runs top to bottom after that page's code, and
the outputs shown are from that run.

## The validation window

The tuning must not see the holdout, so it scores on the window before it. For
each series, the validation history is `y[1:end-36]`, the training history
without its last 18 months, and the validation target is the 18 months that
follow it. `vpanel` is the panel with the holdout cut off each series, so
`evaluate` on it, with `backtest` and `windows = 1`, scores exactly this
window, and the transform is fitted on the validation history only, as it was
fitted on the training history before. 291 of the 1399 validation histories
are shorter than 36 months (the shortest has 30), too short for the
seasonality test, so `Deseasonalize` treats them as non-seasonal. `validate`
returns the sMAPE and MASE.

```julia
vdated = [(id = s.id, start = s.start, y = s.y[1:(end - h)]) for s in dated]
vpanel = long(vdated)
println(length(vpanel.y), " rows")

validate(model) = evaluate(model, vpanel)[1:2]
```

```text
140321 rows
```

## The search spaces

Every learner gets the same budget of 30 configurations, and configuration 1 is
its model from the previous page. For the four
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
    xgboost => c -> XGBoostRegressor(; seed = 1, base_score = 0.0, c...),
    # LightGBM bags only when bagging_freq is above 0.
    lightgbm => c -> LightGBM.MLJInterface.LGBMRegressor(; seed = 1, verbosity = -1,
                         bagging_freq = c.bagging_fraction < 1 ? 1 : 0, c...),
    knn => c -> KNNRegressor(; c...),
]
```

Every XGBoost candidate gets `base_score = 0.0`, as the default model did on
the previous page, where [The models](@ref) gives the reason: XGBoost starts
from `base_score`, 0.5 by default, while ``z`` is centred near 0. The offset
hurts most where the trees cannot remove it, with a small learning rate and few
rounds. The code below scores two such candidates on validation with each
`base_score`.

```julia
for base_score in (0.5, 0.0), num_round in (100, 200)
    model = XGBoostRegressor(; seed = 1, eta = 0.02, num_round, base_score)
    @printf("base_score = %.1f, %d rounds: %.2f\n", base_score, num_round,
            validate(model)[1])
end
```

```text
base_score = 0.5, 100 rounds: 32.25
base_score = 0.5, 200 rounds: 15.46
base_score = 0.0, 100 rounds: 14.15
base_score = 0.0, 200 rounds: 14.06
```

At learning rate 0.02, 100 rounds with the default `base_score` score 32.25
sMAPE on validation against 14.15 with `base_score = 0.0`, and 200 rounds
15.46 against 14.06. Left at 0.5, the search would judge such candidates
mostly by the part of the offset they had not yet removed.

## Tuning on the validation window

Each of the 180 configurations is fitted on the validation panel and scored on
the validation target. Each learner keeps the configuration with the lowest
validation sMAPE; on a tie, the first. `tried` keeps every learner's
candidates and validation scores for the optional diagnostic at the end of the
page. This step takes most of the runtime without that diagnostic, about 13 of
the 14.7 minutes on 20 threads (see the end of the page). For 12 of the 14
lasso fits, those with ``\lambda \le 10^{-2.5}``, MLJLinearModels warns that
proximal gradient descent did not converge in 1000 iterations, and it warns
again when the chosen lasso is fitted for the holdout below. The outputs leave
those warnings out, and the fits are scored as they are.

```julia
chosen, tried = [], []
println(" "^23, "sMAPE % on validation   MASE on validation")
@printf("%-22s %10s %10s %10s %10s %7s\n", "model", "default", "best",
        "default", "best", "config")
for (model, (configs, make)) in zip(models, searches)
    candidates = [model; make.(configs[2:end])]
    results = validate.(candidates)
    i = argmin(first.(results))
    push!(chosen, candidates[i])
    push!(tried, (candidates, results))
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
XGBoostRegressor           14.455     14.112     0.9197     0.8884       2
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
XGBoostRegressor           13.776     13.647     0.8575     0.8538
LGBMRegressor              13.706     13.729     0.8562     0.8626
KNNRegressor               14.317     13.718     0.8978     0.8442
```

## What tuning changed

In this run, tuning moved holdout sMAPE little for four learners: by -0.03 for
`EvoTreeRegressor`, +0.04 for `RandomForestRegressor`, +0.02 for
`LGBMRegressor` and +0.00 for the linear learner, and the MASE of all four got
slightly worse. It helped the other two. `KNNRegressor` gained 0.60 sMAPE and
its MASE fell from 0.898 to 0.844: its default of 5 neighbours is too small
here, and the search picked 75. `XGBoostRegressor` gained 0.13 and its MASE
fell from 0.858 to 0.854, with 800 rounds at learning rate 0.02 in place of
the default 100 at 0.3. After tuning, five of the six learners score 13.65 to
13.73, while the linear learner stays near 14.58.

One 18-month validation window is a noisy guide. On validation every learner
can only improve, since its default is one of the candidates, but the gain
need not carry over to the holdout. The random forest and XGBoost both gained
0.34 sMAPE on validation; on the holdout XGBoost kept 0.13 of it and the random
forest lost 0.04. EvoTrees and LightGBM gained 0.08 and 0.05 on validation,
and changed their holdout sMAPE by -0.03 and +0.02. KNN's gain of 0.66 on
validation carried over almost whole, as 0.60. The two tables are shown in full
so that you can compare them yourself, and [How much tuning can gain
here](@ref) looks closer.

Which configurations are drawn matters as well. On Julia 1.10 the same code
draws other configurations for the four tree ensembles (see the end of the
page). There the tuned holdout sMAPE was 13.645 for EvoTrees, 13.573 for the
random forest, 13.664 for XGBoost and 13.587 for LightGBM, against 13.646,
13.707, 13.647 and 13.729 above: the random forest and LightGBM gained about
0.1 instead of losing a little, and XGBoost gained 0.11 instead of 0.13. For
the tree ensembles, then, the change from tuning, at most 0.13 here, is of the
same order as the change from drawing a different 29 configurations, up to
0.14. The linear and nearest-neighbour searches try their whole space and
chose the same configurations on both versions.

The search loop is written out so that the page can show both sMAPE and MASE
for every candidate. [`tune`](@ref) runs the same search for one metric: given
`vpanel`, `windows = 1` and `horizon = 18`, it scores each candidate on each
series' own 18 months before the holdout, and with `windows` above 1 by its
mean over several such windows. Pass the candidate models in `grid` with
[`RandomSearch`](@ref) or [`GridSearch`](@ref), or write a
[`TuningStrategy`](@ref) for a search of your own.

The choice is made on sMAPE, the measure the M3 competition reported, so the
note "Why sMAPE" under [Scoring](@ref) on the previous page applies here too;
MASE is shown alongside for that reason.

## How much tuning can gain here

The tables above leave two questions open for the four tree ensembles, whose
tuning moved holdout sMAPE by 0.13 at most: were there better configurations
among the 30 to find, and does the validation window rank the configurations
as the holdout does?
The optional code below answers both by scoring every configuration on the
holdout. It reports the holdout sMAPE of the default, of the configuration
picked on validation (`tuned`, as in the table above) and of the configuration
that is best on the holdout (`best`).

It then measures how well a validation score ranks the configurations: the
Spearman correlation between that score and the holdout sMAPE across the 30
configurations, 1 when the two put them in the same order and near 0 when the
orders are unrelated. It is computed with `cor` from Statistics on ranks, and
tied scores share their average rank. Four validation scores are compared:
sMAPE on the validation window above (`window 1`); sMAPE on a second window,
the 18 months before window 1 (`window 2`); the sMAPE of both windows pooled,
with every series-window pair weighted equally (`both`); and MASE on window 1
(`w1 MASE`). Window 2 is scored as window 1 is, by `evaluate` on the panel
with the last 36 months cut off each series, and only for the series with at
least 30 months of history before it, the length of the shortest history in
window 1.

!!! warning "Optional, and for diagnosis only"
    This step is not part of the tuning: it peeks at the holdout, which a
    real forecast cannot do, and nothing on this page is chosen from it. It
    refits 240 models, 120 on the holdout and 120 on window 2, which took 22
    minutes here, 17 of them for the random forest.

```julia
w2 = [(id = s.id, start = s.start, y = s.y[1:(end - h)]) for s in vdated
      if length(s.y) - 2h >= 30]
w2panel = long(w2)
println(length(w2), " of ", length(vdated), " series have window 2")

validate2(model) = evaluate(model, w2panel)[1]

# Average ranks for ties, as standard Spearman uses.
ranks(x) = [count(<(v), x) + (count(==(v), x) + 1) / 2 for v in x]
spearman(a, b) = cor(ranks(a), ranks(b))
n1, n2 = length(vdated), length(w2)

println(" "^23, "sMAPE % on the holdout           Spearman with holdout sMAPE")
@printf("%-22s %8s %8s %8s %9s %9s %9s %9s\n", "model", "default", "tuned", "best",
        "window 1", "window 2", "both", "w1 MASE")
for i in (1, 2, 4, 5)
    candidates, results = tried[i]
    holdout = [evaluate(c)[1] for c in candidates]
    v1, v2 = first.(results), validate2.(candidates)
    # Weight by series so every series-window pair counts once.
    both = (n1 .* v1 .+ n2 .* v2) ./ (n1 + n2)
    rho = [spearman(v, holdout) for v in (v1, v2, both, last.(results))]
    @printf("%-22s %8.3f %8.3f %8.3f %9.2f %9.2f %9.2f %9.2f\n", nameof(typeof(models[i])),
            holdout[1], holdout[argmin(v1)], minimum(holdout), rho...)
end
```

```text
1081 of 1399 series have window 2
                       sMAPE % on the holdout           Spearman with holdout sMAPE
model                   default    tuned     best  window 1  window 2      both   w1 MASE
EvoTreeRegressor         13.674   13.646   13.572      0.70      0.32      0.62      0.49
RandomForestRegressor    13.663   13.707   13.488     -0.20      0.39     -0.14     -0.34
XGBoostRegressor         13.776   13.647   13.565      0.55      0.46      0.53      0.30
LGBMRegressor            13.706   13.729   13.531      0.34      0.33      0.35      0.12
```

Even with hindsight the gain on offer is small: the configuration that is best
on the holdout beats the default by 0.10 sMAPE for EvoTrees, 0.18 for the
random forest, 0.21 for XGBoost and 0.18 for LightGBM. The configurations
picked on validation took 0.03 of that for EvoTrees and 0.13 for XGBoost, and
lost a little for the other two.

Here validation ranks the configurations unreliably. On window 1 the Spearman
correlation with the holdout is 0.70 for EvoTrees, 0.55 for XGBoost, 0.34 for
LightGBM and -0.20 for the random forest, whose validation order is unrelated
to its holdout order. A second window does not fix this. Window 2 alone gives
0.32 to 0.46: better than window 1 for the random forest, worse for EvoTrees
and XGBoost, level for LightGBM. Pooling both windows gives 0.62, -0.14, 0.53
and 0.35 for EvoTrees, the random forest, XGBoost and LightGBM, close to
window 1 alone. Ranking by MASE on window 1 agrees less with the holdout than
ranking by sMAPE, for all four learners.

On Julia 1.10, whose searches draw other configurations (see the end of the
page), the same code gives other numbers. For EvoTrees, the random forest,
XGBoost and LightGBM, the best configuration in hindsight beats the default by
0.11, 0.16, 0.20 and 0.12, again small. The Spearman correlations are 0.56,
0.09, 0.74 and 0.72 on window 1, 0.24, 0.42, 0.71 and 0.57 on window 2, and
0.43, 0.19, 0.74 and 0.66 pooled, and ranking by MASE again agrees less for
all but the random forest. There validation ranks the XGBoost and LightGBM
configurations well, and LightGBM's pick is its best, so how well one window
ranks the configurations depends on which ones are drawn as well as on the
learner.

The gains at stake, about 0.1 to 0.2 sMAPE, are no larger than how much a
configuration's lead over the default changes from one 18-month window to the
next: the random forest's pick led by 0.34 on validation and trailed by 0.04 on
the holdout. On this panel, with these search spaces, the defaults of the tree
ensembles are within about 0.2 sMAPE of the best of their 30 configurations,
found in hindsight, on both versions, and the pick on window 1 was that
configuration in one of the eight searches, LightGBM's on Julia 1.10. These
are findings for this panel and these spaces, not general ones.

**Tested with** Julia 1.12.5 and 1.10.12, each started with `--threads=auto`
(20 threads), running the previous page's code and this page's as one script.
The outputs above are from Julia 1.12.5, where the script took 36 minutes, 14.7
of them before the optional diagnostic, and peaked at about 5.5 GiB of memory.
On 1.10.12 it took 29 minutes, 11.7 before the diagnostic, and peaked at about
17.4 GiB. Julia 1.11 changed how `MersenneTwister` turns an integer seed into
its state, so on Julia 1.10 `MersenneTwister(2026)` gives other random numbers
and the four random searches draw other configurations. The defaults and the
linear and nearest-neighbour searches score the same on both versions. Package
versions are those listed on the previous page.
