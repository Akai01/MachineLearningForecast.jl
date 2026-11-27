# Probabilistic forecasts

This page shows how to get prediction intervals and predictive samples from
[`ConformalSeasonalPool`](@ref), how the method works, and how to score the result. It
marks our own choices as ours. The code blocks are plain Julia: they are not run when the
documentation is built. Their output was produced with Julia 1.12.5, and the random draws
can differ on other Julia versions (see [The draws and the random generator](@ref)).
Sources are listed under [References](conventions.md#References).

Today the package gives probabilistic forecasts in one shape: Conformal Seasonal Pools
(CSP; Manokhin 2026), around the seasonal naive forecast. It trains nothing. Intervals
around other models are not available yet.

## A worked example

A synthetic monthly series, eight years long, with a yearly cycle, a trend and some noise
(no randomness is involved in building it):

```julia
using MachineLearningForecast, Dates, Statistics

t = collect(Date(2012, 1, 1):Month(1):Date(2019, 12, 1))
y = [50 + 10 * sin(2π * i / 12) + 0.2i + sin(i^2) for i in 1:96]
df = (ds = t, y = y)
```

`ConformalSeasonalPool(12; freq=Month(1))` builds a [`Forecaster`](@ref) whose season is 12
steps long. `fit` trains nothing: it keeps each series' history. `forecast` with `levels`
adds the interval columns:

```julia
csp = ConformalSeasonalPool(12; freq = Month(1))
fitted = fit(csp, df)
fcast = forecast(fitted, 12; levels = (80, 95))

keys(fcast)                          # (:ds, :y_hat, :lo_80, :hi_80, :lo_95, :hi_95)
fcast.ds[1:3]                        # 2020-01-01, 2020-02-01, 2020-03-01
round.(fcast.y_hat[1:3]; digits = 1) # [71.4, 76.5, 76.6]
round.(fcast.lo_95[1:3]; digits = 1) # [56.0, 60.5, 61.0]
round.(fcast.hi_95[1:3]; digits = 1) # [75.7, 80.5, 80.6]
```

`y_hat` is the seasonal naive forecast: each step repeats the value one season earlier. It
is not the median of the draws. The interval columns come from 100 draws per step, which
[`forecast_samples`](@ref) returns in long format, one row per step and draw:

```julia
draws = forecast_samples(fitted, 12)

keys(draws)                          # (:ds, :sample, :y_hat)
length(draws.y_hat)                  # 1200 = 12 steps * 100 draws
draws.sample[1:3]                    # [1, 2, 3]
round.(draws.y_hat[1:3]; digits = 1) # [72.4, 70.4, 73.7], draws 1 to 3 of step 1
```

`backtest` scores forecasts of held-out windows. The metrics [`crps`](@ref),
[`Coverage`](@ref) and [`IntervalWidth`](@ref) are scored on the fold's draws, and
`levels` adds the interval columns to `forecasts`:

```julia
r = backtest(csp, df; horizon = 12, windows = 3, levels = (95,),
             metrics = (mae, crps, Coverage(95), IntervalWidth(95)))

keys(r.forecasts)   # (:fold, :origin, :step, :ds, :y, :y_hat, :lo_95, :hi_95)
s = summarize(r)
s.metric            # [:mae, :crps, :coverage_95, :interval_width_95]
round.(s.mean; digits = 2)   # [2.3, 1.7, 0.89, 16.45]
round.(s.std; digits = 2)    # [0.53, 0.29, 0.19, 2.6]
```

The three folds of 12 steps give 36 held-out values: far too few to judge coverage. They
only show how the calls fit together.

## How the method works

For one series `y[1:T]`, season length `m`, and each horizon `h = 1, ..., H`, CSP draws
`B` values (`samples`, 100 by default) from a mixture of two pools. The steps follow
Algorithm 1 of the paper.

1. **The centre.** `mu_h` is the seasonal naive forecast, `y[T + h - m * ceil(h / m)]`:
   the latest value at the same position in the season.
2. **The residual pool `R`.** The signed seasonal differences `y[t] - y[t - m]` over the
   last `calibration` share (0.5) of the series, `t > m`. With 96 values and the default
   share, `R` holds 48 differences. Signed means a difference keeps its sign, so the pool
   can be asymmetric. It is one pool for all horizons.
3. **The seasonal pool `S_h`.** The past values at the same position in the season as
   `T + h`, weighted by recency: the value at time `t` has weight `exp(-recency * (T - t))`
   with `recency = 0.01`.
4. **The mixture weight `w_h`.** With the default
   [`MachineLearningForecast.AdaptiveWeight`](@ref): 0 when `m = 1` (no seasonality),
   0.3 when `S_h` holds one or two values, 0.5 from three on.
5. **Each draw.** Draw `u` uniformly from (0, 1). If `u < w_h`, take a weighted draw from
   `S_h`. Otherwise take `mu_h + r` with `r` drawn uniformly from `R`. Repeat `B` times.

The newest value of `S_h` is the centre itself, so a share of the draws equals `y_hat`
exactly. The intervals are read from the `B` draws of each step (see
[Levels and columns](@ref)).

Some choices are ours, because the paper leaves them open:

- **The unit of `recency`.** We count `T - t` in time steps of `freq`, so the weights
  fall by `exp(-recency * m)` per season (0.887 per season for `m = 12`). The paper gives
  the rate "in normalized time-index units" (p. 13), which is ambiguous. Per step is our
  reading. `recency = 0` weighs the pool equally.
- **The sample quantile.** See the next section.
- **The random generator and how series use it.** See [The draws and the random
  generator](@ref).

The paper's own choices are the defaults.

## Levels and columns

A level is a percentage in (1, 100): `levels=(80, 95)`. A fraction such as `0.95` is an
`ArgumentError` that names the fix, so it is never read as a 0.95 % interval. A single
number (`levels=95`) is allowed; repeating a level is an error.

```julia
forecast(fitted, 12; levels = 0.95)
# ArgumentError: an interval level is a percentage in (1, 100), got 0.95. Pass e.g. 95
# for a 95% interval, not 0.95.
```

`forecast` and `backtest` add `lo_<L>` and `hi_<L>` after `y_hat`, in the order the levels
are given: `lo_80`, `hi_80`, `lo_95`, `hi_95`. A level with a decimal part uses an
underscore: `99.5` gives `lo_99_5` and `hi_99_5`. A Float32 level counts as it prints
(`Float32(99.9)` is 99.9). A level that would give the name of an existing column (for
example a time column called `lo_95`) is an error.

The interval is the central one: the sample quantiles of the step's `B` draws at
`(100 - L) / 200` and `(100 + L) / 200`. The quantile is Hyndman & Fan (1996) type 6, which
is our choice (the paper names none). Type 6 puts the `k`-th smallest of `B` draws at
probability `k / (B + 1)`, so, for draws without ties, a new draw from the same
distribution falls below it with that probability on average. For 100 draws and `L = 95`
the bounds are at positions 2.525 and 98.475 of the sorted draws. A level above
`100 (B - 1) / (B + 1)`, about 98 for 100 draws, gives the smallest and largest draw:

```julia
f84 = fit(csp, (ds = t[1:84], y = y[1:84]))
fc99 = forecast(f84, 12; levels = 99)
keys(fc99)                  # (:ds, :y_hat, :lo_99, :hi_99)
```

Here `fc99.lo_99` equals the minimum of the step's 100 draws, and `fc99.hi_99` the maximum.

## The draws and the random generator

`ConformalSeasonalPool` and `PoolIntervals` take `rng=Random.Xoshiro(42)`. The seed 42 is
the paper's, but our draws are not the paper's random stream. The method copies the
generator when it is built, and again for every series at every call. So:

- repeated calls give the same draws, and the generator you passed in, or the method's
  own, is never advanced;
- the draws are reproducible on the same Julia version. For draws that also reproduce
  across versions, pass a `StableRNGs.StableRNG`;
- a series' draws do not depend on which other series are fitted with it, or on the order
  of the rows.

```julia
forecast_samples(f84, 12).y_hat == forecast_samples(f84, 12).y_hat   # true
```

Because every series (and every backtest fold) starts from its own copy of the same
generator, they start from the same random numbers on different histories. With the default
`Xoshiro` the numbers almost always stay aligned; with a `StableRNG` they do not, because a
draw from the residual pool can use more than one number, how many depending on the pool's
size. Sharing a generator has two consequences:

- the draws of different series are dependent. Never combine them by sample number, for
  example to sum a panel: the sum of the first draw of each series is not a draw of the
  sum, and its spread is wrong;
- with the default `Xoshiro`, the Monte Carlo errors of the folds are correlated, so they
  do not average out over folds as they would with independent draws. To see their size,
  run the backtest with several seeds:

```julia
using Random
for seed in (1, 2, 3, 4, 5, 42)
    c = ConformalSeasonalPool(12; freq = Month(1), rng = Xoshiro(seed))
    rs = backtest(c, df; horizon = 12, windows = 3, metrics = (Coverage(95),))
    println(seed, " ", round(only(summarize(rs).mean); digits = 3))
end
# 1 0.917, 2 0.889, 3 0.917, 4 0.917, 5 0.917, 42 0.889
```

Horizons are drawn independently. The draws of step 1 and of step 2 with the same sample
number are not parts of one path, so the samples cannot give the distribution of a sum
over steps. The paper states the same limit (p. 12).

## Scores

All of these take the actuals `y` and a matrix `S` whose row `i` holds the draws for
`y[i]`. The long table of [`forecast_samples`](@ref) lists the draws of a step one after
the other, so a matrix with a row per step is a reshape and a transpose. Here the first 84
months are the training data and the last 12 months are held out:

```julia
actual = y[85:96]
d = forecast_samples(f84, 12)
S = permutedims(reshape(d.y_hat, 100, 12))   # 12 rows (steps), 100 columns (draws)

round(crps(actual, S); digits = 2)             # 1.67
round(fair_crps(actual, S); digits = 2)        # 1.64
round(pinball_loss(actual, S); digits = 2)     # 0.94
Coverage(95)(actual, S)                        # 1.0
round(IntervalWidth(95)(actual, S); digits = 1) # 19.2
```

- [`crps`](@ref) is the CRPS (Gneiting & Raftery 2007) of the empirical distribution of
  the draws, averaged over the rows. This plug-in form scores the `B` draws as issued, so
  it is the one for verifying the forecast you give out (Zamo & Naveau 2018).
- [`fair_crps`](@ref) uses the spread term over the `B (B - 1)` pairs of distinct draws
  (Ferro, Richardson & Weigel 2008). Use it to compare forecasts with different numbers
  of draws, or to score the distribution the draws come from. It needs `B >= 2`.
  The plug-in `crps` is higher by about `E|X - X'| / (2B)`.
- [`pinball_loss`](@ref) is the plain mean of the pinball loss over the quantiles
  0.1, 0.2, ..., 0.9, the quantiles the paper reports (p. 8). It is not normalised: that
  is our choice.
- [`Coverage`](@ref) is the share of actuals inside the central interval. A calibrated
  forecast scores near `level / 100`; higher is not better.
- [`IntervalWidth`](@ref) is the mean width of the same interval. Compare widths only
  between forecasts with similar coverage (Gneiting & Katzfuss 2014: calibration first,
  then sharpness).

Lower is better for `crps`, `fair_crps` and `pinball_loss`; a lower `IntervalWidth` is
better only at similar coverage. These are
the same numbers `backtest` computes. The `crps` above equals the one of the last fold in
`r.scores`, because that fold trains on the same 84 months and draws from the same copy of
the generator.

In `backtest`, each sample score is computed on a fold's draws, pooled over all series of
the fold. [`tune`](@ref) refuses these metrics. [`summarize`](@ref) reports the
unweighted mean of the per-fold values. For `Coverage` and `IntervalWidth` that is not
the same as the share or the mean over all rows of `forecasts` when folds have different
numbers of rows, as in a ragged panel (next section). The pooled form is:

```julia
f = r.forecasts
mean(f.lo_95 .<= f.y .<= f.hi_95)    # share of all rows inside the interval
mean(f.hi_95 .- f.lo_95)             # mean width over all rows
```

For the single series above both agree with `summarize`, since each fold has 12 rows.

## Panels

With `id`, every series has its own pools and its own centre; no model is shared (the
seasonal naive model has nothing to learn). Series may differ in length:

```julia
yb = y .+ 100
yc = [20 + 5 * cos(2π * i / 12) + 0.1i + cos(i^2) for i in 1:60]
panel = (id = vcat(fill("A", 96), fill("B", 96), fill("C", 60)),
         ds = vcat(t, t, t[1:60]), y = vcat(y, yb, yc))

cp = ConformalSeasonalPool(12; freq = Month(1), id = :id)
fp = fit(cp, panel)

out = forecast(fp, 12; levels = 95)
keys(out)                       # (:id, :ds, :y_hat, :lo_95, :hi_95)
length(out.y_hat)               # 36 = 3 series * 12 steps
dp = forecast_samples(fp, 12)
keys(dp)                        # (:id, :ds, :sample, :y_hat)
length(dp.y_hat)                # 3600 = 3 series * 12 steps * 100 draws
```

Series `B` is series `A` plus 100, and both have 96 values. They use the same random
numbers, so their draws differ by 100, sample by sample (up to rounding). Series `A`
also gets the draws it gets when fitted alone:

```julia
a = dp.y_hat[dp.id .== "A"]
b = dp.y_hat[dp.id .== "B"]
b ≈ a .+ 100                                  # true
a == forecast_samples(fitted, 12).y_hat       # true
```

This is the dependence described above. Do not sum `a` and `b` by sample number.

A panel `backtest` pools the sample scores over all series of a fold, as it does for an
unscaled point metric, so the series with large values dominate. If you need one score
per series, coverage and width follow from `forecasts` (with `f` below,
`g = f.id .== "C"; mean(f.lo_95[g] .<= f.y[g] .<= f.hi_95[g])`). A series backtested alone
with the same fold origins gets the same draws as in the panel. With `initial`
the folds are cut on the global time grid. Series `C` ends 36 months before the others,
so it has fewer actuals in the folds:

```julia
rp = backtest(cp, panel; horizon = 12, initial = 54, levels = 95,
              metrics = (Coverage(95), IntervalWidth(95)))
f = rp.forecasts
[count(==(k), f.fold) for k in unique(f.fold)]   # [30, 24, 24] rows per fold
round.(summarize(rp).mean; digits = 3)           # [0.883, 14.739]
round(mean(f.lo_95 .<= f.y .<= f.hi_95); digits = 3)   # 0.885
round(mean(f.hi_95 .- f.lo_95); digits = 2)            # 14.48
```

The unweighted mean of the fold values (0.883 and 14.739) differs from the pooled values
(0.885 and 14.48).

## Settings

| Keyword | Default | Meaning |
|---|---|---|
| `period` | required | season length in steps of `freq`; 1 for none |
| `weight` | `AdaptiveWeight()` | the seasonal-pool weight; `FixedWeight(w)` is a constant, `FixedWeight(0.5)` is the paper's CSP-Fixed |
| `calibration` | 0.5 | the share of each series' end that forms the residual pool, in (0, 1] |
| `recency` | 0.01 | decay per step of the seasonal-pool weights; 0 weighs equally |
| `samples` | 100 | draws per series and step |
| `rng` | `Xoshiro(42)` | the generator, copied as described above |

`AdaptiveWeight` and `FixedWeight` are public but not exported:
`using MachineLearningForecast: FixedWeight`. The defaults are the paper's.

**Choosing `period`.** It is the number of steps of `freq` in one cycle: 12 for monthly
data with a yearly cycle, 7 for daily data with a weekly cycle, 24 for hourly data with a
daily cycle, 52 for weekly data with a yearly cycle. `period = 1` means no seasonality: the
centre is the last value, the residual pool holds first differences, and the seasonal pool
is not used (`w = 0` with the default weight). A series needs at least `period + 1`
rows; with fewer, `fit` throws an `ArgumentError` that asks for at least `period + 1`
rows. A panel skips such a series with a warning, so it gets no forecast and no intervals.

A series also needs enough values for the residual pool: with the default share, the pool
holds `min(floor(T / 2), T - m)` differences (1 for `T = m + 1`, 8 for `T = 20` and
`m = 12`). If the share is so small that `floor(calibration * T)` is 0, the forecast is an
`ArgumentError` that names the step and the fix:

```julia
cs = ConformalSeasonalPool(12; freq = Month(1), calibration = 0.01)
forecast(fit(cs, df), 3; levels = 95)
# ArgumentError: the residual pool for step 1 is empty: the last calibration=0.01 of the
# 96 training values hold no seasonal difference y[t] - y[t - 12]. Raise calibration or
# provide a longer series.
```

A [`Forecaster`](@ref) built by hand accepts the interval method
([`MachineLearningForecast.PoolIntervals`](@ref)) only in the setup
`ConformalSeasonalPool` builds: a [`SeasonalNaive`](@ref) model, `FeatureSet(Lag(period))`,
`Recursive()` and no target transform. Other setups are refused with an error that names
what to change.

## What is not promised

The method has no coverage guarantee. Its residual pool is motivated by split-conformal
calibration (Papadopoulos et al. 2002), whose finite-sample guarantee needs exchangeable
data; serially dependent or non-stationary series are not. The paper makes no formal
coverage claim, for the residual pool alone or for the mixture, and reports coverage as an
empirical property: a mean of 0.89 at the nominal 0.95 on its benchmark (Table 1, p. 7).
That is the paper's result on its data, not ours.

Other limits:

- The residual pool is uniform over the calibration window and is not scaled by the
  series, so the intervals may be too wide or too narrow where the spread changes. The
  paper names scale normalisation as an untested extension (p. 12).
- The draws at steps `h` and `h + m` come from the same distribution, so the intervals
  repeat with the season and do not widen with the horizon. The error of a seasonal naive
  forecast usually grows with each season ahead (for a seasonal random walk, `k` seasons
  ahead it is a sum of `k` seasonal differences), so beyond the first season the intervals
  can be too narrow. With `FixedWeight(0.0)` the 95 % interval of the example series has the
  same width, about 3.5, at each of 24 steps (2000 draws).
- The seasonal pool holds old values. On a trending series such as the example it pulls
  draws toward past levels and widens the interval: about 20 with the default weight,
  against about 3.5 with `FixedWeight(0.0)` (2000 draws). Whether that is right for your
  series is for a backtest to show.
- The draws are not paths (see above).
- The recency rate, the calibration share and the number of draws are the paper's values,
  and it ran no sensitivity analysis for them (p. 12). The weights were fixed before it
  saw test scores and not tuned per dataset (p. 13).
- Coverage and sharpness depend on the series. Measure coverage on your data with
  `Coverage(95)` in a `backtest`, over many origins and series, and with several seeds.
