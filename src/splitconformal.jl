"""
    SplitConformal(; holdout)

Interval method for any model: the value of the `intervals` field of a
[`Forecaster`](@ref). The last `holdout` rows of each series measure the model's own
forecast errors, and the errors give intervals and sample paths. There is no default for
`holdout`: pick it from the guide below.

`fit` trains the head model (one model per step under `Direct`) on the rows before the
holdout (a panel: before the earliest holdout start of any series). The head model then
forecasts from every origin in the holdout, which gives each series one error path per
origin. The published forecast comes from the head model, forecasting from the full
history. There is no second fit on all rows, so the errors in the pool are the errors of
the model that forecasts. This means `y_hat` differs from a forecaster without
`intervals`: the head model saw fewer rows, and `y_hat` changes when you add `intervals`.
`fit` costs the model fits of one normal fit plus the error paths.

# Intervals and samples

`forecast_samples(f, h)` returns the error paths: sample `i` is the point forecast plus
the errors of calibration origin `i` (1 is the oldest), at every step. At each step the
paths are the atoms of the crisp split conformal predictive distribution (Vovk et al.
2020, eq. 17), which has no validity theorem. There is no resampling and no rng, so
calls are deterministic. The pairing of errors across steps into paths is our design.
`forecast(f, h; levels)` returns type 6 sample quantiles of the same matrix, so intervals
and samples agree. Only origins that have all `h` actuals count: a call of `h` steps has
`K = holdout - h + 1` samples, and a longer call uses fewer, older origins at every step.
This keeps the sample matrix rectangular.

A level `L` (in percent) needs at least `need(L) = 2 / (1 - L / 100) - 1` errors,
rounded up: 9 for 80 %, 19 for 90 %, 39 for 95 %. A call that has fewer is refused with
the level the errors support and the fix. Samples need at least 9 errors per step, so
`forecast_samples` and `levels` reach at most `holdout - 8` steps; plain `forecast(f, h)`
is never refused by this rule. A `Direct(D)` strategy also refuses `h > D`. Rows needed
for a level are `need(L) + h - 1`, for example `holdout >= 50` for 95 % at 12 steps. This
rule makes the interval computable, it does not make it accurate.

# What the method promises

The method has no coverage guarantee. The intervals are empirical prediction intervals.
Their coverage is about `L` when the calibration errors and the error of the published
forecast are exchangeable, for example one step ahead from a well specified model with
independent shocks on a long series. Type 6 gives about `L`, not at least `L`: for
normal errors up to about 2 points above `L` at 80 % with small `K`, and about half a
point at 95 % near `K = 50`, less as `K` grows (our check). Time series are in general not
exchangeable. Theory for overlapping scores exists only as asymptotic results:
consistency of the empirical quantile for stationary ergodic series, and a central limit
result under mixing (Lee and Scholtes 2014; Chernozhukov, Wuthrich and Zhu 2018). None
gives a distribution-free guarantee for finite `K`, and Lee and Scholtes analyse a
forecaster refitted on a fixed window at every origin, not one head model. Measure
coverage by step with [`backtest`](@ref) and [`Coverage`](@ref).

The coverage of one fit is random given its calibration set: its standard deviation is
about 0.033 at `K = 40` and 0.015 at `K = 200` for a 95 % interval (our computation), so
a few hundred errors are needed before the tails are steady.

# Choosing `holdout`

There is no default rule. Give each series at least `need(L) + h - 1` rows for the level
`L` and the longest horizon `h` you will ask for:

| `h` | 80 % | 90 % | 95 % |
|----:|-----:|-----:|-----:|
|   1 |    9 |   19 |   39 |
|  12 |   20 |   30 |   50 |
|  24 |   32 |   42 |   62 |

A larger `holdout` gives a better pool of errors and a weaker model, because the head
model sees fewer rows. For a start, borrowed from accuracy-estimation and i.i.d.
sources and not measured here for your model: 20 to 30 % of the series, never fewer rows
than the table. Keep the head model's rows well above the feature set's history. A series
that is too short is refused, and the message names the fix.

Under `Recursive`, the calibration costs about `holdout^2 / 2` predict rows per series and
stores `8 holdout (holdout - 8)` bytes per series (25 MB per 1000 series at 60, 307 MB at
200).

# Differences from `ConformalSeasonalPool`

Unlike [`ConformalSeasonalPool`](@ref), which has no fitted model and reads one pool of
seasonal differences for all steps, `SplitConformal` fits a model on all but the last
`holdout` rows and stores the errors of that model at each step, one error path per
origin. `PoolIntervals` follows the paper and does not refuse a small pool;
`SplitConformal` refuses a level the sample cannot support.

# Limits

- The method assumes that the errors in the holdout resemble the errors of the forecast.
  A trending or seasonal series needs a [`TargetTransform`](@ref) such as
  [`Difference`](@ref) or [`LastScale`](@ref) first, as Lee and Scholtes (2014) require
  stationary data. A level shift in the holdout is not followed. A transform's state is
  fitted on all rows while the head model was trained under the state of the earlier
  rows; the errors include that.
- At long horizons coverage falls short for persistent series. In our simulations with
  known-parameter models (95 %, `h = 12`), the expected coverage is about 0.80 at
  `K = 40`, 0.87 at 80 and 0.91 at 160 for a random walk; 0.83, 0.89 and 0.92 for
  AR(0.9); 0.93, 0.94 and 0.95 for AR(0.5); and no shortfall for independent errors. For
  a random walk the shortfall is about `0.5 (h - 1) / K`, our own fit; it depends on the
  persistence of the series. `K >= 10 (h - 1)` leaves a shortfall of about 5 points and
  `K >= 25 (h - 1)` about 2 in that case. The coverage of one fit also varies more
  (standard deviation 0.127 at `K = 40` for the random walk against 0.033 for independent
  errors). There is no warning at run time.
- The errors use the realised values of `Exogenous` columns, so for covariates that are
  not known in advance the intervals are too narrow by the error of forecasting them (see
  [`backtest`](@ref)). The calibration rejects `missing` in every input column of an
  exogenous feature, even for a custom [`ExogenousFeature`](@ref) that handles `missing`
  itself, so it is stricter here than a plain `fit`.
- Panels: one head model fits all series, and each series has its own pool of errors. The
  head trains on rows before the earliest holdout start, so the forecast of a series that
  ends later comes from a model that has not seen its last rows, and it moves when a
  series with an earlier end is added. A series that ended long ago pulls the cutoff back
  for every series: leave out the series that ends earliest (under `Direct` the refusal
  names it) or lower `holdout`; trimming the later series does not help. Coverage is
  correlated across series.
- `show` of the fitted forecaster and `SeriesState.n_train` report the history length,
  not the rows the head model was fitted on.
- An error scale that depends on the season gives uneven coverage by month, though the
  average is near `L`. Intervals of sums over steps, such as annual totals, are the least
  reliable use and need `K` well above `need(L)`. The lower bound can fall below zero for
  positive data.
- [`crps`](@ref) on the `K` atoms is the CRPS of the crisp predictive distribution. Its
  plug-in form is biased upward by a factor of about `(K + 1) / K` (1.2, 1.1 and 1.025 at
  `K` = 5, 10 and 40), so compare different `K` with [`fair_crps`](@ref).
- [`tune`](@ref) chooses candidates on rows that include the holdout, so the chosen
  forecaster is no longer calibrated by construction.

# Example
```julia
using MachineLearningForecast, Dates
using MLJDecisionTreeInterface: DecisionTreeRegressor
dates = collect(Date(2010, 1, 1):Month(1):Date(2019, 12, 1))
df = (ds=dates, y=[10 + 3 * sin(2pi * i / 12) + sin(i^2) for i in eachindex(dates)])

fc = Forecaster(DecisionTreeRegressor(max_depth=4);
                features=FeatureSet(Lag(1), Lag(12)), freq=Month(1),
                intervals=SplitConformal(holdout=60))
fitted = fit(fc, df)                       # the model is fitted without the last 60 rows
forecast(fitted, 12; levels=(80, 95))      # y_hat, lo_80, hi_80, lo_95, hi_95
forecast_samples(fitted, 12)               # 49 error paths of 12 steps each
backtest(fc, df; horizon=12, initial=84, step=12, levels=(95,),
         metrics=(crps, Coverage(95)))
```
"""
struct SplitConformal
    holdout::Int
    function SplitConformal(; holdout=nothing)
        holdout === nothing && throw(ArgumentError(
            "SplitConformal needs holdout, the number of last rows of each series " *
            "that measure forecast errors. It has no default. Pass e.g. " *
            "SplitConformal(holdout=60), which is enough for a 95 % interval up to " *
            "12 steps ahead (it needs holdout >= 50 there)."))
        fix = "Pass the number of last rows that measure errors, e.g. " *
              "SplitConformal(holdout=60)."
        _whole(holdout, "SplitConformal holdout", fix)
        holdout >= 9 || throw(ArgumentError(
            "SplitConformal holdout must be a whole number >= 9 (the last rows of " *
            "each series that measure forecast errors; 9 errors give an 80 % " *
            "interval at one step), got $holdout. Pass e.g. SplitConformal(holdout=60)."))
        return new(Int(holdout))
    end
end

SplitConformal(x) = throw(ArgumentError(
    "SplitConformal takes the keyword holdout, got the positional argument " *
    "$(repr(x)). Pass e.g. SplitConformal(holdout=60)."))

Base.show(io::IO, iv::SplitConformal) =
    print(io, "SplitConformal(holdout=", iv.holdout, ")")

# Floats give 2 / (1 - 0.8) = 10.000000000000002.
function _need_errors(L::Float64)
    Lr = round(Int, L * 10_000)
    return cld(2_000_000, max(1_000_000 - Lr, 1)) - 1
end

# Floor, so the printed level is always supported.
function _max_level_text(K::Integer)
    q = (10_000 * (K - 1)) ÷ (K + 1)
    return "$(q ÷ 100).$(lpad(q % 100, 2, '0'))"
end

_depth(strategy, c::Integer) = min(horizon_limit(strategy), c - 8)

_level_text(L::Float64) = isinteger(L) ? string(Int(L)) : string(L)

# Levels nothing or empty get the depth rule only.
function _support_message(c::Int, h::Int, depth::Int, levels, who::AbstractString)
    K = c - h + 1
    back = who == "backtest horizon"
    if h > depth
        n = "$depth step$(depth == 1 ? "" : "s")"
        steps = back ? "Use horizon <= $depth" : "Forecast at most $n"
        return "$who $h needs at least 9 calibration errors per step, but " *
               "intervals=SplitConformal(holdout=$c) gives $(max(K, 0)) at that " *
               "horizon (the origins that have $h actuals). $steps, or fit with " *
               "holdout >= $(h + 8)."
    end
    (levels === nothing || isempty(levels)) && return nothing
    L = maximum(levels)
    need = _need_errors(L)
    K >= need && return nothing
    top = _max_level_text(K)
    max_h = c - need + 1
    n = "$max_h step$(max_h == 1 ? "" : "s")"
    steps = max_h < 1 ? "" : back ? "use horizon <= $max_h, " : "forecast at most $n, "
    return "levels=($(_level_text(L)),) needs at least $need calibration errors per " *
           "step, but intervals=SplitConformal(holdout=$c) gives $K at $who $h " *
           "(the origins that have $h actual$(h == 1 ? "" : "s")), which supports " *
           "central levels up to $top. Pass levels <= $top, $(steps)or fit with " *
           "holdout >= $(need + h - 1)."
end

# Bounds the history copies that one call holds.
const _CALIBRATION_STATES = 4096

function _check_rows(iv::SplitConformal, fc::Forecaster{M,S,Nothing},
                     groups) where {M,S}
    c, mh = iv.holdout, minhistory(fc)
    n = nrows(only(groups)[2])
    n ≥ c + mh + 1 && return nothing
    low = n - mh - 1
    lower = low ≥ 9 ? " or lower holdout to $low or less" : ""
    hist = iszero(dropped_rows(fc.transform)) ? "history " : ""
    throw(ArgumentError(
        "the data has $n rows. intervals=$iv holds out the last $c rows to measure " *
        "errors and leaves $(max(n - c, 0)) to train the calibration model, but " *
        "$(_history_who(fc)) more than $mh $(hist)row$(mh == 1 ? "" : "s"). " *
        "Provide at least " *
        "$(c + mh + 1) rows$lower."))
end

# Series with n <= mh are the point fit's to skip.
function _check_rows(iv::SplitConformal, fc::Forecaster{M,S,Symbol},
                     groups) where {M,S}
    c, mh = iv.holdout, minhistory(fc)
    short = [(id, nrows(sub)) for (id, sub, _) in groups if mh < nrows(sub) < c + mh + 1]
    isempty(short) && return nothing
    k = length(short)
    names = join((string(fc.id, "=", repr(id), " (", n, " rows)")
                  for (id, n) in first(short, 5)), ", ")
    low = minimum(last, short) - mh - 1
    fix = low ≥ 9 ? "Provide longer series, leave these series out, or lower holdout " *
                    "to $low or less." : "Provide longer series or leave these series out."
    throw(ArgumentError(
        "$k series $(k == 1 ? "is" : "are") too short for intervals=$iv, which holds " *
        "out the last $c rows of each series to measure errors and needs at least " *
        "$(c + mh + 1) rows per series: $names$(k > 5 ? ", ..." : ""). $fix"))
end

function _hist_text(fc)
    mh, d = minhistory(fc), dropped_rows(fc.transform)
    rows(k, what) = "$k $what row$(k == 1 ? "" : "s")"
    return iszero(d) ? "the feature set's $(rows(mh, "history"))" :
           "the feature set's $(rows(mh - d, "history")) and the transform's " *
           rows(d, "dropped")
end

_check_head(iv, fc, ::Recursive, usable, rs, cutoff) = nothing
function _check_head(iv::SplitConformal, fc::Forecaster{M,S,Nothing}, s::Direct,
                     usable, rs, cutoff) where {M,S}
    c, mh, D = iv.holdout, minhistory(fc), s.max_horizon
    r = only(rs)
    r - mh ≥ D && return nothing
    n = length(only(usable).t)
    low = n - mh - D
    lower = low ≥ 9 ? "Lower holdout to $low or less, use" : "Use"
    throw(ArgumentError(
        "intervals=$iv leaves $r of the $n rows to train the calibration model, " *
        "$(r - mh) after $(_hist_text(fc)), but $s needs $D. " *
        "$lower Direct($(r - mh)) or a shorter max_horizon, or provide at least " *
        "$(c + mh + D) rows."))
end

function _check_head(iv::SplitConformal, fc::Forecaster{M,S,Symbol}, s::Direct,
                     usable, rs, cutoff) where {M,S}
    c, mh, D = iv.holdout, minhistory(fc), s.max_horizon
    r = maximum(rs)
    r - mh ≥ D && return nothing
    low = c - D + r - mh
    lower = low ≥ 9 ? "Lower holdout to $low or less, use" : "Use"
    setter = usable[findfirst(p -> p.t[length(p.t) - c + 1] == cutoff, usable)]
    throw(ArgumentError(
        "intervals=$iv trains the calibration model on rows before $cutoff, the " *
        "earliest holdout start (series $(fc.id)=$(repr(setter.id))). That leaves $r " *
        "rows in the series with the most rows before it, $(r - mh) after " *
        "$(_hist_text(fc)), but $s needs $D. $lower Direct($(r - mh)) or a shorter " *
        "max_horizon, or provide " *
        "$(D - r + mh) more rows before $cutoff in that series."))
end

_head_rows(::Forecaster{M,S,Nothing}, n, cutoff) where {M,S} = "the first $n rows"
_head_rows(::Forecaster{M,S,Symbol}, n, cutoff) where {M,S} = "the $n rows before $cutoff"

_published_states(usable, step0s) = SeriesState[
    SeriesState(p.id, p.yraw, p.t[end], p.t[1], length(p.t), p.z, p.state, s0)
    for (p, s0) in zip(usable, step0s)]

function _nonfinite_error(fc, p, o, j, v)
    who = fc.id === nothing ? "" : "series $(fc.id)=$(repr(p.id)): "
    return ArgumentError(
        "$(who)the calibration forecast from origin $(p.t[o]) (row $(p.rows[o])) is $v " *
        "at step $j. A non-finite forecast cannot give an interval. Check the " *
        "model's hyperparameters and the training data; a smaller holdout " *
        "calibrates fewer steps.")
end

# A fit on rows 1..o would get this state.
function _pseudo_state(iv, fc, p, o, step0)
    yp = p.yraw[1:o]
    try
        _, z, state = _transform_table(fc, NamedTuple{(fc.target,)}((yp,)), yp)
        return SeriesState(p.id, yp, p.t[o], p.t[1], o, z, state, step0)
    catch err
        err isa ArgumentError || rethrow()
        who = fc.id === nothing ? "" : "series $(fc.id)=$(repr(p.id)): "
        throw(ArgumentError("$(who)intervals=$iv, calibrating the origin at row " *
                            "$(p.rows[o]): " * err.msg))
    end
end

# Scans the rows a fit would, as build_training_frame does.
function _exogenous_history(fc, usable, groups)
    cols = unique(exogenous_columns(fc.features))
    isempty(cols) && return nothing
    mh, names = minhistory(fc), Tuple(cols)
    subs = Dict(g[1] => g[2] for g in groups)
    return map(usable) do p
        sub = subs[p.id]
        _in_series(fc, p.id) do
            foreach(c -> _require_input_column(sub, c), cols)
            for c in cols, i in (mh + 1):length(p.t)
                ismissing(sub[c][i]) && throw(ArgumentError(
                    "feature column :$c has a missing value at row $(p.rows[i]) " *
                    "(time $(p.t[i])). Missing values in exogenous columns are not " *
                    "supported; impute or drop them before fitting."))
            end
        end
        return NamedTuple[NamedTuple{names}(Tuple(sub[c][i] for c in cols))
                          for i in eachindex(p.t)]
    end
end

function _origins_text(::Forecaster{M,S,Nothing}, usable, is, c) where {M,S}
    p = only(usable)
    a, b = (p.rows[length(p.t) - c + i - 1] for i in (first(is), last(is)))
    return "at rows $a to $b"
end
_origins_text(::Forecaster{M,S,Symbol}, usable, is, c) where {M,S} =
    "$(first(is)) to $(last(is)) of the $c in each series (origin 1 is the last " *
    "row before its holdout)"

# Origin-major: the first non-finite is chunk-independent.
function _calibrate_chunk!(E, iv, fc, head, usable, step0s, exog, is, depth)
    c = iv.holdout
    states = SeriesState[]
    hs = Int[]
    exog_rows = exog === nothing ? nothing : Vector{NamedTuple}[]
    for i in is, (k, p) in enumerate(usable)
        o = length(p.t) - c + i - 1
        push!(states, _pseudo_state(iv, fc, p, o, step0s[k]))
        push!(hs, min(c - i + 1, depth))
        exog === nothing || push!(exog_rows, exog[k][(o + 1):(o + hs[end])])
    end
    grids = _grids(states, fc.freq, hs)
    preds = try
        _forecast_states(head, states, hs, grids, exog_rows)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("intervals=$iv, calibrating origins " *
                            _origins_text(fc, usable, is, c) * ": " * err.msg))
    end
    m = 0
    for i in is, (k, p) in enumerate(usable)
        m += 1
        o = length(p.t) - c + i - 1
        for j in 1:hs[m]
            v = preds[m][j]
            isfinite(v) || throw(_nonfinite_error(fc, p, o, j, v))
            E[k][i, j] = p.yraw[o + j] - v
        end
    end
    return nothing
end

# The head model and what the calibration origins need.
function _fit_head(iv::SplitConformal, fc::Forecaster, groups)
    c, mh = iv.holdout, minhistory(fc)
    _check_rows(iv, fc, groups)
    usable, short = _prepare_all(fc, groups)
    _warn_short(fc, short)
    exog = _exogenous_history(fc, usable, groups)
    cutoff = minimum(p.t[length(p.t) - c + 1] for p in usable)
    rmap = Dict(p.id => searchsortedfirst(p.t, cutoff) - 1 for p in usable)
    rs = [rmap[p.id] for p in usable]
    _check_head(iv, fc, fc.strategy, usable, rs, cutoff)
    head_groups = [(id, row_subset(sub, 1:rmap[id]), rows[1:rmap[id]])
                   for (id, sub, rows) in groups if get(rmap, id, 0) > mh]
    head_tbl = _vcat_frames([g[2] for g in head_groups])
    head_base = reduce(vcat, [g[3] for g in head_groups])
    local head, step0s
    try
        origin = _fit_origin(fc, first(_prepare_all(fc, head_groups)))
        step0s = [_step0(fc, p.tz, origin) for p in usable]
        head = _fit_point(fc, head_tbl, head_base)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("intervals=$iv, fitting the calibration model on " *
                            _head_rows(fc, nrows(head_tbl), cutoff) * ": " * err.msg))
    end
    return head, usable, step0s, exog
end

"""
    _calibrate(iv, fc, groups; chunk) -> (head, triangles, states)

Fit the head model on the rows before the common cutoff and measure its errors at the
last `iv.holdout` origins of every series. `triangles[k]` is a `holdout x depth` matrix,
row `i` the origin `i` rows into the holdout and column `j` step `j`; entries without an
actual stay `NaN`. `states` are the full-history states the head forecasts from.
`chunk` is the number of origins per `_forecast_states` call.
"""
function _calibrate(iv::SplitConformal, fc::Forecaster, groups;
                    chunk::Integer=max(1, _CALIBRATION_STATES ÷ length(groups)))
    c = iv.holdout
    head, usable, step0s, exog = _fit_head(iv, fc, groups)
    depth = _depth(fc.strategy, c)
    E = [fill(NaN, c, depth) for _ in usable]
    for lo in 1:chunk:c
        is = lo:min(lo + chunk - 1, c)
        _calibrate_chunk!(E, iv, fc, head, usable, step0s, exog, is, depth)
    end
    return head, E, _published_states(usable, step0s)
end

_check_pairing(model, features, strategy, transform, ::SplitConformal) = nothing
_require_intervals(::SplitConformal, who) = nothing

# The head publishes: no final fit on all rows.
function _fit_spec(iv::SplitConformal, fc, tbl, base)
    head, E, states = _calibrate(iv, fc, _series_groups(fc, tbl, base))
    return FittedForecaster(fc, head.models, states, head.feature_names, E)
end

# A fold needs the published head, not the triangle.
function _fit_fold(iv::SplitConformal, fc, tbl, base)
    head, usable, step0s, _ = _fit_head(iv, fc, _series_groups(fc, tbl, base))
    return FittedForecaster(fc, head.models, _published_states(usable, step0s),
                            head.feature_names)
end

function _sample_matrices(iv::SplitConformal, f::FittedForecaster, preds)
    E = getfield(f, :calibration)
    hmax = max(1, maximum(length, preds))
    depth = _depth(f.spec.strategy, iv.holdout)
    hmax > depth && throw(ArgumentError(
        _support_message(iv.holdout, hmax, depth, nothing, "forecast horizon")))
    K = iv.holdout - hmax + 1
    return Matrix{Float64}[preds[k] .+ transpose(E[k][1:K, 1:length(preds[k])])
                           for k in eachindex(preds)]
end

_strategy(f::FittedForecaster) = f.spec.strategy
_strategy(fc::Forecaster) = fc.strategy

# Direct's own horizon check goes first, so its text stands.
function _check_support(iv::SplitConformal, f, lv, h, who)
    _check_horizon(f, h)
    msg = _support_message(iv.holdout, h, _depth(_strategy(f), iv.holdout), lv, who)
    msg === nothing || throw(ArgumentError(msg))
    return nothing
end
