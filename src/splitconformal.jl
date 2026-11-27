"Interval method that calibrates on the last `holdout` rows of each series."
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

# Single series only; a panel needs its own messages.
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
        "$(_history_who(fc)) more than $mh $(hist)rows. Provide at least " *
        "$(c + mh + 1) rows$lower."))
end

_check_head(iv, fc, ::Recursive, usable, rs) = nothing
function _check_head(iv::SplitConformal, fc::Forecaster{M,S,Nothing}, s::Direct,
                     usable, rs) where {M,S}
    c, mh, D = iv.holdout, minhistory(fc), s.max_horizon
    r = only(rs)
    r - mh ≥ D && return nothing
    n = length(only(usable).t)
    low = n - mh - D
    lower = low ≥ 9 ? "Lower holdout to $low or less, use" : "Use"
    d = dropped_rows(fc.transform)
    hist = iszero(d) ? "the feature set's $mh history rows" :
           "the feature set's $(mh - d) history rows and the transform's $d dropped rows"
    throw(ArgumentError(
        "intervals=$iv leaves $r of the $n rows to train the calibration model, " *
        "$(r - mh) after $hist, but $s needs $D. " *
        "$lower Direct($(r - mh)) or a shorter max_horizon, or provide at least " *
        "$(c + mh + D) rows."))
end

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
        return SeriesState(p.id, p.yraw, p.t[o], p.t[1], o, z, state, step0)
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

# One _forecast_states call serves the origins `is`.
function _calibrate_chunk!(E, iv, fc, head, usable, step0s, exog, is, depth)
    c = iv.holdout
    states = SeriesState[]
    hs = Int[]
    exog_rows = exog === nothing ? nothing : Vector{NamedTuple}[]
    for (k, p) in enumerate(usable), i in is
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
        p, n = first(usable), length(first(usable).t)
        a, b = (p.rows[n - c + i - 1] for i in (first(is), last(is)))
        throw(ArgumentError("intervals=$iv, calibrating origins at rows $a to $b: " *
                            err.msg))
    end
    m = 0
    for (k, p) in enumerate(usable), i in is
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
    c, mh = iv.holdout, minhistory(fc)
    _check_rows(iv, fc, groups)
    usable, short = _prepare_all(fc, groups)
    _warn_short(fc, short)
    exog = _exogenous_history(fc, usable, groups)
    cutoff = minimum(p.t[length(p.t) - c + 1] for p in usable)
    rmap = Dict(p.id => searchsortedfirst(p.t, cutoff) - 1 for p in usable)
    rs = [rmap[p.id] for p in usable]
    _check_head(iv, fc, fc.strategy, usable, rs)
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
        throw(ArgumentError("intervals=$iv, fitting the calibration model on the " *
                            "first $(sum(rs)) rows: " * err.msg))
    end
    depth = _depth(fc.strategy, c)
    E = [fill(NaN, c, depth) for _ in usable]
    for lo in 1:chunk:c
        is = lo:min(lo + chunk - 1, c)
        _calibrate_chunk!(E, iv, fc, head, usable, step0s, exog, is, depth)
    end
    return head, E, _published_states(usable, step0s)
end
