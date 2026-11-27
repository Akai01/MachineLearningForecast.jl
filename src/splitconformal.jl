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
