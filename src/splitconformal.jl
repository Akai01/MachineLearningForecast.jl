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
