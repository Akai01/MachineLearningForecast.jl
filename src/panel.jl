"Group row indices by the id column, preserving first-appearance order."
function _group_rows(ids::AbstractVector, id::Symbol)
    order = Any[]
    groups = Dict{Any,Vector{Int}}()
    for (i, v) in enumerate(ids)
        ismissing(v) && throw(ArgumentError(
            "id column :$id has a missing value at row $i. Every row must belong " *
            "to a series; drop or fill the rows with a missing id before fitting."))
        rows = get(groups, v, nothing)
        if rows === nothing
            groups[v] = Int[i]
            push!(order, v)
        else
            push!(rows, i)
        end
    end
    return order, groups
end

"""
    panel_groups(fc, tbl, base=1:nrows(tbl)) -> Vector{Tuple{Any,NamedTuple,Vector{Int}}}

Split a long-format panel table into `(id, subtable, rows)` tuples in
first-appearance order, validating each series' time column independently.
Rows of a series need not be contiguous or sorted in the input: each series is
sorted by time, and its timestamps must be duplicate-free and gap-free with
respect to `fc.freq`. `rows[i]` is the user's row of the subtable's row `i`,
where `tbl`'s row `j` is the user's row `base[j]`.
"""
function panel_groups(fc::Forecaster, tbl::NamedTuple, base=1:nrows(tbl))
    ids = require_column(tbl, fc.id, "id")
    require_column(tbl, fc.time, "time")
    require_column(tbl, fc.target, "target")
    order, groups = _group_rows(ids, fc.id)
    isempty(order) && throw(ArgumentError(
        "the panel is empty: no rows found in the id column :$(fc.id). Pass a " *
        "table with at least one row."))
    out = Tuple{Any,NamedTuple,Vector{Int}}[]
    for key in order
        rows = groups[key]
        t = tbl[fc.time][rows]
        issorted(t) || (rows = rows[sortperm(t)])
        sub = row_subset(tbl, rows)
        user_rows = base[rows]
        _in_series(fc, key) do
            validate_time_column(sub[fc.time], fc.time, fc.freq, user_rows)
        end
        push!(out, (key, sub, user_rows))
    end
    return out
end

function _in_series(f, id::Symbol, key)
    try
        return f()
    catch err
        _rethrow_in(id, err, key)
    end
end

# Must run inside a catch block, for rethrow().
_rethrow_in(::Nothing, err, key) = rethrow()
function _rethrow_in(id::Symbol, err, key)
    err isa ArgumentError || rethrow()
    throw(ArgumentError("series $id=$(repr(key)): " * err.msg))
end

_series_groups(fc::Forecaster{M,S,Symbol}, tbl::NamedTuple,
               base=1:nrows(tbl)) where {M,S} = panel_groups(fc, tbl, base)

# One pass in table order names the first bad series.
function _prepare_all(fc::Forecaster{M,S,Symbol}, groups) where {M,S}
    mh = minhistory(fc)
    usable = nothing
    short = Tuple{Any,Int}[]
    for (id, sub, rows) in groups
        if nrows(sub) ≤ mh
            # The raw length rules it out, so no transform sees it.
            _in_series(() -> target_vector(sub, fc.target, rows), fc, id)
            push!(short, (id, nrows(sub)))
            continue
        end
        p = _prepare_series(fc, id, sub, rows)
        if nrows(p.tz) ≤ minhistory(fc.features)
            push!(short, (id, nrows(p.tz)))
        else
            usable = usable === nothing ? [p] : _push_widen!(usable, p)
        end
    end
    usable === nothing && throw(ArgumentError(
        "no series has enough rows to train on: $(_needs_clause(fc, "row")), and " *
        "the longest series has $(maximum(last, short; init=0)). Provide longer " *
        "series or reduce $(_reduce_hint(fc))."))
    return usable, short
end

function _warn_short(fc::Forecaster, short)
    isempty(short) && return nothing
    d = dropped_rows(fc.transform)
    names = join((string(fc.id, "=", repr(k), " (", n, " rows)")
                  for (k, n) in first(short, 5)), ", ")
    more = length(short) > 5 ? ", …" : ""
    need = _history_who(fc) * (d == 0 ? "" : " ($(minhistory(fc.features)) history " *
                                             "rows plus $d dropped by the transform)")
    @warn "skipping $(length(short)) series with fewer than $(minhistory(fc) + 1) " *
          "rows, which is the minimum $need: $names$more"
    return nothing
end

_fit_origin(fc::Forecaster{M,S,Symbol}, usable) where {M,S} =
    minimum(p.tz[fc.time][1] for p in usable)

"Vertically concatenate columntables that share a schema."
function _vcat_frames(Xs::Vector{<:NamedTuple})
    length(Xs) == 1 && return only(Xs)
    names = keys(first(Xs))
    # A vector, not a generator, hits vcat's linear fast path.
    return NamedTuple{names}(ntuple(j -> reduce(vcat, [X[names[j]] for X in Xs]),
                                    length(names)))
end

function _output_table(spec::Forecaster{M,S,Symbol}, states, grids, preds) where {M,S}
    n = sum(length, grids)
    idcol = Vector{Any}(undef, n)
    tcol = Vector{eltype(first(grids))}(undef, n)
    ycol = Vector{Float64}(undef, n)
    r = 1
    for k in eachindex(states), s in eachindex(grids[k])
        idcol[r] = states[k].id
        tcol[r] = grids[k][s]
        ycol[r] = preds[k][s]
        r += 1
    end
    return NamedTuple{(spec.id, spec.time, :y_hat)}(
        (identity.(idcol), tcol, ycol))
end

function _exogenous_rows(spec::Forecaster{M,S,Symbol}, states, grids, future,
                         hs) where {M,S}
    exogcols = exogenous_columns(spec.features)
    if isempty(exogcols)
        if future !== nothing
            @warn "`future` was passed but the feature set has no Exogenous " *
                  "features; it will be ignored."
        end
        return nothing
    end
    future === nothing && throw(ArgumentError(
        "features contain Exogenous($(join(":" .* string.(exogcols), ", "))) but " *
        "forecast() was called without `future`. Pass a table with columns " *
        "(:$(spec.id), :$(spec.time), $(join(":" .* string.(exogcols), ", "))) " *
        "covering all $(length(states)) series over their $(maximum(hs)) " *
        "forecast steps."))
    nd = normalize_table(future)
    needed = [spec.id; spec.time; exogcols]
    absent = [c for c in needed if !haskey(nd, c)]
    isempty(absent) || throw(ArgumentError(
        "`future` is missing column$(length(absent) == 1 ? "" : "s") " *
        "$(join(":" .* string.(absent), ", ")); a panel forecast needs the id " *
        "column :$(spec.id), the time column :$(spec.time) and the exogenous " *
        "column$(length(exogcols) == 1 ? "" : "s") " *
        "$(join(":" .* string.(exogcols), ", "))."))
    idv, tv = nd[spec.id], nd[spec.time]
    lookup = Dict{Tuple{Any,Any},Int}()
    for i in eachindex(tv)
        key = (idv[i], _new_time(tv[i], eltype(first(grids)), spec.time, i))
        haskey(lookup, key) && throw(ArgumentError(
            "`future` has duplicate rows for $(spec.id)=$(repr(idv[i])) at " *
            "$(tv[i]). Deduplicate `future` — otherwise which row supplies each " *
            "forecast step is arbitrary."))
        lookup[key] = i
    end
    names = Tuple(exogcols)
    out = Vector{Vector{NamedTuple}}(undef, length(states))
    for (k, st) in enumerate(states)
        rows = Vector{NamedTuple}(undef, length(grids[k]))
        for (s, t) in enumerate(grids[k])
            i = get(lookup, (st.id, t), nothing)
            i === nothing && throw(ArgumentError(
                "`future` has no row for $(spec.id)=$(repr(st.id)) at $t, which is " *
                "step $s of that series' forecast. Provide exogenous values for " *
                "every series at every forecast timestamp."))
            row = NamedTuple{names}(Tuple(nd[c][i] for c in exogcols))
            for c in exogcols
                ismissing(row[c]) && throw(ArgumentError(
                    "`future` has a missing value in exogenous column :$c for " *
                    "$(spec.id)=$(repr(st.id)) at $t. Provide complete exogenous " *
                    "values for every series at every forecast step."))
            end
            rows[s] = row
        end
        out[k] = rows
    end
    return out
end

"Each row's 1-based position among the rows of its id so far."
function _steps_by_id(ids::AbstractVector)
    seen = Dict{eltype(ids),Int}()
    return Int[seen[id] = get(seen, id, 0) + 1 for id in ids]
end

# One cutoff per series and fold: the last timestamp it trains on.
function _panel_cuts(s::FromStart, fc, groups, t_all, horizon)
    mh = minhistory(fc)
    grid = sort!(unique(t_all))
    ngrid = length(grid)
    s.initial > mh || throw(ArgumentError(
        "backtest initial=$(s.initial) must exceed $(_history_name(fc)) " *
        "($mh timestamps) so the first training window has at least one usable row. " *
        "For a panel, initial counts distinct timestamps, not rows. Pass " *
        "initial=$(mh + 1) or more, or reduce $(_reduce_hint(fc))."))
    origins = s.initial:s.step:(ngrid - horizon)
    isempty(origins) && throw(ArgumentError(
        "no complete backtest folds: the panel spans $ngrid distinct timestamps, " *
        "but the first fold needs initial + horizon = $(s.initial + horizon). " *
        "Provide more history or reduce initial/horizon."))
    keys_ = [key for (key, _, _) in groups]
    return [Dict(key => grid[o] for key in keys_) for o in origins]
end

function _panel_cuts(s::FromEnd, fc, groups, t_all, horizon)
    mh = minhistory(fc)
    longest = maximum(nrows(sub) for (_, sub, _) in groups)
    first = longest - horizon - (s.windows - 1) * s.step
    first > mh || throw(ArgumentError(
        "backtest windows=$(s.windows) with step=$(s.step) and horizon=$horizon " *
        "would train its first fold on at most $first rows of the longest series " *
        "($longest rows), but $(_history_who(fc)) more than $mh. Reduce windows, " *
        "step or horizon$(_transform_hint(fc)), or provide longer series."))
    # A series too short for a fold sits that fold out.
    return [Dict(key => sub[fc.time][c] for (key, sub, _) in groups
                 for c in (nrows(sub) - horizon - (s.windows - w) * s.step,) if c ≥ 1)
            for w in 1:s.windows]
end

_rows_before(cut, ids, t) =
    [i for i in eachindex(t) if haskey(cut, ids[i]) && t[i] ≤ cut[ids[i]]]
_rows_after(cut, ids, t) =
    [i for i in eachindex(t) if haskey(cut, ids[i]) && t[i] > cut[ids[i]]]

function _fold_origin(cut)
    origins = unique(values(cut))
    return length(origins) == 1 ? only(origins) : missing
end

function _backtest_panel(fc::Forecaster, tbl::NamedTuple, horizon, scheme, metrics)
    groups = panel_groups(fc, tbl)
    for (key, sub, rows) in groups                       # validates each series
        _in_series(() -> target_vector(sub, fc.target, rows), fc, key)
    end
    ids = tbl[fc.id]
    t_all = tbl[fc.time]
    y_all = target_vector(tbl, fc.target)
    cuts = _panel_cuts(scheme, fc, groups, t_all, horizon)
    exogcols = exogenous_columns(fc.features)

    actual = Dict{Tuple{Any,Any},Float64}()
    for i in eachindex(y_all)
        actual[(ids[i], t_all[i])] = y_all[i]
    end

    T = eltype(t_all)
    I = eltype(ids)
    fold_col = Int[]
    origin_col = T[]
    step_col = Int[]
    id_col = I[]
    time_col = T[]
    y_col = Float64[]
    yhat_col = Float64[]
    m_fold = Int[]
    m_origin = Union{Missing,T}[]
    m_metric = Symbol[]
    m_value = Float64[]

    for (k, cut) in enumerate(cuts)
        train_rows = _rows_before(cut, ids, t_all)
        fitted = _fit_spec(fc, row_subset(tbl, train_rows), train_rows)
        fcast = if isempty(exogcols)
            forecast(fitted, horizon)
        else
            nd = row_subset(tbl[Tuple([fc.id; fc.time; exogcols])],
                            _rows_after(cut, ids, t_all))
            # Steps past a series' data have no exogenous row or actual.
            hs = map(fitted.series) do st
                g = future_grid(st.t_start, fc.freq, st.n_train, horizon)
                something(findfirst(t -> !haskey(actual, (st.id, t)), g), horizon + 1) - 1
            end
            _forecast(fitted, hs, nd)
        end

        fid, ft, fy = fcast[fc.id], fcast[fc.time], fcast.y_hat
        steps = _steps_by_id(fid)
        ytrue = Float64[]
        yhat = Float64[]
        scored = eltype(fid)[]
        for j in eachindex(fy)
            a = get(actual, (fid[j], ft[j]), nothing)
            a === nothing && continue
            push!(ytrue, a)
            push!(yhat, fy[j])
            push!(scored, fid[j])
            push!(fold_col, k)
            push!(origin_col, cut[fid[j]])
            push!(step_col, steps[j])
            push!(id_col, fid[j])
            push!(time_col, ft[j])
            push!(y_col, a)
            push!(yhat_col, fy[j])
        end
        origin = _fold_origin(cut)
        isempty(ytrue) && throw(ArgumentError(
            "backtest fold $k (origin $origin) produced no forecast that lines up " *
            "with an actual: no series trained on data up to the origin has data " *
            "after it. Choose initial and step so each origin falls inside the data " *
            "of a series."))
        history = Dict(st.id => st.y_history for st in getfield(fitted, :series))
        fold = (; fc, history, ids=scored, y=ytrue, ŷ=yhat)
        for (name, m) in metrics
            push!(m_fold, k)
            push!(m_origin, origin)
            push!(m_metric, name)
            push!(m_value, Float64(score(metric_inputs(m), m, fold)))
        end
    end
    for (name, _) in metrics
        vals = [m_value[i] for i in eachindex(m_value)
                if m_metric[i] == name && m_fold[i] > 0]
        push!(m_fold, 0)
        push!(m_origin, missing)
        push!(m_metric, name)
        push!(m_value, Statistics.mean(vals))
    end
    folds = NamedTuple{(:fold, :origin, :step, fc.id, fc.time, :y, :y_hat)}(
        (fold_col, origin_col, step_col, id_col, time_col, y_col, yhat_col))
    return BacktestResult(folds, (fold=m_fold, origin=m_origin,
                                  metric=m_metric, value=m_value))
end
