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

function _in_series(f, fc::Forecaster, key)
    try
        return f()
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("series $(fc.id)=$(repr(key)): " * err.msg))
    end
end

# Skip short series so the rest can still train.
function _panel_frames(fc::Forecaster, groups)
    mh = minhistory(fc.features)
    usable = Tuple{Any,NamedTuple,Vector{Int},Vector{Float64},NamedTuple,
                   Vector{Float64},Any}[]
    short = Tuple{Any,Int}[]
    for (key, sub, rows) in groups
        yraw, tz, z, state = _in_series(fc, key) do
            yraw = target_vector(sub, fc.target, rows)
            return (yraw, _transform_table(fc, sub, yraw)...)
        end
        n = nrows(tz)
        n ≤ mh ? push!(short, (key, n)) :
                 push!(usable, (key, sub, rows, yraw, tz, z, state))
    end
    if isempty(usable)
        throw(ArgumentError(
            "no series has enough rows to train on: the feature set needs $mh " *
            "history rows before the first usable row, and the longest series has " *
            "$(maximum(last, short; init=0)). Provide longer series or reduce " *
            "lags/windows."))
    end
    if !isempty(short)
        names = join((string(fc.id, "=", repr(k), " (", n, " rows)")
                      for (k, n) in first(short, 5)), ", ")
        more = length(short) > 5 ? ", …" : ""
        @warn "skipping $(length(short)) series with fewer than $(mh + 1) rows, " *
              "which is the minimum the feature set needs: $names$more"
    end
    origin = minimum(u[5][fc.time][1] for u in usable)
    return map(usable) do (key, sub, rows, yraw, tz, z, state)
        step0 = _steps_between(origin, tz[fc.time][1], fc.freq)
        X, y, _ = _in_series(fc, key) do
            build_training_frame(fc.features, tz, fc.target, fc.time,
                                 rows[(nrows(sub) - nrows(tz) + 1):end]; step0)
        end
        return (key, X, y, sub, yraw, z, state, step0)
    end
end

"Vertically concatenate columntables that share a schema."
function _vcat_frames(Xs::Vector{<:NamedTuple})
    length(Xs) == 1 && return only(Xs)
    names = keys(first(Xs))
    # A vector, not a generator, hits vcat's linear fast path.
    return NamedTuple{names}(ntuple(j -> reduce(vcat, [X[names[j]] for X in Xs]),
                                    length(names)))
end

function _fit_panel(fc::Forecaster, tbl::NamedTuple, base=1:nrows(tbl))
    groups = panel_groups(fc, tbl, base)
    frames = _panel_frames(fc, groups)
    models = _panel_models(fc, fc.strategy, frames)
    states = [SeriesState(key, yraw, sub[fc.time][end], sub[fc.time][1], nrows(sub),
                          z, state, step0)
              for (key, _, _, sub, yraw, z, state, step0) in frames]
    return FittedForecaster(fc, models, states, outputnames(fc.features))
end

function _panel_models(fc::Forecaster, ::Recursive, frames)
    X = _vcat_frames([f[2] for f in frames])
    y = reduce(vcat, [f[3] for f in frames])
    return [_train(fc.model, X, y)]
end

# Shift within each series, as in single-series Direct.
function _panel_models(fc::Forecaster, s::Direct, frames)
    istarget = targetcolumnmask(fc.features)
    X = _vcat_frames([f[2] for f in frames])
    y = reduce(vcat, [f[3] for f in frames])
    lens = [length(f[3]) for f in frames]
    return map(1:s.max_horizon) do i
        # One gather per column, not one slice per series.
        m = sum(nX -> max(nX - i + 1, 0), lens)
        from, to = Vector{Int}(undef, m), Vector{Int}(undef, m)
        k = o = 0
        for nX in lens
            for j in 1:(nX - i + 1)
                k += 1
                from[k], to[k] = o + j, o + j + i - 1
            end
            o += nX
        end
        m == 0 && throw(ArgumentError(
            "not enough data for Direct($(s.max_horizon)): no series has enough " *
            "rows to train the step-$i model. Shorten max_horizon, provide longer " *
            "series, or reduce lags/windows."))
        Xi = NamedTuple{keys(X)}(ntuple(c -> values(X)[c][istarget[c] ? from : to],
                                        length(X)))
        _train(fc.model, Xi, y[to])
    end
end

# One predict call per step, not per series per step.
function _panel_batch(f::FittedForecaster, states, histories, step::Integer,
                      grids, exog_rows, active)
    spec = f.spec
    names = f.feature_names
    cols = [Vector{Float64}(undef, length(active)) for _ in names]
    scratch = [Vector{Float64}(undef, 1) for _ in names]
    bufs = _feature_buffers(spec.features)
    for (r, k) in enumerate(active)
        ex = exog_rows === nothing ? nothing : exog_rows[k][step]
        ctx = _context(states[k], histories[k], grids[k][step], step, ex)
        _fill_row!(scratch, bufs, spec.features, ctx)
        for j in eachindex(cols)
            cols[j][r] = scratch[j][1]
        end
    end
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

_forecast_panel(f::FittedForecaster, h::Integer, new_data) =
    _forecast_panel(f, fill(Int(h), nseries(f)), new_data)

# hs[k] is series k's horizon, which backtest may shorten.
function _forecast_panel(f::FittedForecaster, hs::Vector{Int}, new_data)
    spec = f.spec
    states = getfield(f, :series)
    grids = [future_grid(st.t_start, spec.freq, st.n_train, hs[k])
             for (k, st) in enumerate(states)]

    exogcols = exogenous_columns(spec.features)
    exog_rows = nothing
    if !isempty(exogcols)
        new_data === nothing && throw(ArgumentError(
            "features contain Exogenous($(join(":" .* string.(exogcols), ", "))) but " *
            "forecast() was called without new_data. Pass a table with columns " *
            "(:$(spec.id), :$(spec.time), $(join(":" .* string.(exogcols), ", "))) " *
            "covering all $(length(states)) series over their $(maximum(hs)) " *
            "forecast steps."))
        exog_rows = _panel_exogenous_rows(spec, exogcols, states, grids, new_data)
    elseif new_data !== nothing
        @warn "new_data was passed but the feature set has no Exogenous features; " *
              "it will be ignored."
    end

    preds = [Vector{Float64}(undef, hs[k]) for k in eachindex(states)]
    histories = [copy(st.z_history) for st in states]
    _panel_predict!(preds, f, spec.strategy, states, histories, grids, exog_rows, hs)
    preds = [untransform(spec.transform, preds[k], states[k].transform_state)
             for k in eachindex(states)]

    n = sum(hs)
    idcol = Vector{Any}(undef, n)
    tcol = Vector{eltype(first(grids))}(undef, n)
    ycol = Vector{Float64}(undef, n)
    r = 1
    for k in eachindex(states), s in 1:hs[k]
        idcol[r] = states[k].id
        tcol[r] = grids[k][s]
        ycol[r] = preds[k][s]
        r += 1
    end
    return NamedTuple{(spec.id, spec.time, :y_hat)}(
        (identity.(idcol), tcol, ycol))
end

function _panel_predict!(preds, f, ::Recursive, states, histories, grids, exog_rows, hs)
    trained = only(f.models)
    for s in 1:maximum(hs)
        active = findall(≥(s), hs)
        row = _panel_batch(f, states, histories, s, grids, exog_rows, active)
        ŷ = _predict(trained, row)
        for (r, k) in enumerate(active)
            v = Float64(ŷ[r])
            preds[k][s] = v
            push!(histories[k], v)
        end
    end
    return preds
end

function _panel_predict!(preds, f, strat::Direct, states, histories, grids, exog_rows, hs)
    h = maximum(hs)
    h ≤ strat.max_horizon || throw(ArgumentError(
        "strategy=Direct($(strat.max_horizon)) was fit with max_horizon=" *
        "$(strat.max_horizon) but forecast(h=$h) was requested. Refit with " *
        "Direct($h) or use Recursive()."))
    for s in 1:h
        active = findall(≥(s), hs)
        row = _panel_batch(f, states, histories, s, grids, exog_rows, active)
        ŷ = _predict(f.models[s], row)
        for (r, k) in enumerate(active)
            preds[k][s] = Float64(ŷ[r])
        end
    end
    return preds
end

function _panel_exogenous_rows(spec::Forecaster, exogcols, states, grids, new_data)
    nd = normalize_table(new_data)
    needed = [spec.id; spec.time; exogcols]
    absent = [c for c in needed if !haskey(nd, c)]
    isempty(absent) || throw(ArgumentError(
        "new_data is missing column$(length(absent) == 1 ? "" : "s") " *
        "$(join(":" .* string.(absent), ", ")); a panel forecast needs the id " *
        "column :$(spec.id), the time column :$(spec.time) and the exogenous " *
        "column$(length(exogcols) == 1 ? "" : "s") " *
        "$(join(":" .* string.(exogcols), ", "))."))
    idv, tv = nd[spec.id], nd[spec.time]
    lookup = Dict{Tuple{Any,Any},Int}()
    for i in eachindex(tv)
        key = (idv[i], _new_time(tv[i], eltype(first(grids)), spec.time, i))
        haskey(lookup, key) && throw(ArgumentError(
            "new_data has duplicate rows for $(spec.id)=$(repr(idv[i])) at " *
            "$(tv[i]). Deduplicate new_data — otherwise which row supplies each " *
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
                "new_data has no row for $(spec.id)=$(repr(st.id)) at $t, which is " *
                "step $s of that series' forecast. Provide exogenous values for " *
                "every series at every forecast timestamp."))
            row = NamedTuple{names}(Tuple(nd[c][i] for c in exogcols))
            for c in exogcols
                ismissing(row[c]) && throw(ArgumentError(
                    "new_data has a missing value in exogenous column :$c for " *
                    "$(spec.id)=$(repr(st.id)) at $t. Provide complete exogenous " *
                    "values for every series at every forecast step."))
            end
            rows[s] = row
        end
        out[k] = rows
    end
    return out
end

"""
    _panel_metric(m, fc, fitted, ids, y, ŷ) -> Real

Score one panel backtest fold. A metric that needs the training target (see
[`needs_ytrain`](@ref)) is computed per series, with that series' own training
values, and averaged with each series weighted by its scored points; any other
metric pools all scored rows, which weighs them the same way.
"""
function _panel_metric(m, fc::Forecaster, fitted::FittedForecaster, ids, y, ŷ)
    needs_ytrain(m) || return m(y, ŷ)
    history = Dict(st.id => st.y_history for st in getfield(fitted, :series))
    rows = Dict{eltype(ids),Vector{Int}}()
    for (j, id) in enumerate(ids)
        push!(get!(rows, id, Int[]), j)
    end
    total = 0.0
    # First-appearance order keeps the float sum reproducible.
    for id in unique(ids)
        r = rows[id]
        v = _in_series(fc, id) do
            m(y[r], ŷ[r]; y_train=history[id])
        end
        total += length(r) * v
    end
    return total / length(y)
end

"Each row's 1-based position among the rows of its id so far."
function _steps_by_id(ids::AbstractVector)
    seen = Dict{eltype(ids),Int}()
    return Int[seen[id] = get(seen, id, 0) + 1 for id in ids]
end

# One cutoff per series and fold: the last timestamp it trains on.
function _panel_cuts(s::FromStart, fc, groups, t_all, horizon, mh)
    grid = sort!(unique(t_all))
    ngrid = length(grid)
    s.initial > mh || throw(ArgumentError(
        "backtest initial=$(s.initial) must exceed the feature set's minimum history " *
        "($mh timestamps) so the first training window has at least one usable row. " *
        "For a panel, initial counts distinct timestamps, not rows. Pass " *
        "initial=$(mh + 1) or more, or reduce lags/windows."))
    origins = s.initial:s.step:(ngrid - horizon)
    isempty(origins) && throw(ArgumentError(
        "no complete backtest folds: the panel spans $ngrid distinct timestamps, " *
        "but the first fold needs initial + horizon = $(s.initial + horizon). " *
        "Provide more history or reduce initial/horizon."))
    keys_ = [key for (key, _, _) in groups]
    return [Dict(key => grid[o] for key in keys_) for o in origins]
end

function _panel_cuts(s::FromEnd, fc, groups, t_all, horizon, mh)
    longest = maximum(nrows(sub) for (_, sub, _) in groups)
    first = longest - horizon - (s.windows - 1) * s.step
    first > mh || throw(ArgumentError(
        "backtest windows=$(s.windows) with step=$(s.step) and horizon=$horizon " *
        "would train its first fold on at most $first rows of the longest series " *
        "($longest rows), but the feature set needs more than $mh. Reduce windows, " *
        "step or horizon, or provide longer series."))
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
    cuts = _panel_cuts(scheme, fc, groups, t_all, horizon, minhistory(fc.features))
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
        fitted = _fit_panel(fc, row_subset(tbl, train_rows), train_rows)
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
            _forecast_panel(fitted, hs, nd)
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
        for m in metrics
            push!(m_fold, k)
            push!(m_origin, origin)
            push!(m_metric, _metric_name(m))
            push!(m_value, Float64(_panel_metric(m, fc, fitted, scored, ytrue, yhat)))
        end
    end
    for m in metrics
        name = _metric_name(m)
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
