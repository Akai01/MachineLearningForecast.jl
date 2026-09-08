# Panel (multi-series) forecasting.
#
# A panel forecaster fits ONE global model on the rows of every series pooled
# together, which is the point of panel forecasting: short or noisy series
# borrow strength from the rest. Features are always materialised WITHIN a
# series — a lag never reaches across a series boundary — and only the finished
# design-matrix rows are pooled.
#
# Series may be ragged: they need not share a start date, an end date or a
# length. Each series forecasts forward from its own last timestamp.

"Group row indices by the id column, preserving first-appearance order."
function _group_rows(ids::AbstractVector, id::Symbol)
    order = Any[]
    groups = Dict{Any,Vector{Int}}()
    for (i, v) in enumerate(ids)
        ismissing(v) && throw(ArgumentError(
            "id column :$id has a missing value at row $i. Every row must belong " *
            "to a series."))
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
    panel_groups(fc, tbl) -> Vector{Pair{Any,NamedTuple}}

Split a long-format panel table into `id => subtable` pairs in first-appearance
order, validating each series' time column independently. Rows of a series need
not be contiguous in the input, but within a series the timestamps must be
sorted, duplicate-free and gap-free with respect to `fc.freq`.
"""
function panel_groups(fc::Forecaster, tbl::NamedTuple)
    ids = require_column(tbl, fc.id, "id")
    require_column(tbl, fc.time, "time")
    require_column(tbl, fc.target, "target")
    order, groups = _group_rows(ids, fc.id)
    isempty(order) && throw(ArgumentError(
        "the panel is empty: no rows found in the id column :$(fc.id)."))
    out = Pair{Any,NamedTuple}[]
    for key in order
        sub = rowsubset(tbl, groups[key])
        t = sub[fc.time]
        if !issorted(t)
            perm = sortperm(t)
            sub = rowsubset(sub, perm)
            t = sub[fc.time]
        end
        try
            validate_time_column(t, fc.time, fc.freq)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("series $(fc.id)=$(repr(key)): " * err.msg))
        end
        push!(out, key => sub)
    end
    return out
end

# Build each series' one-step design matrix, reporting which series are too
# short rather than failing on the first one.
function _panel_frames(fc::Forecaster, groups)
    mh = minhistory(fc.features)
    frames = Tuple{Any,NamedTuple,Vector{Float64},NamedTuple}[]
    short = Tuple{Any,Int}[]
    for (key, sub) in groups
        n = nrows(sub)
        if n <= mh
            push!(short, (key, n))
            continue
        end
        X, y, _ = build_training_frame(fc.features, sub, fc.target, fc.time)
        push!(frames, (key, X, y, sub))
    end
    if isempty(frames)
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
    return frames
end

"Vertically concatenate columntables that share a schema."
function _vcat_frames(Xs::Vector{<:NamedTuple})
    length(Xs) == 1 && return only(Xs)
    names = keys(first(Xs))
    return NamedTuple{names}(ntuple(j -> reduce(vcat, (X[names[j]] for X in Xs)),
                                    length(names)))
end

function _fit_panel(fc::Forecaster, tbl::NamedTuple)
    groups = panel_groups(fc, tbl)
    frames = _panel_frames(fc, groups)
    machines = _panel_machines(fc, fc.strategy, frames)
    states = [SeriesState(key, target_vector(sub, fc.target),
                          sub[fc.time][end], sub[fc.time][1], nrows(sub))
              for (key, _, _, sub) in frames]
    return FittedForecaster(fc, machines, states, outputnames(fc.features))
end

# Recursive: pool every series' one-step frame and fit a single global model.
function _panel_machines(fc::Forecaster, ::Recursive, frames)
    X = _vcat_frames([f[2] for f in frames])
    y = reduce(vcat, (f[3] for f in frames))
    mach = MLJBase.machine(fc.model, X, y)
    MLJBase.fit!(mach; verbosity=0)
    return MLJBase.Machine[mach]
end

# Direct: the per-step target shift happens WITHIN each series, then the shifted
# rows are pooled. Anchoring matches the single-series path — target-history
# columns stay on the feature row, time/exogenous columns move to the target row.
function _panel_machines(fc::Forecaster, s::Direct, frames)
    istarget = targetcolumnmask(fc.features)
    machines = Vector{MLJBase.Machine}(undef, s.max_horizon)
    colnames = keys(frames[1][2])
    for i in 1:s.max_horizon
        Xs = NamedTuple[]
        ys = Vector{Float64}[]
        for (_, X, y, _) in frames
            nX = length(y)
            nX >= i || continue          # this series cannot reach step i
            push!(Xs, NamedTuple{colnames}(ntuple(
                c -> istarget[c] ? values(X)[c][1:(nX - i + 1)] : values(X)[c][i:nX],
                length(colnames))))
            push!(ys, y[i:nX])
        end
        isempty(Xs) && throw(ArgumentError(
            "not enough data for Direct($(s.max_horizon)): no series has enough " *
            "rows to train the step-$i model. Shorten max_horizon, provide longer " *
            "series, or reduce lags/windows."))
        mach = MLJBase.machine(deepcopy(fc.model), _vcat_frames(Xs), reduce(vcat, ys))
        MLJBase.fit!(mach; verbosity=0)
        machines[i] = mach
    end
    return machines
end

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

# One future row per series, assembled into a single design matrix so the model
# is called once per horizon step rather than once per series per step.
function _panel_batch(f::FittedForecaster, states, histories, step::Integer,
                      grids, exog_rows)
    spec = f.spec
    names = f.feature_names
    cols = [Vector{Float64}(undef, length(states)) for _ in names]
    scratch = [Vector{Float64}(undef, 1) for _ in names]
    for (k, st) in enumerate(states)
        n_next = st.n_train - 1 + step
        ex = exog_rows === nothing ? nothing : exog_rows[k][step]
        _fill_row!(scratch, spec.features, histories[k], grids[k][step], n_next, ex)
        for j in eachindex(cols)
            cols[j][k] = scratch[j][1]
        end
    end
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

function _forecast_panel(f::FittedForecaster, h::Integer, new_data)
    spec = f.spec
    states = getfield(f, :series)
    grids = [future_grid(st.t_start, spec.freq, st.n_train, h) for st in states]

    exogcols = exogenouscolumns(spec.features)
    exog_rows = nothing
    if !isempty(exogcols)
        new_data === nothing && throw(ArgumentError(
            "features contain Exogenous($(join(":" .* string.(exogcols), ", "))) but " *
            "forecast() was called without new_data. Pass a table with columns " *
            "(:$(spec.id), :$(spec.time), $(join(":" .* string.(exogcols), ", "))) " *
            "covering all $(length(states)) series over their $h forecast steps."))
        exog_rows = _panel_exogenous_rows(spec, exogcols, states, grids, new_data)
    elseif new_data !== nothing
        @warn "new_data was passed but the feature set has no Exogenous features; " *
              "it will be ignored."
    end

    preds = [Vector{Float64}(undef, h) for _ in states]
    histories = [copy(st.y_history) for st in states]
    _panel_predict!(preds, f, spec.strategy, states, histories, grids, exog_rows, h)

    nser = length(states)
    idcol = Vector{Any}(undef, nser * h)
    tcol = Vector{eltype(first(grids))}(undef, nser * h)
    ycol = Vector{Float64}(undef, nser * h)
    r = 1
    for k in 1:nser, s in 1:h
        idcol[r] = states[k].id
        tcol[r] = grids[k][s]
        ycol[r] = preds[k][s]
        r += 1
    end
    return NamedTuple{(spec.id, spec.time, :y_hat)}(
        (identity.(idcol), tcol, ycol))
end

# Recursive: predict every series for step s in one call, then feed each
# prediction back into that series' own history.
function _panel_predict!(preds, f, ::Recursive, states, histories, grids, exog_rows, h)
    mach = only(f.machines)
    for s in 1:h
        row = _panel_batch(f, states, histories, s, grids, exog_rows)
        ŷ = MLJBase.predict(mach, row)
        for k in eachindex(states)
            v = Float64(ŷ[k])
            preds[k][s] = v
            push!(histories[k], v)
        end
    end
    return preds
end

# Direct: step s uses machine s, with every series' target features conditioned
# on its own training-end history.
function _panel_predict!(preds, f, strat::Direct, states, histories, grids, exog_rows, h)
    h <= strat.max_horizon || throw(ArgumentError(
        "strategy=Direct($(strat.max_horizon)) was fit with max_horizon=" *
        "$(strat.max_horizon) but forecast(h=$h) was requested. Refit with " *
        "Direct($h) or use Recursive()."))
    for s in 1:h
        row = _panel_batch(f, states, histories, s, grids, exog_rows)
        ŷ = MLJBase.predict(f.machines[s], row)
        for k in eachindex(states)
            preds[k][s] = Float64(ŷ[k])
        end
    end
    return preds
end

# Align new_data to each series' forecast grid: exog_rows[k][s] is the NamedTuple
# of exogenous values for series k at step s. The join is on (id, timestamp).
function _panel_exogenous_rows(spec::Forecaster, exogcols, states, grids, new_data)
    nd = normalize_table(new_data)
    needed = [spec.id; spec.time; exogcols]
    absent = [c for c in needed if !haskey(nd, c)]
    isempty(absent) || throw(ArgumentError(
        "new_data is missing column$(length(absent) == 1 ? "" : "s") " *
        "$(join(":" .* string.(absent), ", ")); a panel forecast needs the id " *
        "column :$(spec.id), the time column :$(spec.time) and the exogenous " *
        "column$(length(exogcols) == 1 ? "" : "s") $(join(":" .* string.(exogcols), ", "))."))
    idv, tv = nd[spec.id], nd[spec.time]
    eltype(tv) == eltype(first(grids)) || throw(ArgumentError(
        "new_data's time column :$(spec.time) has element type $(eltype(tv)) but " *
        "the training time column is $(eltype(first(grids))). Convert it so " *
        "timestamps compare equal."))
    lookup = Dict{Tuple{Any,Any},Int}()
    for i in eachindex(tv)
        key = (idv[i], tv[i])
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
                    "$(spec.id)=$(repr(st.id)) at $t."))
            end
            rows[s] = row
        end
        out[k] = rows
    end
    return out
end

# ---------------------------------------------------------------------------
# Backtesting a panel
# ---------------------------------------------------------------------------

# Folds are cut on the GLOBAL timestamp grid, not on row counts: a panel's rows
# are spread across series, so "the first 300 rows" is not a point in time.
# Each fold trains on every row at or before the origin timestamp and scores by
# joining forecasts to actuals on (id, time), which handles ragged series.
function _backtest_panel(fc::Forecaster, tbl::NamedTuple, horizon, initial, step, metrics)
    groups = panel_groups(fc, tbl)                       # validates each series
    ids = tbl[fc.id]
    t_all = tbl[fc.time]
    y_all = target_vector(tbl, fc.target)
    grid = sort!(unique(t_all))
    ngrid = length(grid)
    mh = minhistory(fc.features)
    initial > mh || throw(ArgumentError(
        "backtest initial=$initial must exceed the feature set's minimum history " *
        "($mh timestamps) so the first training window has at least one usable row. " *
        "For a panel, initial counts distinct timestamps, not rows."))
    origins = initial:step:(ngrid - horizon)
    isempty(origins) && throw(ArgumentError(
        "no complete backtest folds: the panel spans $ngrid distinct timestamps, " *
        "but the first fold needs initial + horizon = $(initial + horizon). " *
        "Provide more history or reduce initial/horizon."))
    exogcols = exogenouscolumns(fc.features)

    # actuals indexed by (id, timestamp) so ragged series line up
    actual = Dict{Tuple{Any,Any},Float64}()
    for i in eachindex(y_all)
        actual[(ids[i], t_all[i])] = y_all[i]
    end

    T = eltype(t_all)
    I = eltype(ids)
    origin_col = T[]; step_col = Int[]; id_col = I[]
    time_col = T[]; y_col = Float64[]; yhat_col = Float64[]
    m_fold = Int[]; m_origin = Union{Missing,T}[]
    m_metric = Symbol[]; m_value = Float64[]

    for (k, o) in enumerate(origins)
        t_origin = grid[o]
        train_rows = findall(<=(t_origin), t_all)
        future_rows = findall(t -> t_origin < t <= grid[min(o + horizon, ngrid)], t_all)
        fitted = fit(fc, rowsubset(tbl, train_rows))
        nd = isempty(exogcols) ? nothing :
             rowsubset(tbl[Tuple([fc.id; fc.time; exogcols])], future_rows)
        fcast = forecast(fitted, horizon; new_data=nd)

        # keep only forecasts that have a matching actual
        fid, ft, fy = fcast[fc.id], fcast[fc.time], fcast.y_hat
        ytrue = Float64[]; yhat = Float64[]
        for j in eachindex(fy)
            a = get(actual, (fid[j], ft[j]), nothing)
            a === nothing && continue
            push!(ytrue, a); push!(yhat, fy[j])
            push!(origin_col, t_origin)
            push!(step_col, 1 + count(==(fid[j]), @view fid[1:j-1]))
            push!(id_col, fid[j]); push!(time_col, ft[j])
            push!(y_col, a); push!(yhat_col, fy[j])
        end
        isempty(ytrue) && throw(ArgumentError(
            "backtest fold $k (origin $t_origin) produced no forecast that lines up " *
            "with an actual. Check that the series share the freq=$(fc.freq) grid."))
        ytrain = y_all[train_rows]
        for m in metrics
            push!(m_fold, k); push!(m_origin, t_origin)
            push!(m_metric, _metric_name(m))
            push!(m_value, Float64(_apply_metric(m, ytrue, yhat, ytrain)))
        end
    end
    for m in metrics
        name = _metric_name(m)
        vals = [m_value[i] for i in eachindex(m_value) if m_metric[i] == name && m_fold[i] > 0]
        push!(m_fold, 0); push!(m_origin, missing)
        push!(m_metric, name); push!(m_value, Statistics.mean(vals))
    end
    folds = NamedTuple{(:origin, :step, fc.id, fc.time, :y, :y_hat)}(
        (origin_col, step_col, id_col, time_col, y_col, yhat_col))
    return BacktestResult(folds, (fold=m_fold, origin=m_origin,
                                  metric=m_metric, value=m_value))
end
