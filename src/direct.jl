# See test "Direct anchors time/exogenous columns".
function _fit(fc::Forecaster, s::Direct, tbl::NamedTuple, rows)
    X, y, _ = _training_frame(fc, tbl, rows)
    nX = length(y)
    mh = minhistory(fc.features)
    nX ≥ s.max_horizon || throw(ArgumentError(
        "not enough data for Direct($(s.max_horizon)): after dropping the feature " *
        "set's $mh history rows, only $nX training rows remain, but the target " *
        "must be shifted up to $(s.max_horizon - 1) steps. Provide at least " *
        "$(mh + s.max_horizon) rows, reduce max_horizon, or reduce lags/windows."))
    anch = anchors(fc.features)
    colnames = keys(X)
    models = map(1:s.max_horizon) do i
        # AtTarget columns use the target row, as forecast does.
        Xi = NamedTuple{colnames}(ntuple(
            c -> _shifted(anch[c], values(X)[c], i, nX), length(colnames)))
        yi = y[i:nX]
        _train(fc.model, Xi, yi)
    end
    return models
end

function _forecast(f::FittedForecaster, s::Direct, h::Integer, grid, exog_rows)
    h ≤ s.max_horizon || throw(ArgumentError(
        "strategy=Direct($(s.max_horizon)) was fit with max_horizon=" *
        "$(s.max_horizon) but forecast(h=$h) was requested. Refit with " *
        "Direct($h) or use Recursive()."))
    spec = f.spec
    colvecs, row, bufs = _prediction_row(f)
    st = only(getfield(f, :series))
    preds = Vector{Float64}(undef, h)
    for step in 1:h
        exog_row = exog_rows === nothing ? nothing : exog_rows[step]
        ctx = _context(st, st.z_history, grid[step], step, exog_row)
        _fill_row!(colvecs, bufs, spec.features, ctx)
        preds[step] = Float64(only(_predict(f.models[step], row)))
    end
    ŷ = untransform(spec.transform, preds, f.transform_state)
    return _forecast_table(spec, grid, ŷ)
end
