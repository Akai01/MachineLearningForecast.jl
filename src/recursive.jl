function _fit(fc::Forecaster, ::Recursive, tbl::NamedTuple, rows)
    X, y, _ = _training_frame(fc, tbl, rows)
    return [_train(fc.model, X, y)]
end

function _forecast(f::FittedForecaster, ::Recursive, h::Integer, grid, exog_rows)
    spec = f.spec
    y_hist = copy(f.z_history)
    sizehint!(y_hist, length(y_hist) + h)
    colvecs, row, bufs = _prediction_row(f)
    st = only(getfield(f, :series))
    preds = Vector{Float64}(undef, h)
    trained = only(f.models)
    for s in 1:h
        exog_row = exog_rows === nothing ? nothing : exog_rows[s]
        ctx = _context(st, y_hist, grid[s], s, exog_row)
        _fill_row!(colvecs, bufs, spec.features, ctx)
        ŷ = Float64(only(_predict(trained, row)))
        preds[s] = ŷ
        push!(y_hist, ŷ)
    end
    ŷ = untransform(spec.transform, preds, f.transform_state)
    return _forecast_table(spec, grid, ŷ)
end
