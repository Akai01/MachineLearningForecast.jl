function _fit(fc::Forecaster, ::Recursive, tbl::NamedTuple)
    X, y, _ = _training_frame(fc, tbl)
    return _fitted(fc, tbl, [_train(fc.model, X, y)])
end

function _forecast(f::FittedForecaster, ::Recursive, h::Integer, grid, exog_rows)
    spec = f.spec
    y_hist = copy(f.y_history)
    sizehint!(y_hist, length(y_hist) + h)
    colvecs, row = _prediction_row(f)
    preds = Vector{Float64}(undef, h)
    trained = only(f.models)
    for s in 1:h
        # Fourier index continues the 0-based training index.
        n_next = f.n_train - 1 + s
        exog_row = exog_rows === nothing ? nothing : exog_rows[s]
        _fill_row!(colvecs, spec.features, y_hist, grid[s], n_next, exog_row)
        ŷ = Float64(only(_predict(trained, row)))
        preds[s] = ŷ
        push!(y_hist, ŷ)
    end
    return _forecast_table(spec, grid, preds)
end
