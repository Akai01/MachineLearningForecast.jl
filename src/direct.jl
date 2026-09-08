# Direct strategy: one machine per horizon step. Machine i predicts y_{t+i}
# from the information available at time t.
#
# Column alignment matters here, and the two feature groups are anchored
# differently — this is what `forecast` feeds, so `fit` must match it exactly:
#
#   * TargetFeature columns (lags, rolling stats, diffs, custom) describe the
#     target's own HISTORY, so they stay anchored to the feature row j: at
#     forecast time they are computed once from the training-end history.
#   * TimeFeature and ExogenousFeature columns (calendar, Fourier, covariates)
#     are KNOWN for the future, so they are anchored to the TARGET row j+i-1:
#     at forecast time step `s` they are taken at the forecast timestamp.
#
# Anchoring both groups to row j at fit time (while forecasting with the target
# row's time/exogenous values) is a (step-1) train/serve skew that silently
# destroys every step >= 2 — see the "Direct target alignment" tests.

function _fit(fc::Forecaster, s::Direct, tbl::NamedTuple)
    X, y, _ = _training_frame(fc, tbl)
    nX = length(y)
    mh = minhistory(fc.features)
    nX ≥ s.max_horizon || throw(ArgumentError(
        "not enough data for Direct($(s.max_horizon)): after dropping the feature " *
        "set's $mh history rows, only $nX training rows remain, but the target " *
        "must be shifted up to $(s.max_horizon - 1) steps. Provide at least " *
        "$(mh + s.max_horizon) rows, reduce max_horizon, or reduce lags/windows."))
    machines = Vector{MLJBase.Machine}(undef, s.max_horizon)
    istarget = targetcolumnmask(fc.features)
    colnames = keys(X)
    for i in 1:s.max_horizon
        # target-history columns from row j; time/exogenous columns from the
        # target row j+i-1, matching what `_forecast` assembles per step.
        Xi = NamedTuple{colnames}(ntuple(
            c -> istarget[c] ? values(X)[c][1:(nX - i + 1)] : values(X)[c][i:nX],
            length(colnames)))
        yi = y[i:nX]
        mach = MLJBase.machine(deepcopy(fc.model), Xi, yi)
        MLJBase.fit!(mach; verbosity=0)
        machines[i] = mach
    end
    return _fitted(fc, tbl, machines)
end

function _forecast(f::FittedForecaster, s::Direct, h::Integer, grid, exog_rows)
    h ≤ s.max_horizon || throw(ArgumentError(
        "strategy=Direct($(s.max_horizon)) was fit with max_horizon=" *
        "$(s.max_horizon) but forecast(h=$h) was requested. Refit with " *
        "Direct($h) or use Recursive()."))
    spec = f.spec
    colvecs, row = _prediction_row(f)
    preds = Vector{Float64}(undef, h)
    for step in 1:h
        # Target-history features are conditioned on the training-end history
        # for every step (the v1 simplification); time and exogenous features
        # are taken at this step's target timestamp, exactly as machine `step`
        # was trained (Fourier index continues the training index).
        n_next = f.n_train - 1 + step
        exog_row = exog_rows === nothing ? nothing : exog_rows[step]
        _fill_row!(colvecs, spec.features, f.y_history, grid[step], n_next, exog_row)
        preds[step] = Float64(only(MLJBase.predict(f.machines[step], row)))
    end
    return _forecast_table(spec, grid, preds)
end
