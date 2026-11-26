"""
    SeasonalNaive(period)

An MLJ `Deterministic` model that needs no training: it predicts its `y_lag_<period>`
input, the target one season back. With `FeatureSet(Lag(period))` and [`Recursive`](@ref)
a [`Forecaster`](@ref) repeats the last observed value at each step's position in the
season, the seasonal naive forecast, at every horizon: the fed-back forecasts supply the
lags beyond one season. This holds without a target transform or with a scaling one such
as [`MeanScale`](@ref); `Difference` or `LogRatio` change the forecasts. Use it as a
baseline in [`backtest`](@ref), or as the centre of [`ConformalSeasonalPool`](@ref).
Under [`Direct`](@ref), `Lag(period)` is not in phase with the target at every step, so
the forecasts are not seasonal naive: use `Recursive()`.

# Example
```julia
fc = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1))
summarize(backtest(fc, df; horizon=12, windows=3, metrics=(mae, smape)))
```
"""
mutable struct SeasonalNaive <: MLJModelInterface.Deterministic
    period::Int
    function SeasonalNaive(period)
        fix = "Pass the season length in steps, e.g. SeasonalNaive(12) for monthly data."
        _whole(period, "SeasonalNaive period", fix)
        period ≥ 1 || throw(ArgumentError(
            "SeasonalNaive needs a period ≥ 1, got $period. $fix"))
        return new(Int(period))
    end
end

MLJModelInterface.fit(::SeasonalNaive, verbosity, X, y) = (nothing, nothing, NamedTuple())

function MLJModelInterface.predict(m::SeasonalNaive, fitresult, Xnew)
    col = Symbol("y_lag_", m.period)
    cols = Tables.columns(Xnew)
    col in Tables.columnnames(cols) || throw(ArgumentError(
        "SeasonalNaive($(m.period)) predicts the feature column :$col, which the " *
        "feature set does not produce. Add Lag($(m.period)) to the FeatureSet, e.g. " *
        "FeatureSet(Lag($(m.period)))."))
    return Float64.(Tables.getcolumn(cols, col))
end

MLJModelInterface.input_scitype(::Type{SeasonalNaive}) =
    MLJModelInterface.Table(MLJModelInterface.Continuous)
MLJModelInterface.target_scitype(::Type{SeasonalNaive}) =
    AbstractVector{MLJModelInterface.Continuous}
