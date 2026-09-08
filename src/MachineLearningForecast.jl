"""
    MachineLearningForecast

A generic, composable time-series forecasting package built on the MLJ model
interface: any MLJ `Deterministic` regressor is a valid base model.

- Declarative feature engineering: [`FeatureSet`](@ref), [`Lag`](@ref),
  [`RollingMean`](@ref), [`RollingStd`](@ref), [`RollingMin`](@ref),
  [`RollingMax`](@ref), [`Diff`](@ref), [`Calendar`](@ref), [`Fourier`](@ref),
  [`Exogenous`](@ref), [`CustomFeature`](@ref).
- Pluggable strategies: [`Recursive`](@ref), [`Direct`](@ref).
- [`fit`](@ref) / [`forecast`](@ref) as a functional pair; [`backtest`](@ref)
  for expanding-window cross-validation; [`tune`](@ref) for pipeline tuning
  with an extensible ask/tell [`TuningStrategy`](@ref) interface.
- Metrics: [`mae`](@ref), [`rmse`](@ref), [`mape`](@ref), [`smape`](@ref),
  [`mase`](@ref).
"""
module MachineLearningForecast

using Dates
import MLJBase
import MLJModelInterface
import Random
import Statistics
import Tables

export Forecaster, FittedForecaster, fit, forecast, backtest, BacktestResult,
       FeatureSet, Lag, RollingMean, RollingStd, RollingMin, RollingMax,
       Diff, Calendar, Fourier, Exogenous, CustomFeature,
       Recursive, Direct,
       mae, rmse, mape, smape, mase,
       tune, TuneResult, TuningStrategy, GridSearch, RandomSearch, ask, tell!

include("utils.jl")
include("features.jl")
include("featureset.jl")
include("strategies.jl")
include("metrics.jl")
include("forecaster.jl")
include("recursive.jl")
include("direct.jl")
include("backtest.jl")
include("tune.jl")

end # module
