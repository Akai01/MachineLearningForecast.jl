"""
    MachineLearningForecast

A generic, composable time-series forecasting package built on the MLJ model
interface: any MLJ `Deterministic` regressor is a valid base model, and a
`Probabilistic` one forecasts its predicted mean.

- Declarative feature engineering: [`FeatureSet`](@ref), [`Lag`](@ref),
  [`RollingMean`](@ref), [`RollingStd`](@ref), [`RollingMin`](@ref),
  [`RollingMax`](@ref), [`Diff`](@ref), [`Calendar`](@ref), [`Fourier`](@ref),
  [`Exogenous`](@ref), [`CustomFeature`](@ref).
- Pluggable strategies: [`Recursive`](@ref), [`Direct`](@ref).
- Target transforms, fitted per series and undone on the forecasts:
  [`NoTransform`](@ref), [`MeanScale`](@ref), [`LastScale`](@ref),
  [`Deseasonalize`](@ref), [`LogRatio`](@ref), [`Difference`](@ref),
  [`TransformChain`](@ref); your own subtype [`TargetTransform`](@ref) implements
  [`fit_transform`](@ref) and [`untransform`](@ref) (public, not exported:
  `import MachineLearningForecast: fit_transform, untransform`).
- Panel (multi-series) forecasting: set `id` on [`Forecaster`](@ref) to fit
  one global model across many series ([`nseries`](@ref), [`SeriesState`](@ref)).
- [`fit`](@ref) / [`forecast`](@ref) as a functional pair; [`backtest`](@ref)
  for expanding-window cross-validation; [`tune`](@ref) for pipeline tuning
  with an extensible ask!/tell! [`TuningStrategy`](@ref) interface.
- Metrics: [`mae`](@ref), [`rmse`](@ref), [`mape`](@ref), [`smape`](@ref),
  [`mase`](@ref), [`rmsse`](@ref).
"""
module MachineLearningForecast

using Dates
# Loading it gives models MLJModelInterface's full interface.
import MLJBase
import MLJModelInterface
import PrecompileTools
import Random
import Statistics
import Tables

export Forecaster, FittedForecaster, fit, forecast, backtest, BacktestResult,
       FeatureSet, Lag, RollingMean, RollingStd, RollingMin, RollingMax,
       Diff, Calendar, Fourier, Exogenous, CustomFeature,
       TargetFeature, TimeFeature, ExogenousFeature,
       ForecastStrategy, Recursive, Direct,
       mae, rmse, mape, smape, mase, rmsse,
       tune, TuneResult, TuningStrategy, GridSearch, RandomSearch,
       SeriesState, nseries,
       TargetTransform, NoTransform, MeanScale, LastScale,
       Deseasonalize, LogRatio, Difference, TransformChain

include("utils.jl")
include("features.jl")
include("featureset.jl")
include("strategies.jl")
include("transforms.jl")
include("metrics.jl")
include("forecaster.jl")
include("recursive.jl")
include("direct.jl")
include("backtest.jl")
include("panel.jl")
include("tune.jl")
include("precompile.jl")

end
