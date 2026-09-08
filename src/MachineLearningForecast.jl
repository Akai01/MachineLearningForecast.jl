"""
    MachineLearningForecast

Generic, composable time-series forecasting built on the MLJ model interface:
any MLJ `Deterministic` regressor can be used as the base model.
"""
module MachineLearningForecast

using Dates
import MLJBase
import MLJModelInterface
import Random
import Statistics
import Tables

export FeatureSet, Lag, RollingMean, RollingStd, RollingMin, RollingMax, Diff,
       Calendar, Fourier, Exogenous, CustomFeature, Recursive, Direct

include("utils.jl")
include("features.jl")
include("featureset.jl")
include("strategies.jl")

end # module
