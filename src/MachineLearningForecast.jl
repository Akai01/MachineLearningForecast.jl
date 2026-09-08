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

include("utils.jl")

end # module
