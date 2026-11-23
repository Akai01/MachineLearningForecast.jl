using Test
using Logging
using MachineLearningForecast
using Dates
using Tables
using Statistics: mean, std
using Random: shuffle
using StableRNGs: StableRNG
using EvoTrees: EvoTreeRegressor, EvoTreeGaussian
using MLJDecisionTreeInterface: DecisionTreeRegressor, RandomForestRegressor
import Aqua

include("testmodels.jl")
using .TestModels
using .FeatureHelpers

@testset "MachineLearningForecast" begin
    include("test_features.jl")
    include("test_strategies.jl")
    include("test_forecaster.jl")
    include("test_metrics.jl")
    include("test_backtest.jl")
    include("test_tune.jl")
    include("test_validation.jl")
    include("test_panel.jl")
    include("test_transforms.jl")
    include("test_integration.jl")
    include("test_leakage.jl")

    @testset "Aqua quality checks" begin
        Aqua.test_all(MachineLearningForecast; ambiguities=false)
        Aqua.test_ambiguities(MachineLearningForecast)
    end
end
