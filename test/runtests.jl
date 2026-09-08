using Test
using Logging
using MachineLearningForecast
using Dates
using Tables
using Statistics: mean, std
using StableRNGs: StableRNG
using EvoTrees: EvoTreeRegressor
using MLJDecisionTreeInterface: DecisionTreeRegressor, RandomForestRegressor
import Aqua

include("testmodels.jl")
using .TestModels

@testset "MachineLearningForecast" begin
    include("test_features.jl")
    include("test_strategies.jl")
    include("test_forecaster.jl")
    include("test_backtest.jl")
    include("test_tune.jl")
    include("test_validation.jl")
end
