using Test
using MachineLearningForecast
using Dates
using Statistics: mean, std

include("testmodels.jl")
using .TestModels

@testset "MachineLearningForecast" begin
    include("test_features.jl")
    include("test_strategies.jl")
    include("test_forecaster.jl")
    include("test_backtest.jl")
end
