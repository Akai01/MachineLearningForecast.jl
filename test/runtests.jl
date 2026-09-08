using Test
using MachineLearningForecast
using Dates
using Statistics: mean, std

@testset "MachineLearningForecast" begin
    include("test_features.jl")
end
