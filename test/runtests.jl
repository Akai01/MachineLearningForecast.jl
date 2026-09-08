using Test
using MachineLearningForecast
using Dates
using Statistics: mean, std

@testset "MachineLearningForecast" begin
    @test MachineLearningForecast isa Module
end
