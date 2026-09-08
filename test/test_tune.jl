# A trivial sequential tuning strategy defined AS IF it were third-party code:
# only exported names are used (TuningStrategy, ask, tell!). It proposes the
# middle candidate of its own space and stops after 3 asks.
module ThirdPartyTuning

using MachineLearningForecast: TuningStrategy
import MachineLearningForecast: ask, tell!

mutable struct Midpoint <: TuningStrategy
    space::Vector{<:NamedTuple}
    asks::Int
    told::Vector{Tuple{NamedTuple,Union{Missing,Float64}}}
end
Midpoint(space) = Midpoint(space, 0, Tuple{NamedTuple,Union{Missing,Float64}}[])

function ask(s::Midpoint)
    s.asks ≥ 3 && return nothing
    s.asks += 1
    return s.space[(length(s.space) + 1) ÷ 2]
end

tell!(s::Midpoint, candidate, score) = push!(s.told, (candidate, score))

end # module

@testset "tune" begin
    n = 120
    t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
    rng = StableRNG(42)
    y = 10.0 .+ 2 .* sin.(2π .* (1:n) ./ 7) .+ 0.3 .* randn(rng, n)
    df = (ds=t, y=y)
    ncand(r) = length(r.table.mean_score)
    base = Forecaster(DecisionTreeRegressor(max_depth=3);
                      features=FeatureSet(Lag(1), Lag(7)),
                      strategy=Recursive(), freq=Day(1))

    @testset "2×2 GridSearch with DecisionTree" begin
        grid = (model=[DecisionTreeRegressor(max_depth=2), DecisionTreeRegressor(max_depth=4)],
                features=[FeatureSet(Lag(1), Lag(7)),
                          FeatureSet(Lag(1), Lag(7), Calendar(:dayofweek))])
        result = tune(base, df; grid=grid, horizon=14, initial=90, step=14, metric=smape)
        @test result isa TuneResult
        @test ncand(result) == 4
        @test all(ismissing, result.table.error)
        # best score equals the min of the table, and best matches that row
        best_row = argmin(result.table.mean_score)
        @test result.table.mean_score[best_row] == minimum(result.table.mean_score)
        @test result.best.model == result.table.model[best_row]
        @test result.best.features == result.table.features[best_row]
        # best_fitted forecasts without error
        fcast = forecast(result.best_fitted, 7)
        @test length(fcast.y_hat) == 7 && all(isfinite, fcast.y_hat)
        # deterministic candidate ordering: first grid key varies fastest
        @test result.table.model[1:2] == grid.model
        @test result.table.features[1] == result.table.features[2] == grid.features[1]
        # show prints a top-candidates summary
        s = sprint(show, MIME"text/plain"(), result)
        @test occursin("4 candidates", s) && occursin("Top 4", s)
    end

    @testset "failed candidates are recorded, not fatal" begin
        # Lag(200) cannot be materialized on 120 rows → per-candidate failure
        grid = (features=[FeatureSet(Lag(1)), FeatureSet(Lag(200))],)
        result = tune(base, df; grid=grid, horizon=14, initial=90, metric=mae)
        @test ncand(result) == 2
        @test ismissing(result.table.error[1])
        @test !ismissing(result.table.error[2])
        @test occursin("history", result.table.error[2])   # the underlying ArgumentError text
        @test ismissing(result.table.mean_score[2])
        # failed candidate excluded from ranking
        @test result.best.features == FeatureSet(Lag(1))
        # all candidates failing raises an informative error
        bad = (features=[FeatureSet(Lag(200)), FeatureSet(Lag(300))],)
        err = try tune(base, df; grid=bad, horizon=14, initial=90) catch e; e end
        @test err isa ErrorException && occursin("all 2 tuning candidates failed", err.msg)
    end

    @testset "third-party sequential strategy drives the ask/tell loop" begin
        space = [(model=DecisionTreeRegressor(max_depth=d),) for d in 1:5]
        tuner = ThirdPartyTuning.Midpoint(space)
        result = tune(base, df; tuner=tuner, horizon=14, initial=90, metric=mae)
        # tune asked 3 times (the strategy's own termination), told 3 scores
        @test ncand(result) == 3
        @test length(tuner.told) == 3
        @test all(x -> x[1] == space[3], tuner.told)        # midpoint of 5 = index 3
        # The scores handed to tell! must be the SAME numbers tune reports, not
        # merely Float64s — this is the load-bearing half of the contract.
        @test [x[2] for x in tuner.told] ≈ collect(result.table.mean_score)
        @test result.best.model == space[3].model
        @test length(forecast(result.best_fitted, 5).y_hat) == 5
    end

    @testset "a failing candidate is told to the strategy as `missing`" begin
        # Documented contract: tell! receives `missing` when a candidate could
        # not be evaluated. Lag(200) needs more history than the 120-row series.
        bad = [(features=FeatureSet(Lag(200)),) for _ in 1:1]
        tuner = ThirdPartyTuning.Midpoint(bad)
        @test_throws ErrorException tune(base, df; tuner=tuner, horizon=14, initial=90)
        @test length(tuner.told) == 3
        @test all(x -> x[2] === missing, tuner.told)

        # Mixed good/bad through a grid: the search completes, the failure is
        # recorded with its message, and ranking ignores it.
        r = tune(base, df; grid=(features=[FeatureSet(Lag(1), Lag(7)),
                                           FeatureSet(Lag(200))],),
                 horizon=14, initial=90, metric=mae)
        @test ncand(r) == 2
        @test count(ismissing, r.table.mean_score) == 1
        @test count(!ismissing, r.table.error) == 1
        @test r.best.features == FeatureSet(Lag(1), Lag(7))
    end

    @testset "max_evals budget terminates a never-ending strategy" begin
        space = [(model=DecisionTreeRegressor(max_depth=d),) for d in 1:5]
        endless = ThirdPartyTuning.Midpoint(space)
        endless.asks = -10^9   # never reaches its own stop condition in this test
        result = tune(base, df; tuner=endless, max_evals=2, horizon=14, initial=90)
        @test ncand(result) == 2
    end

    @testset "RandomSearch" begin
        grid = (model=[DecisionTreeRegressor(max_depth=d) for d in 1:4],)
        r1 = tune(base, df; grid=grid, tuner=RandomSearch(3; rng=StableRNG(7)),
                  horizon=14, initial=90)
        @test ncand(r1) == 3
        # reproducible with the same seed
        r2 = tune(base, df; grid=grid, tuner=RandomSearch(3; rng=StableRNG(7)),
                  horizon=14, initial=90)
        @test r1.table.model == r2.table.model
        @test all(r1.table.mean_score .≈ r2.table.mean_score)
        @test_throws ArgumentError RandomSearch(0)

        # A RandomSearch that always returned the same candidate (or always the
        # first) would pass the checks above. Draw enough to make that visible:
        # over 24 draws from 4 values, seeing only one value has probability
        # 4·(1/4)^24, and a different seed must give a different sequence.
        many = tune(base, df; grid=grid, tuner=RandomSearch(24; rng=StableRNG(7)),
                    horizon=14, initial=90)
        depths = [m.max_depth for m in many.table.model]
        @test length(unique(depths)) > 1
        @test Set(depths) ⊆ Set(1:4)
        other = tune(base, df; grid=grid, tuner=RandomSearch(24; rng=StableRNG(99)),
                     horizon=14, initial=90)
        @test [m.max_depth for m in other.table.model] != depths
    end

    @testset "grid validation" begin
        @test_throws ArgumentError tune(base, df; horizon=14, initial=90)  # GridSearch needs grid
        @test_throws ArgumentError tune(base, df; grid=(bogus=[1, 2],), horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid=(model=Int[],), horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid="not a namedtuple", horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid=(model=[base.model],),
                                        horizon=14, initial=90, max_evals=0)
    end

    @testset "fit-count guard warns on large searches" begin
        big = (features=[FeatureSet(Lag(k)) for k in 1:3],
               strategy=fill(Direct(14), 9))
        # 27 candidates × 2 folds × 14 machines = 756 > 500
        @test_logs (:warn, r"756 model fits") match_mode=:any tune(
            base, df; grid=big, horizon=14, initial=90)

        # The guard must count only the fits that will ACTUALLY run: with
        # max_evals=1 just 1 candidate × 2 folds × 14 machines = 28 fits are
        # planned, which is under the threshold, so warning about 756 (and
        # advising "lower max_evals", already done) would be misinformation.
        @test_logs min_level=Logging.Warn tune(
            base, df; grid=big, max_evals=1, horizon=14, initial=90)
        # ...and a budget that still exceeds the threshold does warn, with the
        # budgeted count rather than the full grid's.
        @test_logs (:warn, r"560 model fits") match_mode=:any tune(
            base, df; grid=big, max_evals=20, horizon=14, initial=90)
    end
end
