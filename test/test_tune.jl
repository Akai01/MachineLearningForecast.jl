# Uses only exported names, as third-party code would.
module ThirdPartyTuning

using MachineLearningForecast: TuningStrategy
import MachineLearningForecast: ask, tell!

mutable struct Midpoint{T<:NamedTuple} <: TuningStrategy
    space::Vector{T}
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

"Always proposes `out`, to check what tune does with each kind of answer."
struct Fixed{T} <: TuningStrategy
    out::T
end
ask(s::Fixed) = s.out
tell!(::Fixed, candidate, score) = nothing

end

@testset "tune" begin
    n = 120
    t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
    rng = StableRNG(42)
    y = 10.0 .+ 2 .* sin.(2π .* (1:n) ./ 7) .+ 0.3 .* randn(rng, n)
    df = (ds=t, y=y)
    ncand(r) = length(r.table.mean_score)
    base = Forecaster(DecisionTreeRegressor(max_depth=3, rng=StableRNG(1));
                      features=FeatureSet(Lag(1), Lag(7)),
                      strategy=Recursive(), freq=Day(1))

    @testset "2×2 GridSearch with DecisionTree" begin
        grid = (model=[DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                       DecisionTreeRegressor(max_depth=4, rng=StableRNG(1))],
                features=[FeatureSet(Lag(1), Lag(7)),
                          FeatureSet(Lag(1), Lag(7), Calendar(:dayofweek))])
        result = tune(base, df; grid=grid, horizon=14, initial=90, step=14, metric=smape)
        @test result isa TuneResult
        @test ncand(result) == 4
        @test all(ismissing, result.table.error)
        best_row = argmin(result.table.mean_score)
        @test result.table.mean_score[best_row] == minimum(result.table.mean_score)
        @test result.best.model == result.table.model[best_row]
        @test result.best.features == result.table.features[best_row]
        fcast = forecast(result.best_fitted, 7)
        @test length(fcast.y_hat) == 7 && all(isfinite, fcast.y_hat)
        # Deterministic order: the first grid key varies fastest.
        @test result.table.model[1:2] == grid.model
        @test result.table.features[1] == result.table.features[2] == grid.features[1]
        s = sprint(show, MIME"text/plain"(), result)
        @test occursin("4 candidates", s) && occursin("Top 4", s)
    end

    @testset "failed candidates are recorded, not fatal" begin
        # Lag(200) needs more history than initial=90 allows.
        grid = (features=[FeatureSet(Lag(1)), FeatureSet(Lag(200))],)
        result = tune(base, df; grid=grid, horizon=14, initial=90, metric=mae)
        @test ncand(result) == 2
        @test ismissing(result.table.error[1])
        @test !ismissing(result.table.error[2])
        # the underlying ArgumentError text
        @test occursin("history", result.table.error[2])
        @test ismissing(result.table.mean_score[2])
        @test result.best.features == FeatureSet(Lag(1))
        bad = (features=[FeatureSet(Lag(200)), FeatureSet(Lag(300))],)
        err = try tune(base, df; grid=bad, horizon=14, initial=90) catch e; e end
        @test err isa ErrorException && occursin("all 2 tuning candidates failed", err.msg)
    end

    @testset "a non-finite score fails the candidate, never wins" begin
        for good in (DecisionTreeRegressor(max_depth=3, rng=StableRNG(1)),
                     EvoTreeRegressor(nrounds=10))
            r = tune(base, df; grid=(model=[TestModels.NaNModel(), good],),
                     horizon=14, initial=90, metric=mae)
            @test ismissing(r.table.mean_score[1]) && ismissing(r.table.std_score[1])
            @test occursin("non-finite mean score (NaN)", r.table.error[1])
            @test r.best.model == good
            @test all(isfinite, forecast(r.best_fitted, 7).y_hat)
            s = sprint(show, MIME"text/plain"(), r)
            @test occursin("(1 failed)", s) && occursin("Top 1", s)
        end
        tuner = ThirdPartyTuning.Midpoint([(model=TestModels.NaNModel(),)])
        @test_throws "all 3 tuning candidates failed" tune(base, df; tuner=tuner,
                                                           horizon=14, initial=90)
        @test length(tuner.told) == 3 && all(ismissing(sc) for (_, sc) in tuner.told)
    end

    @testset "third-party sequential strategy drives the ask/tell loop" begin
        space = [(model=DecisionTreeRegressor(max_depth=d, rng=StableRNG(1)),) for d in 1:5]
        tuner = ThirdPartyTuning.Midpoint(space)
        result = tune(base, df; tuner=tuner, horizon=14, initial=90, metric=mae)
        @test ncand(result) == 3
        @test length(tuner.told) == 3
        @test all(x -> x[1] == space[3], tuner.told)        # midpoint of 5 = index 3
        @test [x[2] for x in tuner.told] ≈ collect(result.table.mean_score)
        @test result.best.model == space[3].model
        @test length(forecast(result.best_fitted, 5).y_hat) == 5
    end

    @testset "a failing candidate is told to the strategy as `missing`" begin
        # Lag(200) needs more history than initial=90 allows.
        bad = [(features=FeatureSet(Lag(200)),) for _ in 1:1]
        tuner = ThirdPartyTuning.Midpoint(bad)
        @test_throws ErrorException tune(base, df; tuner=tuner, horizon=14, initial=90)
        @test length(tuner.told) == 3
        @test all(x -> x[2] === missing, tuner.told)

        # A grid with one failure completes and ranks the rest.
        r = tune(base, df; grid=(features=[FeatureSet(Lag(1), Lag(7)),
                                           FeatureSet(Lag(200))],),
                 horizon=14, initial=90, metric=mae)
        @test ncand(r) == 2
        @test count(ismissing, r.table.mean_score) == 1
        @test count(!ismissing, r.table.error) == 1
        @test r.best.features == FeatureSet(Lag(1), Lag(7))
    end

    @testset "max_evals budget terminates a never-ending strategy" begin
        space = [(model=DecisionTreeRegressor(max_depth=d, rng=StableRNG(1)),) for d in 1:5]
        endless = ThirdPartyTuning.Midpoint(space)
        endless.asks = -10^9   # never reaches its own stop condition in this test
        result = tune(base, df; tuner=endless, max_evals=2, horizon=14, initial=90)
        @test ncand(result) == 2
    end

    @testset "RandomSearch" begin
        grid = (model=[DecisionTreeRegressor(max_depth=d, rng=StableRNG(1)) for d in 1:4],)
        r1 = tune(base, df; grid=grid, tuner=RandomSearch(3; rng=StableRNG(7)),
                  horizon=14, initial=90)
        @test ncand(r1) == 3
        r2 = tune(base, df; grid=grid, tuner=RandomSearch(3; rng=StableRNG(7)),
                  horizon=14, initial=90)
        @test r1.table.model == r2.table.model
        @test all(r1.table.mean_score .≈ r2.table.mean_score)
        @test_throws ArgumentError RandomSearch(0)
        @test_throws "RandomSearch n must be ≥ 1, got 0. Pass the number of candidates " *
                     "to draw, e.g. RandomSearch(10)." RandomSearch(0)

        # 24 draws from 4 values: all-equal has p = 4·4^-24.
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
        @test_throws ArgumentError tune(base, df; horizon=14, initial=90)
        @test_throws "GridSearch() requires the grid keyword: pass grid=(model=[...], " *
                     "features=[...], ...) to tune." tune(base, df; horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid=(bogus=[1, 2],), horizon=14,
                                        initial=90)
        @test_throws "grid has unknown key :bogus; valid keys are :model, :features" tune(
            base, df; grid=(bogus=[1, 2],), horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid=(model=Int[],), horizon=14,
                                        initial=90)
        @test_throws "grid key :model must map to a nonempty vector or tuple of " *
                     "candidate values, got Int64[]." tune(base, df; grid=(model=Int[],),
                                                           horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid="not a namedtuple", horizon=14,
                                        initial=90)
        @test_throws "candidate value lists, got String. Use e.g. grid=(model=[m1, m2]" (
            tune(base, df; grid="not a namedtuple", horizon=14, initial=90))
        @test_throws "got Vector{FeatureSet}" tune(base, df; grid=[FeatureSet(Lag(1))],
                                                   horizon=14, initial=90)
        @test_throws ArgumentError tune(base, df; grid=(model=[base.model],),
                                        horizon=14, initial=90, max_evals=0)
        @test_throws "max_evals must be ≥ 1 (or nothing for no budget), got 0." tune(
            base, df; grid=(model=[base.model],), horizon=14, initial=90, max_evals=0)
    end

    @testset "ask must answer with a NamedTuple of known fields or nothing" begin
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
            fcm = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            run(out) = tune(fcm, df; tuner=ThirdPartyTuning.Fixed(out), max_evals=2,
                            horizon=14, initial=90)
            @test_throws ArgumentError run(42)
            @test_throws "ask(Fixed) returned 42; the ask/tell contract requires a " *
                         "NamedTuple of Forecaster field overrides, or nothing to stop." (
                run(42))
            @test_throws ArgumentError run(nothing)
            @test_throws "the tuning strategy proposed no candidates (Fixed): ask " *
                         "returned nothing on its first call, so there is nothing to " *
                         "tune. Make ask return at least one NamedTuple of Forecaster " *
                         "field overrides before nothing." run(nothing)
            # reconstruct rejects it inside the candidate's try
            @test_throws "all 2 tuning candidates failed" run((bogus=1,))
            @test_throws "ArgumentError: cannot reconstruct Forecaster with unknown " *
                         "field :bogus; valid fields are :model, :features" run(
                (bogus=1,))
        end
    end

    @testset "a planned strategy of the wrong type is rejected before any fit" begin
        # StableRNG(1) draws the second grid value, then the first.
        cases = (((strategy=[:recursive, :direct],), "candidate 1 is :recursive",
                  "candidate 1 is :direct"),
                 ((strategy=[Recursive(), "direct"],), "candidate 2 is \"direct\"",
                  "candidate 1 is \"direct\""))
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5)),
            (grid, gphrase, rphrase) in cases
            fcm = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            for (tuner, phrase) in ((GridSearch, gphrase),
                                    (() -> RandomSearch(2; rng=StableRNG(1)), rphrase))
                run() = tune(fcm, df; grid=grid, tuner=tuner(), horizon=14, initial=90)
                @test_throws ArgumentError run()
                @test_throws "grid key :strategy must hold values such as Recursive() " *
                             "or Direct(28), but $phrase." run()
            end
        end
        @test_throws "such as Recursive() or Direct(28)" tune(
            base, df; grid=(strategy=[:direct],), horizon=14, initial=90)
    end

    @testset "only the planned candidates' strategies are checked" begin
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
            fcm = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            run(grid; kw...) = tune(fcm, df; grid=grid, horizon=14, initial=90,
                                    metric=mae, kw...)
            ref = run((strategy=[Recursive()],))
            mixed = (strategy=[Recursive(), :direct],)
            # StableRNG(3) draws only the first grid value.
            for r in (run(mixed; max_evals=1),
                      run(mixed; tuner=RandomSearch(1; rng=StableRNG(3))))
                @test ncand(r) == 1 && all(ismissing, r.table.error)
                @test r.table.strategy == [Recursive()]
                @test r.table.mean_score == ref.table.mean_score
            end
        end
    end

    @testset "a features value that is not a FeatureSet fails only its candidate" begin
        msg = "ArgumentError: tune candidate key :features must be a FeatureSet such as " *
              "FeatureSet(Lag(1), Lag(7)), got Lag(1). Wrap the features in " *
              "FeatureSet(...)."
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
            fcm = Forecaster(m; features=FeatureSet(Lag(7)), freq=Day(1))
            run(grid; kw...) = tune(fcm, df; grid=grid, horizon=14, initial=90,
                                    metric=mae, kw...)
            ref = run((features=[FeatureSet(Lag(1))],))
            r = run((features=[Lag(1), FeatureSet(Lag(1))],))
            @test r isa TuneResult && ncand(r) == 2
            @test ismissing(r.table.mean_score[1]) && r.table.error[1] == msg
            @test ismissing(r.table.error[2])
            @test r.table.mean_score[2] == only(ref.table.mean_score)
            @test r.best.features == FeatureSet(Lag(1))
            # StableRNG(3) draws only the first grid value.
            rs = run((features=[FeatureSet(Lag(1)), Lag(1)],);
                     tuner=RandomSearch(1; rng=StableRNG(3)))
            @test rs.table.mean_score == ref.table.mean_score
            @test_throws ErrorException run((features=[Lag(1)],))
            @test_throws msg run((features=[Lag(1)],))
        end
        @test_throws "such as FeatureSet(Lag(1), Lag(7))" tune(
            base, df; grid=(features=[Lag(1)],), horizon=14, initial=90)
    end

    @testset "metric must be one function, checked before any fit" begin
        grid = (model=[DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                       EvoTreeRegressor(nrounds=5)],)
        for bad in ((mae, rmse), [mae], :mae, "smape")
            @test_throws ArgumentError tune(base, df; grid=grid, horizon=14, initial=90,
                                            metric=bad)
            @test_throws "got $(repr(bad)). Pass e.g. metric=smape" tune(
                base, df; grid=grid, horizon=14, initial=90, metric=bad)
        end
        r = tune(base, df; grid=grid, horizon=14, initial=90, metric=(a, b) -> mae(a, b))
        @test r.table.mean_score == tune(base, df; grid=grid, horizon=14, initial=90,
                                         metric=mae).table.mean_score
    end

    @testset "fit-count guard warns on large searches" begin
        big = (features=[FeatureSet(Lag(k)) for k in 1:3],
               strategy=fill(Direct(14), 9))
        # 27 candidates × 2 folds × 14 machines = 756 > 500
        @test_logs (:warn, r"756 model fits") match_mode=:any tune(
            base, df; grid=big, horizon=14, initial=90)

        # max_evals=1 plans only 28 fits: no warning.
        @test_logs min_level=Logging.Warn tune(
            base, df; grid=big, max_evals=1, horizon=14, initial=90)
        # A budget over the threshold warns with its own count.
        @test_logs (:warn, r"560 model fits") match_mode=:any tune(
            base, df; grid=big, max_evals=20, horizon=14, initial=90)
    end

    @testset "edge cases: horizon 1 and bad data" begin
        grid = (model=[DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                       EvoTreeRegressor(nrounds=5)],)
        r = tune(base, df; grid=grid, horizon=1, initial=110, metric=mae)
        @test ncand(r) == 2 && all(!ismissing, r.table.mean_score)
        for (k, m) in enumerate(grid.model)
            bt = backtest(Forecaster(m; features=base.features, freq=Day(1)), df;
                          horizon=1, initial=110, metrics=(mae,))
            @test r.table.mean_score[k] == only(bt.metrics.value[bt.metrics.fold .== 0])
        end
        @test length(forecast(r.best_fitted, 1).y_hat) == 1
        @test_throws TypeError tune(base, df; grid=grid, horizon=1.5, initial=110)
        @test_throws MethodError RandomSearch(1.5)
        @test_throws ArgumentError tune(base, 42; grid=grid, horizon=1, initial=5)
        @test_throws "expected a Tables.jl-compatible table" tune(base, 42; grid=grid,
                                                                  horizon=1, initial=5)
        ym = Vector{Union{Missing,Float64}}(y); ym[55] = missing
        yn = copy(y); yn[55] = NaN
        bad = [(ds=Date[], y=Float64[]) => "time column :ds is empty",
               (ds=t[1:3], y=y[1:3]) => "no complete backtest folds: data has 3 rows",
               (ds=t, y=ym) => "contains missing values (first at row 55)",
               (ds=t, y=yn) => "the non-finite value NaN at row 55",
               (ds=[t[1:30]; t[32:end]], y=y[1:119]) => "has 1 gap for freq=1 day",
               (ds=t, y=string.(y)) => "target column :y has element type String",
               (ds=string.(t), y=y) => "has element type String, which does not support"]
        # Each candidate fails on the data, so tune raises.
        for (data, msg) in bad
            run() = tune(base, data; grid=grid, horizon=1, initial=8)
            @test_throws ErrorException run()
            @test_throws "all 2 tuning candidates failed to evaluate" run()
            @test_throws msg run()
        end
    end
end
