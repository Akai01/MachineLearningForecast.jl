# Uses only the public API, as third-party code would.
module ThirdPartyTuning

using MachineLearningForecast: TuningStrategy
import MachineLearningForecast: ask!, tell!

mutable struct Midpoint{T<:NamedTuple} <: TuningStrategy
    space::Vector{T}
    asks::Int
    told::Vector{Tuple{NamedTuple,Union{Missing,Float64}}}
end
Midpoint(space) = Midpoint(space, 0, Tuple{NamedTuple,Union{Missing,Float64}}[])

function ask!(s::Midpoint)
    s.asks ≥ 3 && return nothing
    s.asks += 1
    return s.space[(length(s.space) + 1) ÷ 2]
end

tell!(s::Midpoint, candidate, score) = push!(s.told, (candidate, score))

"Always proposes `out`, to check what tune does with each kind of answer."
struct Fixed{T} <: TuningStrategy
    out::T
end
ask!(s::Fixed) = s.out
tell!(::Fixed, candidate, score) = nothing

"Proposes each candidate of `list` once, in order."
mutable struct Replay <: TuningStrategy
    list::Vector{NamedTuple}
    i::Int
end
ask!(s::Replay) = s.i < length(s.list) ? s.list[s.i += 1] : nothing
tell!(::Replay, candidate, score) = nothing

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
        @test err isa ArgumentError
        @test_throws "all 2 tuning candidates failed to evaluate, so there is nothing " *
                     "to rank. Fix the first error listed below, or check horizon, " *
                     "initial and the grid (or the tuner's candidates) against the data. " *
                     "Candidates:\n" *
                     "  candidate 1 (features=FeatureSet(Lag(200))): ArgumentError: " tune(
            base, df; grid=bad, horizon=14, initial=90)
        @test occursin("\n  candidate 2 (features=FeatureSet(Lag(300))): ArgumentError: ",
                       err.msg)
        @test !occursin("... and", err.msg)
    end

    @testset "all candidates failing lists them, capped at 5" begin
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
            fcm = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            bad = (features=[FeatureSet(Lag(200 + k)) for k in 1:7],)
            err = try tune(fcm, df; grid=bad, horizon=14, initial=90) catch e; e end
            @test err isa ArgumentError
            @test occursin("all 7 tuning candidates failed to evaluate", err.msg)
            lines = split(err.msg, '\n')
            @test length(lines) == 1 + 5 + 1
            for k in 1:5
                @test startswith(lines[1 + k],
                                 "  candidate $k (features=FeatureSet(Lag($(200 + k)))): ")
            end
            @test lines[end] == "  ... and 2 more"
            @test !occursin("Lag(206)", err.msg)
            # Exactly the cap: all listed and no tail.
            five = (features=[FeatureSet(Lag(200 + k)) for k in 1:5],)
            err = try tune(fcm, df; grid=five, horizon=14, initial=90) catch e; e end
            @test length(split(err.msg, '\n')) == 1 + 5
            @test occursin("candidate 5 (features=FeatureSet(Lag(205))): ", err.msg)
            @test !occursin("... and", err.msg)
        end
        # Two override keys, a model rendered by _short, 2 learners.
        grid = (features=[FeatureSet(Lag(200)), FeatureSet(Lag(300))],
                model=[DecisionTreeRegressor(max_depth=2), EvoTreeRegressor(nrounds=5)])
        err = try tune(base, df; grid=grid, horizon=14, initial=90) catch e; e end
        @test err isa ArgumentError && occursin("all 4 tuning candidates", err.msg)
        @test occursin("\n  candidate 1 (features=FeatureSet(Lag(200)), model=" *
                       "DecisionTreeRegressor(max_depth=2)): ArgumentError: ", err.msg)
        @test occursin("\n  candidate 4 (features=FeatureSet(Lag(300)), " *
                       "model=EvoTreeRegressor(nrounds=5)): ArgumentError: ", err.msg)
        # Candidate 1 shows its whole error, indented under its line.
        boom = CustomFeature(:boom, h -> error("\nfirst line\nsecond line"), 1)
        run_boom(fs) = try tune(base, df; grid=(features=fs,), horizon=14,
                                initial=90) catch e; e end
        err = run_boom([FeatureSet(boom)])
        @test err isa ArgumentError
        @test occursin(r"\n  candidate 1 \(features=[^\n]*\): [^\n]*first line\n" *
                       r"    second line$", err.msg)
        # Later candidates show only their first non-empty line.
        err = run_boom([FeatureSet(Lag(200)), FeatureSet(boom)])
        @test occursin(r"\n  candidate 2 \(features=[^\n]*\): [^\n]*first line$",
                       err.msg)
        @test !occursin("second line", err.msg)
        tuner = ThirdPartyTuning.Midpoint([(model=TestModels.NaNModel(),)])
        err = try tune(base, df; tuner=tuner, horizon=14, initial=90) catch e; e end
        @test err isa ArgumentError && occursin("candidate 3 (model=NaNModel()): ", err.msg)
    end

    @testset "_short renders missing and NaN hyperparameters" begin
        short = MachineLearningForecast._short
        @test short(TestModels.Hyper()) == "Hyper(a=missing, b=NaN)"
        @test short(TestModels.Hyper(; b=2.0)) == "Hyper(b=2.0)"
        @test short(TestModels.Hyper(; a=1.0)) == "Hyper(a=1.0)"
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

    @testset "third-party sequential strategy drives the ask!/tell! loop" begin
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
        @test_throws ArgumentError tune(base, df; tuner=tuner, horizon=14, initial=90)
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
        for bad in (true, false)
            @test_throws "RandomSearch n must be a whole number, not the Bool $bad. " *
                         "Pass the number of candidates to draw, e.g. " *
                         "RandomSearch(10)." RandomSearch(bad)
        end
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
        for bad in (true, false)
            g = (model=[base.model],)
            @test_throws "max_evals must be a whole number, not the Bool $bad. Pass " *
                         "the number of candidates to fit, e.g. max_evals=10, or " *
                         "nothing for no budget." tune(base, df; grid=g, horizon=14,
                                                       initial=90, max_evals=bad)
            @test_throws "tune horizon must be a whole number, not the Bool $bad. Pass " *
                         "the number of steps each fold forecasts, e.g. " *
                         "horizon=28." tune(base, df; grid=g, horizon=bad, initial=90)
            @test_throws "tune initial must be a whole number, not the Bool $bad. Pass " *
                         "the size of the first training window, e.g. " *
                         "initial=730." tune(base, df; grid=g, horizon=14, initial=bad)
            @test_throws "tune windows must be a whole number, not the Bool $bad. Pass " *
                         "the number of folds to hold out at the end of each series, " *
                         "e.g. windows=3." tune(base, df; grid=g, horizon=14,
                                                windows=bad)
            @test_throws "tune step must be a whole number, not the Bool $bad. Pass how " *
                         "many steps each fold moves the origin, e.g. step=14, which " *
                         "equals horizon (the default)." tune(base, df; grid=g,
                                                              horizon=14, initial=90,
                                                              step=bad)
        end
    end

    @testset "ask! must answer with a NamedTuple of known fields or nothing" begin
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
            fcm = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            run(out) = tune(fcm, df; tuner=ThirdPartyTuning.Fixed(out), max_evals=2,
                            horizon=14, initial=90)
            @test_throws ArgumentError run(42)
            @test_throws "ask!(Fixed) returned 42; the ask!/tell! contract requires a " *
                         "NamedTuple of Forecaster field overrides, or nothing to stop." (
                run(42))
            @test_throws ArgumentError run(nothing)
            @test_throws "the tuning strategy proposed no candidates (Fixed): ask! " *
                         "returned nothing on its first call, so there is nothing to " *
                         "tune. Make ask! return at least one NamedTuple of Forecaster " *
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
        msg = "ArgumentError: features must be a FeatureSet such as " *
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
            @test_throws ArgumentError run((features=[Lag(1)],))
            @test_throws msg run((features=[Lag(1)],))
        end
        @test_throws "such as FeatureSet(Lag(1), Lag(7))" tune(
            base, df; grid=(features=[Lag(1)],), horizon=14, initial=90)
    end

    @testset "reconstruct round-trips every field and applies each change" begin
        MLF = MachineLearningForecast
        alt = (model=EvoTreeRegressor(nrounds=7), features=FeatureSet(Lag(3)),
               strategy=Direct(2), freq=Hour(1), target=:v, time=:t, id=:uid,
               transform=LastScale(3), intervals=MLF.PoolIntervals(7))
        # A new field must be added here, or this test fails.
        @test Set(keys(alt)) == Set(fieldnames(Forecaster))
        @test Set(MLF.TUNABLE_FIELDS) ⊆ Set(fieldnames(Forecaster))
        for m in (TestModels.LinAR(1.0, 0.0),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5)),
            id in (nothing, :unique_id)
            fc = Forecaster(m; features=FeatureSet(Lag(1), Lag(2)), strategy=Direct(3),
                            freq=Month(1), target=:z, time=:when, id,
                            transform=LastScale(12))
            same = MLF.reconstruct(fc)
            @test same isa Forecaster
            for f in fieldnames(Forecaster)
                @test getfield(same, f) === getfield(fc, f)
            end
            for f in fieldnames(Forecaster)
                if f in MLF.TUNABLE_FIELDS
                    changed = MLF.reconstruct(fc; f => alt[f])
                    for g in fieldnames(Forecaster)
                        @test getfield(changed, g) === (g === f ? alt[f] : getfield(fc, g))
                    end
                else
                    @test_throws ArgumentError MLF.reconstruct(fc; f => alt[f])
                end
            end
            both = MLF.reconstruct(fc; model=alt.model, strategy=alt.strategy)
            @test both.model === alt.model && both.strategy === alt.strategy
            @test both.features === fc.features && both.transform === fc.transform
            # The prototype is never mutated.
            @test fc.strategy == Direct(3) && fc.model === m
        end
    end

    @testset "reconstruct rejects keys that are not tunable" begin
        MLF = MachineLearningForecast
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        tunable = ":model, :features, :strategy, :transform"
        @test_throws ArgumentError MLF.reconstruct(fc; bogus=1)
        @test_throws "cannot reconstruct Forecaster with unknown field :bogus; valid " *
                     "fields are $tunable." MLF.reconstruct(fc; bogus=1)
        for key in (:freq, :target, :time, :id)
            @test_throws ArgumentError MLF.reconstruct(fc; NamedTuple{(key,)}((1,))...)
            @test_throws "field :$key describes the data and cannot be tuned; " *
                         "tunable fields are $tunable." (
                MLF.reconstruct(fc; NamedTuple{(key,)}((1,))...))
        end
        @test_throws "fields :freq, :id describe the data and cannot be tuned; " *
                     "tunable fields are" MLF.reconstruct(fc; freq=Day(2), id=:u)
        # A name that is not a field is reported as unknown, alone.
        @test_throws "unknown field :bogus; valid fields are" (
            MLF.reconstruct(fc; bogus=1, freq=Day(2)))
        @test_throws "unknown fields :bogus, :other; valid fields are" (
            MLF.reconstruct(fc; bogus=1, other=2))
        @test_throws "grid key :freq describes the data and cannot be tuned; tunable " *
                     "keys are $tunable." tune(
            fc, (ds=t, y=y); grid=(freq=[Day(1), Day(2)],), horizon=14, initial=90)
        @test_throws "grid keys :target, :id describe the data and cannot be tuned" tune(
            fc, (ds=t, y=y); grid=(target=[:y], id=[:u]), horizon=14, initial=90)
        @test_throws "grid has unknown key :bogus; valid keys are $tunable." tune(
            fc, (ds=t, y=y); grid=(bogus=[1],), horizon=14, initial=90)
    end

    @testset "an ask! candidate with a data field fails only that candidate" begin
        for m in (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
            fcm = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            msg = "ArgumentError: field :freq describes the data and cannot be tuned; " *
                  "tunable fields are :model, :features, :strategy, :transform."
            r = tune(fcm, df; tuner=ThirdPartyTuning.Replay(
                         NamedTuple[(freq=Day(2),), (strategy=Recursive(),)], 0),
                     max_evals=2, horizon=14, initial=90)
            @test ncand(r) == 2
            @test ismissing(r.table.mean_score[1]) && r.table.error[1] == msg
            @test ismissing(r.table.error[2]) && isfinite(r.table.mean_score[2])
            @test r.best.freq == Day(1) && r.best.strategy == Recursive()
        end
    end

    @testset "reconstruct re-runs the constructor checks" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        @test_throws "the feature set produces a column named :y, which is the target " *
                     "column" MachineLearningForecast.reconstruct(
            fc; features=FeatureSet(Lag(1), Exogenous(:y)))
        @test_throws "model must be an MLJ model instance" (
            MachineLearningForecast.reconstruct(fc; model="not a model"))
        @test_throws "got Lag(1). Wrap the features" MachineLearningForecast.reconstruct(
            fc; features=Lag(1))
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
        # 27 candidates × 2 folds × 14 models = 756 > 500
        @test_logs (:warn, r"756 model fits") match_mode=:any tune(
            base, df; grid=big, horizon=14, initial=90)

        # max_evals=1 plans only 28 fits: no warning.
        @test_logs min_level=Logging.Warn tune(
            base, df; grid=big, max_evals=1, horizon=14, initial=90)
        # A budget over the threshold warns with its own count.
        @test_logs (:warn, r"560 model fits") match_mode=:any tune(
            base, df; grid=big, max_evals=20, horizon=14, initial=90)
    end

    @testset "mean_score and std_score come from summarize" begin
        grid = (model=[DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                       TestModels.LinAR(1.0, 0.0)],)
        for kw in ((initial=70, step=14), (windows=3, step=7))
            r = tune(base, df; grid, horizon=7, kw..., metric=mae)
            for (k, m) in enumerate(grid.model)
                bt = backtest(MachineLearningForecast.reconstruct(base; model=m), df;
                              horizon=7, kw..., metrics=(mae,))
                s = summarize(bt)
                @test r.table.mean_score[k] == only(s.mean)
                @test r.table.std_score[k] == only(s.std)
                @test only(s.nfolds) > 1
            end
        end
        # One fold: summarize gives NaN, the table keeps missing.
        r1 = tune(base, df; grid, horizon=7, windows=1, metric=mae)
        @test r1.table.std_score isa Vector{Union{Missing,Float64}}
        @test all(ismissing, r1.table.std_score)
        for (k, m) in enumerate(grid.model)
            s = summarize(backtest(MachineLearningForecast.reconstruct(base; model=m),
                                   df; horizon=7, windows=1, metrics=(mae,)))
            @test only(s.nfolds) == 1 && isnan(only(s.std))
            @test r1.table.mean_score[k] == only(s.mean)
        end
    end

    @testset "edge cases: horizon 1 and bad data" begin
        grid = (model=[DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                       EvoTreeRegressor(nrounds=5)],)
        r = tune(base, df; grid=grid, horizon=1, initial=110, metric=mae)
        @test ncand(r) == 2 && all(!ismissing, r.table.mean_score)
        for (k, m) in enumerate(grid.model)
            bt = backtest(Forecaster(m; features=base.features, freq=Day(1)), df;
                          horizon=1, initial=110, metrics=(mae,))
            @test r.table.mean_score[k] == only(summarize(bt).mean)
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
            @test_throws ArgumentError run()
            @test_throws "all 2 tuning candidates failed to evaluate" run()
            @test_throws msg run()
        end
    end

    @testset "metric protocol: scaled metrics and up-front checks" begin
        models = (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=5))
        grid = (model=models,)
        pd = let ids = String[], ds = Date[], yy = Float64[]
            for (id, lv, len) in (("a", 100.0, 100), ("b", 5.0, 90)), i in 1:len
                push!(ids, id); push!(ds, Date(2022, 1, 1) + Day(i - 1))
                push!(yy, lv + i + 3 * sin(i) + (i % 7))
            end
            (unique_id=ids, ds=ds, y=yy)
        end
        pbase = Forecaster(models[1]; features=FeatureSet(Lag(1), Lag(7)),
                           freq=Day(1), id=:unique_id)

        @testset "Mase and Rmsse as the tune metric match backtest" begin
            for (spec, data) in ((base, df), (pbase, pd)),
                metric in (Mase(7), Rmsse(), Mase(), Rmsse(3))
                r = tune(spec, data; grid, horizon=14, initial=70, step=7, metric)
                @test all(ismissing, r.table.error)
                for (k, m) in enumerate(models)
                    bt = backtest(MachineLearningForecast.reconstruct(spec; model=m),
                                  data; horizon=14, initial=70, step=7, metrics=(metric,))
                    @test r.table.mean_score[k] == only(summarize(bt).mean)
                end
                @test r.best.model == models[argmin(r.table.mean_score)]
            end
        end

        @testset "the metric is checked before any fit" begin
            for bad in ((mae, rmse), [mae], :mae, "smape", :a => mae, (a=mae,),
                        (Mase(7),))
                @test_throws ArgumentError tune(base, df; grid, horizon=14, initial=90,
                                                metric=bad)
                @test_throws "got $(repr(bad)). Pass e.g. metric=smape" tune(
                    base, df; grid, horizon=14, initial=90, metric=bad)
            end
            @test_throws "tune metric: got the type Mase; pass Mase() or Mase(12)" tune(
                base, df; grid, horizon=14, initial=90, metric=Mase)
            @test_throws "tune metric: got the type Rmsse; pass Rmsse() or Rmsse(12)" tune(
                base, df; grid, horizon=14, initial=90, metric=Rmsse)
            @test_throws "tune metric: metric_inputs returned true, which is not " *
                         "Point() or Scaled()." tune(
                base, df; grid, horizon=14, initial=90, metric=TestModels.badtrait)
            # A tuner that would fail on its first ask never runs.
            tuner = ThirdPartyTuning.Fixed(1)
            @test_throws "tune metric: got the type Mase" tune(
                base, df; tuner, horizon=14, initial=90, metric=Mase)
        end

        @testset "a model predicting NaN fails the candidate under Mase" begin
            r = tune(base, df; grid=(model=[TestModels.NaNModel(), models[1]],),
                     horizon=14, initial=90, metric=Mase(7))
            @test ismissing(r.table.mean_score[1])
            @test occursin("non-finite mean score (NaN)", r.table.error[1])
            @test r.best.model == models[1]
            @test_throws "all 1 tuning candidates failed to evaluate" tune(
                base, df; grid=(model=[TestModels.NaNModel()],), horizon=14,
                initial=90, metric=Rmsse())
        end

        @testset "tune does not ask the metric for its name" begin
            r = tune(base, df; grid, horizon=14, initial=90, metric=TestModels.BadName())
            @test all(ismissing, r.table.error)
        end
    end

    @testset "tune refuses a metric that scores samples" begin
        d = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(59)),
             y=Float64.(1:60))
        spec = Forecaster(TestModels.EchoColumn(:y_lag_1); features=FeatureSet(Lag(1)),
                          freq=Day(1))
        for metric in (crps, fair_crps, pinball_loss, Coverage(95), IntervalWidth(80))
            @test_throws ArgumentError tune(
                spec, d; grid=(strategy=[Recursive()],), horizon=5, initial=40, metric)
            @test_throws "tune ranks candidates by a point-forecast metric, but " *
                         "$(repr(metric)) scores sampled forecasts, which tune does not " *
                         "produce. Pass a point metric such as smape or Mase(12), and " *
                         "compare interval settings with backtest(...; metrics=(crps, " *
                         "Coverage(95)))." tune(
                spec, d; grid=(strategy=[Recursive()],), horizon=5, initial=40, metric)
        end
    end
end
