@testset "target transforms" begin
    y = [4.0, 6.0, 5.0, 9.0, 8.0, 10.0]

    @testset "each built-in's forward map and exact inverse" begin
        z, s = fit_transform(NoTransform(), y)
        @test z === y && untransform(NoTransform(), [1.0, 2.0], s) == [1.0, 2.0]

        z, m = fit_transform(MeanScale(), y)
        @test m == sum(y) / 6 && z == y ./ m
        @test untransform(MeanScale(), [1.5], m) == [1.5 * m]

        z, m = fit_transform(LastScale(3), y)
        @test m == (9.0 + 8.0 + 10.0) / 3 && z == y ./ m
        # A series shorter than k is scaled by all of its values.
        @test last(fit_transform(LastScale(10), y)) == sum(y) / 6

        z, last2 = fit_transform(LogRatio(2), y)
        @test z ≈ [log(y[t] / ((y[t - 2] + y[t - 1]) / 2)) for t in 3:6]
        @test last2 == [8.0, 10.0]
        future = [12.0, 9.0, 11.0]
        hist = [y; future]
        zf = [log(hist[t] / ((hist[t - 2] + hist[t - 1]) / 2)) for t in 7:9]
        @test untransform(LogRatio(2), zf, last2) ≈ future

        z, last1 = fit_transform(Difference(1), y)
        @test z == diff(y) && last1 == [10.0]
        @test untransform(Difference(1), diff([10.0; future]), last1) == future
        z, last3 = fit_transform(Difference(3), y)
        @test z == y[4:6] .- y[1:3] && last3 == y[4:6]
        @test untransform(Difference(3), future .- y[4:6], last3) == future
    end

    @testset "Deseasonalize adjusts only series that pass the seasonality test" begin
        season = [0.8, 1.3, 1.1, 0.8]
        seasonal = [10.0 * season[mod1(t, 4)] for t in 1:24]
        z, st = fit_transform(Deseasonalize(4), seasonal)
        @test st.index ≈ season ./ (sum(season) / 4)
        @test all(≈(z[1]), z)
        @test untransform(Deseasonalize(4), fill(z[1], 4), st) ≈ seasonal[1:4]
        noise = 10 .+ randn(StableRNG(3), 40)
        z, st = fit_transform(Deseasonalize(4), noise)
        @test st.index == ones(4) && z == noise
        # Fewer than 3 periods cannot be tested, so they pass through.
        @test last(fit_transform(Deseasonalize(4), seasonal[1:11])).index == ones(4)
        bad = copy(seasonal); bad[5] = -1.0
        @test_throws("needs positive values, but the series has -1.0 at position 5",
                     fit_transform(Deseasonalize(4), bad))
    end

    @testset "a chain applies in order and undoes in reverse" begin
        series = [50 + 10 * sin(2π * t / 12) + 0.3t for t in 1:60]
        chain = TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))
        z, states = fit_transform(chain, series)
        z1, m = fit_transform(LastScale(12), series)
        z2, d = fit_transform(Deseasonalize(12), z1)
        z3, l = fit_transform(LogRatio(12), z2)
        @test z == z3 && length(z) == 48
        ẑ = [0.01, -0.02, 0.03]
        @test untransform(chain, ẑ, states) ==
              untransform(LastScale(12), untransform(Deseasonalize(12),
                                                     untransform(LogRatio(12), ẑ, l), d), m)
        @test TransformChain(Difference(1)) == TransformChain(Difference(1))
        @test hash(TransformChain(Difference(1))) == hash(TransformChain(Difference(1)))
    end

    @testset "invalid transforms and data throw actionable errors" begin
        @test_throws "LastScale needs k ≥ 1 values to average, got 0" LastScale(0)
        @test_throws "Deseasonalize needs a period of at least 2, got 1" Deseasonalize(1)
        @test_throws "LogRatio needs window ≥ 1, got 0" LogRatio(0)
        @test_throws "Difference needs lag ≥ 1, got 0" Difference(0)
        for bad in (true, false)
            @test_throws "LastScale k must be a whole number, not the Bool $bad. Pass " *
                         "e.g. LastScale(12)." LastScale(bad)
            @test_throws "Deseasonalize period must be a whole number, not the Bool " *
                         "$bad. Pass the season length in steps, e.g. " *
                         "Deseasonalize(12)." Deseasonalize(bad)
            @test_throws "LogRatio window must be a whole number, not the Bool $bad. " *
                         "Pass e.g. LogRatio(12)." LogRatio(bad)
            @test_throws "Difference lag must be a whole number, not the Bool $bad. " *
                         "Pass e.g. Difference(1)." Difference(bad)
        end
        @test_throws "TransformChain needs at least one transform" TransformChain()
        @test_throws("MeanScale() cannot scale a series whose training mean is 0.0",
                     fit_transform(MeanScale(), [1.0, -1.0]))
        @test_throws("needs positive values, but the series has 0.0 at position 2",
                     fit_transform(LogRatio(1), [1.0, 0.0, 2.0]))
        @test_throws("LogRatio(3) needs 3 training values to rebuild forecasts, " *
                     "but the series had 2", untransform(LogRatio(3), [0.1], [1.0, 2.0]))
        @test_throws "Difference(2) needs 2 training values" untransform(
            Difference(2), [0.1], [1.0])
    end

    # A naive model on differences extrapolates the last change.
    t0 = Date(2022, 1, 1)
    trend = (ds=collect(t0:Day(1):t0 + Day(39)), y=[3.0 + 2.0 * t for t in 1:40])
    naive(transform; strat=Recursive(), id=nothing) =
        Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                   strategy=strat, freq=Day(1), id=id, transform=transform)

    @testset "forecasts are mapped back to the original scale" begin
        @test naive(NoTransform()).transform isa NoTransform
        @test Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                         freq=Day(1)).transform isa NoTransform
        @test occursin("transform=Difference(1)", sprint(show, naive(Difference(1))))
        @test !occursin("transform", sprint(show, naive(NoTransform())))
        for strat in (Recursive(), Direct(4))
            f = fit(naive(Difference(1); strat), trend)
            @test forecast(f, 4).y_hat ≈ [83.0 + 2.0 * s for s in 1:4]
            @test f.y_history == trend.y && f.z_history == fill(2.0, 39)
            @test f.n_train == 40 && f.transform_state == [83.0]
        end
        # Without the transform, the naive model repeats the last value.
        @test forecast(fit(naive(NoTransform()), trend), 2).y_hat == [83.0, 83.0]
    end

    @testset "each series of a panel keeps its own transform state" begin
        ids = [fill("up", 40); fill("down", 30)]
        ds = [collect(t0:Day(1):t0 + Day(39)); collect(t0:Day(1):t0 + Day(29))]
        yp = [[1.0 + t for t in 1:40]; [500.0 - 5.0 * t for t in 1:30]]
        panel = (unique_id=ids, ds=ds, y=yp)
        for strat in (Recursive(), Direct(3))
            f = fit(naive(Difference(1); strat, id=:unique_id), panel)
            fc = forecast(f, 3)
            @test fc.y_hat[fc.unique_id .== "up"] ≈ [41.0 + s for s in 1:3]
            @test fc.y_hat[fc.unique_id .== "down"] ≈ [350.0 - 5.0 * s for s in 1:3]
            @test [st.transform_state for st in f.series] == [[41.0], [350.0]]
            @test [st.y_history for st in f.series] == [yp[1:40], yp[41:70]]
        end
        # A series the transform leaves too short is skipped, by name.
        short = (unique_id=[ids; "tiny"; "tiny"], ds=[ds; t0; t0 + Day(1)],
                 y=[yp; 1.0; 2.0])
        f = @test_logs (:warn, r"skipping 1 series .*unique_id=\"tiny\"") fit(
            naive(Difference(1); id=:unique_id), short)
        @test nseries(f) == 2
        neg = (unique_id=ids, ds=ds, y=[yp[1:40]; [-1.0; yp[42:70]]])
        @test_throws "series unique_id=\"down\": LogRatio(1) takes logarithms" fit(
            naive(LogRatio(1); id=:unique_id), neg)
    end

    @testset "backtest refits the transform in every fold" begin
        bumpy = (ds=trend.ds, y=trend.y .+ [isodd(t) ? 1.0 : -1.0 for t in 1:40])
        fc = naive(LastScale(5))
        r = backtest(fc, bumpy; horizon=3, initial=30, step=4)
        for o in unique(r.folds.origin)
            k = findfirst(==(o), bumpy.ds)
            manual = forecast(fit(fc, map(c -> c[1:k], bumpy)), 3).y_hat
            @test r.folds.y_hat[r.folds.origin .== o] == manual
        end
        # Data after an origin cannot change that fold's forecasts.
        later = (ds=bumpy.ds, y=[bumpy.y[1:33]; 1000 .* bumpy.y[34:40]])
        r2 = backtest(fc, later; horizon=3, initial=30, step=4)
        first_fold = r.folds.origin .== bumpy.ds[30]
        @test r2.folds.y_hat[first_fold] == r.folds.y_hat[first_fold]
    end

    @testset "tune searches over transforms" begin
        res = tune(naive(NoTransform()), trend;
                   grid=(transform=[NoTransform(), Difference(1), LastScale(4)],),
                   horizon=4, initial=30, step=4, metric=mae)
        @test res.best.transform == Difference(1)
        @test res.table.mean_score[2] ≈ 0.0 atol = 1e-9
    end

    @testset "real learners extrapolate a trend only with a transform" begin
        n = 120
        td = collect(t0:Day(1):t0 + Day(n - 1))
        series = (ds=td, y=[100 + 2.0 * t + 5 * sin(2π * t / 7) +
                            randn(StableRNG(t), 1)[1] for t in 1:n])
        train = map(c -> c[1:100], series)
        actual = series.y[101:120]
        feats = FeatureSet(Lag.(1:7)..., Calendar(:dayofweek))
        for model in (EvoTreeRegressor(nrounds=50, seed=1),
                      DecisionTreeRegressor(max_depth=6, rng=StableRNG(1)))
            err(t) = mae(actual, forecast(fit(Forecaster(model; features=feats, freq=Day(1),
                                                         transform=t), train), 20).y_hat)
            plain, diffed = err(NoTransform()), err(Difference(1))
            @test diffed < plain / 2
        end
    end
end
