# Panel (multi-series) forecasting: one global model over many series.

@testset "panel" begin
    # three ragged series with very different levels, so any cross-series bleed
    # shows up as an obviously wrong number rather than a subtle one
    function makepanel(; lens=(40, 30, 35), levels=(100.0, 10.0, 50.0))
        ids = String[]; ds = Date[]; y = Float64[]
        for (k, (n, lv)) in enumerate(zip(lens, levels))
            t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
            append!(ids, fill("s$k", n)); append!(ds, t)
            append!(y, lv .+ Float64.(1:n))
        end
        return (unique_id=ids, ds=ds, y=y)
    end
    panel = makepanel()
    mk(model, feats; strat=Recursive()) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1),
                   target=:y, time=:ds, id=:unique_id)

    @testset "construction validates the id column" begin
        f = FeatureSet(Lag(1))
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0); features=f,
                                              freq=Day(1), target=:y, time=:ds, id=:y)
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0); features=f,
                                              freq=Day(1), target=:y, time=:ds, id=:ds)
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0); features=f,
                                              freq=Day(1), target=:y, time=:ds, id=:y_hat)
        err = try Forecaster(TestModels.LinAR(1.0, 0.0);
                             features=FeatureSet(Lag(1), Exogenous(:sid)),
                             freq=Day(1), target=:y, time=:ds, id=:sid) catch e; e end
        @test err isa ArgumentError && occursin("series id", err.msg)
        # id=nothing is the single-series default, and dispatch reflects it
        @test !MachineLearningForecast.ispanel(Forecaster(TestModels.LinAR(1.0, 0.0);
                                              features=f, freq=Day(1)))
        @test MachineLearningForecast.ispanel(mk(TestModels.LinAR(1.0, 0.0), f))
    end

    @testset "fit covers every series and keeps their state apart" begin
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel)
        @test nseries(f) == 3
        @test length(f.machines) == 1                    # ONE global model
        @test [s.id for s in f.series] == ["s1", "s2", "s3"]
        @test [s.n_train for s in f.series] == [40, 30, 35]
        @test [s.t_last for s in f.series] ==
              [Date(2022, 2, 9), Date(2022, 1, 30), Date(2022, 2, 4)]
        # single-series accessors are meaningless here and say so
        err = try f.y_history catch e; e end
        @test err isa ArgumentError && occursin("3 series", err.msg)
    end

    @testset "features never reach across a series boundary" begin
        # MeanModel predicts the mean of its own training target, so the pooled
        # target set is directly observable. Lag(1) drops exactly one row per
        # series; a lag bleeding across the boundary would add rows (and change
        # the mean), so this pins per-series materialisation.
        f = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(1))), panel)
        expected = Float64[]
        for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))
            append!(expected, lv .+ Float64.(2:n))       # row 1 of each series dropped
        end
        @test only(unique(forecast(f, 1).y_hat)) ≈ mean(expected)

        # ...and with Lag(7) each series loses exactly 7 rows, not 7 overall
        f7 = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(7))), panel)
        exp7 = Float64[]
        for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))
            append!(exp7, lv .+ Float64.(8:n))
        end
        @test only(unique(forecast(f7, 1).y_hat)) ≈ mean(exp7)
    end

    @testset "each series forecasts from its own history and timestamp" begin
        # LinAR(1, 0) echoes y_lag_1, so every step returns that series' own last value
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel)
        out = forecast(f, 3)
        @test keys(out) == (:unique_id, :ds, :y_hat)
        @test length(out.y_hat) == 3 * 3
        for (k, st) in enumerate(f.series)
            rows = (3k - 2):(3k)
            @test all(out.unique_id[rows] .== st.id)
            @test out.y_hat[rows] ≈ fill(st.y_history[end], 3)      # own level
            @test out.ds[rows] == [st.t_last + Day(i) for i in 1:3] # own grid (ragged)
        end
    end

    @testset "grouping tolerates interleaved and unsorted rows" begin
        n = length(panel.y)
        perm = shuffle(StableRNG(3), 1:n)
        shuffled = (unique_id=panel.unique_id[perm], ds=panel.ds[perm], y=panel.y[perm])
        a = forecast(fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel), 2)
        b = forecast(fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), shuffled), 2)
        # series order follows first appearance, so shuffling reorders the rows
        # but must not change any series' forecast
        keyed(o) = Dict((o.unique_id[i], o.ds[i]) => o.y_hat[i] for i in eachindex(o.y_hat))
        @test Set(keys(keyed(a))) == Set(keys(keyed(b)))
        @test all(keyed(a)[k] ≈ keyed(b)[k] for k in keys(keyed(a)))
    end

    @testset "per-series validation errors name the series" begin
        bad = makepanel()
        drop = findfirst(i -> bad.unique_id[i] == "s2" && bad.ds[i] == Date(2022, 1, 15),
                         eachindex(bad.y))
        @test drop !== nothing                      # the fixture really has that row
        keep = setdiff(eachindex(bad.y), drop)
        gapped = (unique_id=bad.unique_id[keep], ds=bad.ds[keep], y=bad.y[keep])
        err = try fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), gapped) catch e; e end
        @test err isa ArgumentError
        @test occursin("series unique_id=\"s2\"", err.msg) && occursin("gap", err.msg)

        miss = (unique_id=Vector{Union{Missing,String}}(bad.unique_id),
                ds=bad.ds, y=bad.y)
        miss.unique_id[3] = missing
        err2 = try fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), miss) catch e; e end
        @test err2 isa ArgumentError && occursin("missing value", err2.msg)
    end

    @testset "series too short to train are skipped, not fatal" begin
        short = makepanel(lens=(40, 3, 35))
        f = @test_logs (:warn, r"skipping 1 series") match_mode=:any fit(
            mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(7))), short)
        @test nseries(f) == 2 && [s.id for s in f.series] == ["s1", "s3"]
        # but if none is usable that is an error, not an empty model
        err = try fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(30))),
                      makepanel(lens=(5, 4, 3))) catch e; e end
        @test err isa ArgumentError && occursin("no series has enough rows", err.msg)
    end

    @testset "exogenous covariates join on (id, time)" begin
        ex = (unique_id=panel.unique_id, ds=panel.ds, y=panel.y,
              promo=Float64.(eachindex(panel.y)))
        fc = mk(TestModels.EchoColumn(:promo), FeatureSet(Lag(1), Exogenous(:promo)))
        f = fit(fc, ex)
        grids = [[s.t_last + Day(i) for i in 1:2] for s in f.series]
        nd = (unique_id=repeat(["s1", "s2", "s3"], inner=2),
              ds=vcat(grids...), promo=[11.0, 12.0, 21.0, 22.0, 31.0, 32.0])
        @test forecast(f, 2; new_data=nd).y_hat == [11.0, 12.0, 21.0, 22.0, 31.0, 32.0]
        # out-of-order new_data joins by key, not by position
        p2 = shuffle(StableRNG(9), 1:6)
        shuf = (unique_id=nd.unique_id[p2], ds=nd.ds[p2], promo=nd.promo[p2])
        @test forecast(f, 2; new_data=shuf).y_hat == [11.0, 12.0, 21.0, 22.0, 31.0, 32.0]

        @test_throws ArgumentError forecast(f, 2)                    # new_data required
        missing_row = (unique_id=nd.unique_id[1:5], ds=nd.ds[1:5], promo=nd.promo[1:5])
        err = try forecast(f, 2; new_data=missing_row) catch e; e end
        @test err isa ArgumentError && occursin("no row for", err.msg)
        noid = (ds=nd.ds, promo=nd.promo)
        err2 = try forecast(f, 2; new_data=noid) catch e; e end
        @test err2 isa ArgumentError && occursin("unique_id", err2.msg)
    end

    @testset "Direct on a panel" begin
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=Direct(4)), panel)
        @test length(f.machines) == 4
        out = forecast(f, 4)
        @test length(out.y_hat) == 3 * 4
        @test_throws ArgumentError forecast(f, 5)

        # each step-model must see the target shifted WITHIN its series: MeanModel
        # exposes which targets machine i was trained on.
        fm = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(1)); strat=Direct(3)), panel)
        got = forecast(fm, 3)
        for i in 1:3
            pooled = Float64[]
            for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))
                append!(pooled, lv .+ Float64.((1 + i):n))   # y[i:nX] within series
            end
            @test got.y_hat[i] ≈ mean(pooled)
        end
    end

    @testset "backtest cuts folds on the global timestamp grid" begin
        big = makepanel(lens=(60, 50, 55))
        fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        r = backtest(fc, big; horizon=5, initial=40, step=10, metrics=(mae, rmse))
        @test keys(r.folds) == (:origin, :step, :unique_id, :ds, :y, :y_hat)
        @test length(unique(r.folds.origin)) == length(40:10:(60 - 5))
        @test extrema(r.folds.step) == (1, 5)
        @test Set(unique(r.folds.unique_id)) ⊆ Set(["s1", "s2", "s3"])
        # every scored row has a real actual behind it
        actual = Dict((big.unique_id[i], big.ds[i]) => big.y[i] for i in eachindex(big.y))
        @test all(actual[(r.folds.unique_id[j], r.folds.ds[j])] == r.folds.y[j]
                  for j in eachindex(r.folds.y))
        @test occursin("horizon 5", sprint(show, MIME"text/plain"(), r))
        @test occursin("folds", sprint(show, r))
        # initial is counted in timestamps, and must clear minhistory
        @test_throws ArgumentError backtest(mk(TestModels.LinAR(1.0, 0.0),
                                               FeatureSet(Lag(30))), big;
                                            horizon=5, initial=20, step=10)
        for bad in (mae, (:mae,), ("mae",))
            @test_throws ArgumentError backtest(fc, big; horizon=5, initial=40,
                                                metrics=bad)
            @test_throws "metrics=(mae, rmse)" backtest(fc, big; horizon=5,
                                                        initial=40, metrics=bad)
        end
    end

    @testset "backtest with Exogenous on ragged and phase-offset panels" begin
        rng = StableRNG(11)
        # s1 ends before the last origins, s3 inside a window.
        daily = makepanel(lens=(50, 60, 56))
        daily = merge(daily, (y=daily.y .+ randn(rng, length(daily.y)),
                              promo=Float64.(eachindex(daily.y))))
        mondays = collect(Date(2024, 1, 1):Week(1):Date(2024, 7, 22))
        weekly = (unique_id=repeat(["a", "b"], inner=30),
                  ds=vcat(mondays, mondays .+ Day(3)),
                  y=repeat([10.0, 20.0], inner=30) .+ randn(rng, 60),
                  promo=Float64.(1:60))
        cases = ((daily, Day(1), 5, 30, 4), (weekly, Week(1), 2, 20, 3))
        for (data, freq, h, init, stp) in cases,
            m in (EvoTreeRegressor(nrounds=10), DecisionTreeRegressor(max_depth=3)),
            s in (Recursive(), Direct(h))
            bt(fs) = backtest(Forecaster(m; features=fs, strategy=s, freq=freq,
                                         id=:unique_id),
                              data; horizon=h, initial=init, step=stp)
            plain = bt(FeatureSet(Lag(1)))
            exog = bt(FeatureSet(Lag(1), Exogenous(:promo)))
            @test exog.folds.origin == plain.folds.origin
            @test exog.folds.step == plain.folds.step
            @test exog.folds.unique_id == plain.folds.unique_id
            @test exog.folds.ds == plain.folds.ds
        end
        # EchoColumn shows each step reads its own (id, time) row.
        for (data, freq, h, init, stp) in cases
            fc = Forecaster(TestModels.EchoColumn(:promo); freq=freq, id=:unique_id,
                            features=FeatureSet(Lag(1), Exogenous(:promo)))
            r = backtest(fc, data; horizon=h, initial=init, step=stp)
            promo = Dict((data.unique_id[i], data.ds[i]) => data.promo[i]
                         for i in eachindex(data.promo))
            @test r.folds.y_hat == [promo[(r.folds.unique_id[j], r.folds.ds[j])]
                                    for j in eachindex(r.folds.ds)]
        end
    end

    @testset "tune drives a panel through backtest" begin
        big = makepanel(lens=(60, 50, 55))
        base = mk(TestModels.MeanModel(), FeatureSet(Lag(1)))
        res = tune(base, big; grid=(features=[FeatureSet(Lag(1)), FeatureSet(Lag(1), Lag(7))],),
                   horizon=5, initial=40, step=10, metric=mae)
        @test length(res.table.mean_score) == 2
        @test all(!ismissing, res.table.mean_score)
        @test MachineLearningForecast.ispanel(res.best)                 # id survives reconstruct
        @test nseries(res.best_fitted) == 3
    end

    @testset "fit, backtest and tune leave the model prototype unchanged" begin
        big = makepanel(lens=(60, 50, 55))
        noisy = merge(big, (y=big.y .+ randn(StableRNG(5), length(big.y)),))
        fs = FeatureSet(Lag(1), Lag(2), Lag(7), RollingMean(3))
        # MLJ's == ignores RNG state, so compare every field.
        same(a, b) = all(k -> isequal(getfield(a, k), getfield(b, k)),
                         fieldnames(typeof(a)))
        for m in (DecisionTreeRegressor(n_subfeatures=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=10, rowsample=0.5, colsample=0.5)),
            s in (Recursive(), Direct(3))
            before = deepcopy(m)
            fc = mk(m, fs; strat=s)
            first_fit = forecast(fit(fc, noisy), 3).y_hat
            @test same(m, before)
            @test forecast(fit(fc, noisy), 3).y_hat == first_fit
            backtest(fc, noisy; horizon=3, initial=40, step=10)
            @test same(m, before)
            r = tune(fc, noisy; grid=(features=[fs, fs, fs],), horizon=3, initial=40,
                     step=10)
            @test same(m, before)
            @test allequal(r.table.mean_score)
        end
    end

    @testset "show displays panel specs and fitted panels" begin
        for s in (Recursive(), Direct(2))
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=s)
            @test occursin("id=:unique_id", sprint(show, fc))
            f = fit(fc, panel)
            for txt in (sprint(show, f), sprint(show, MIME"text/plain"(), f))
                @test occursin("trained on 3 series", txt)
                @test occursin(string(s), txt)
            end
        end
        single = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                            freq=Day(1))
        @test !occursin("id=", sprint(show, single))
    end

    @testset "single-series behaviour is unchanged" begin
        df = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19)), y=Float64.(1:50))
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        strategy=Recursive(), freq=Day(1))
        f = fit(fc, df)
        @test nseries(f) == 1
        @test f.y_history == Float64.(1:50)                 # documented accessors work
        @test f.t_last == Date(2022, 2, 19) && f.n_train == 50
        @test keys(forecast(f, 3)) == (:ds, :y_hat)         # no id column
    end
end
