@testset "SplitConformal seam" begin
    C = MachineLearningForecast
    t = collect(Date(2020, 1, 1):Month(1):Date(2022, 12, 1))
    y = [10.0 + mod1(i, 12) + i / 4 for i in 1:36]
    data = (ds=t, y=y)
    ids = [fill("a", 36); fill("b", 30)]
    panel = (unique_id=ids, ds=[t; t[7:36]], y=[y; 2 .* y[7:36]])
    fc = Forecaster(TestModels.LinAR(0.5, 1.0); features=FeatureSet(Lag(1)), freq=Month(1))
    pfc = Forecaster(TestModels.LinAR(0.5, 1.0); features=FeatureSet(Lag(1)),
                     freq=Month(1), id=:unique_id)
    csp = ConformalSeasonalPool(12; freq=Month(1), rng=StableRNG(9))

    # Recorded at dd55ba3 under --check-bounds=yes.
    @testset "the point fit gives the pre-seam forecasts" begin
        plain = [16.5, 9.25, 5.625, 3.8125]
        for fit_ in (C._fit_spec, C._fit_point)
            @test forecast(fit_(fc, data, 1:36), 4).y_hat == plain
            @test forecast(fit_(csp, data, 1:36), 3).y_hat == [17.25, 18.5, 19.75]
            @test forecast(fit_(pfc, panel, 1:66), 3).y_hat ==
                  [16.5, 9.25, 5.625, 32.0, 17.0, 9.5]
        end
        @test forecast(C._fit_spec(fc, data), 4).y_hat == plain
        @test forecast(fit(fc, data), 4).y_hat == plain
    end

    @testset "a pool forecaster keeps its intervals and samples" begin
        for fit_ in (C._fit_spec, C._fit_point)
            f = fit_(csp, data, 1:36)
            o = forecast(f, 3; levels=(80, 95))
            @test o.lo_80 == [11.25, 12.5, 13.75]
            @test o.hi_80 == [20.25, 21.5, 22.75]
            @test o.lo_95 == [11.25, 12.5, 13.75]
            @test o.hi_95 == [20.25, 21.5, 22.75]
            s = forecast_samples(f, 2).y_hat
            @test length(s) == 200
            @test s[1:6] == [20.25, 17.25, 20.25, 20.25, 20.25, 20.25]
            @test sum(s) == 3488.0
        end
    end

    @testset "the fold fit falls back to the point fit" begin
        for (f_, d, n) in ((fc, data, 36), (csp, data, 36), (pfc, panel, 66))
            tbl = C.normalize_table(d)
            a = C._fit_fold(f_.intervals, f_, tbl, 1:n)
            b = C._fit_point(f_, tbl, 1:n)
            @test forecast(a, 3).y_hat == forecast(b, 3).y_hat
        end
    end

    @testset "the hooks do nothing without SplitConformal" begin
        tbl = C.normalize_table(panel)
        groups = C.panel_groups(pfc, tbl)
        rows = collect(1:10)
        for (f_, iv) in ((fc, nothing), (csp, csp.intervals))
            @test f_.intervals === iv
            @test C._check_support(iv, f_, (0.8,), 3, "forecast with levels") === nothing
            @test C._check_calibratable(iv, f_, C.normalize_table(data)) === nothing
            @test C._check_fold_rows(iv, f_, [20, 26, 32], t) === nothing
            @test C._fold_train_rows(iv, pfc, groups, Dict("a" => t[10]), rows, 1,
                                     t[10]) === rows
            @test C._fold_train_rows(iv, pfc, groups, Dict("a" => t[10]), rows, 2,
                                     missing) === rows
        end
    end

    @testset "point-only backtest folds equal fit folds" begin
        r = backtest(fc, data; horizon=3, windows=3)
        for k in 1:3
            o = only(unique(r.forecasts.origin[r.forecasts.fold .== k]))
            n = findfirst(==(o), t)
            f = fit(fc, (ds=t[1:n], y=y[1:n]))
            @test r.forecasts.y_hat[r.forecasts.fold .== k] == forecast(f, 3).y_hat
        end
        # A row table is normalized once, before the folds.
        rows = [(ds=t[i], y=y[i]) for i in 1:36]
        @test backtest(fc, rows; horizon=3, windows=3).forecasts == r.forecasts

        pdata = (unique_id=repeat(["a", "b"]; inner=36), ds=[t; t], y=[y; 2 .* y])
        rp = backtest(pfc, pdata; horizon=3, windows=3)
        for k in 1:3
            o = only(unique(rp.forecasts.origin[rp.forecasts.fold .== k]))
            keep = pdata.ds .<= o
            f = fit(pfc, (unique_id=pdata.unique_id[keep], ds=pdata.ds[keep],
                          y=pdata.y[keep]))
            @test rp.forecasts.y_hat[rp.forecasts.fold .== k] == forecast(f, 3).y_hat
        end

        # A levels fold fits through fit; y_hat must agree.
        rl = backtest(csp, data; horizon=3, windows=2, levels=(80,))
        rq = backtest(csp, data; horizon=3, windows=2)
        @test rl.forecasts.y_hat == rq.forecasts.y_hat
    end
end

@testset "SplitConformal type and level arithmetic" begin
    C = MachineLearningForecast
    fix = "Pass e.g. SplitConformal(holdout=60)."

    @testset "constructor" begin
        @test C.SplitConformal(holdout=9).holdout == 9
        @test C.SplitConformal(holdout=60).holdout == 60
        s = C.SplitConformal(holdout=Int8(60))
        @test s.holdout === 60
        @test fieldtype(C.SplitConformal, :holdout) === Int
        @test sprint(show, C.SplitConformal(holdout=60)) == "SplitConformal(holdout=60)"
    end

    @testset "constructor errors" begin
        @test_throws ArgumentError(
            "SplitConformal needs holdout, the number of last rows of each series " *
            "that measure forecast errors. It has no default. Pass e.g. " *
            "SplitConformal(holdout=60), which is enough for a 95 % interval up to " *
            "12 steps ahead (it needs holdout >= 50 there).") C.SplitConformal()
        @test_throws ArgumentError(
            "SplitConformal takes the keyword holdout, got the positional " *
            "argument 60. $fix") C.SplitConformal(60)
        wf = "Pass the number of last rows that measure errors, e.g. " *
             "SplitConformal(holdout=60)."
        whole = "SplitConformal holdout must be a whole number"
        @test_throws ArgumentError("$whole, got 2.5. $wf") C.SplitConformal(holdout=2.5)
        @test_throws ArgumentError("$whole, got \"a\". $wf") C.SplitConformal(
            holdout="a")
        @test_throws ArgumentError("$whole, not the Bool true. $wf") C.SplitConformal(
            holdout=true)
        for bad in (8, 0, -3)
            @test_throws ArgumentError(
                "SplitConformal holdout must be a whole number >= 9 (the last rows " *
                "of each series that measure forecast errors; 9 errors give an " *
                "80 % interval at one step), got $bad. $fix") C.SplitConformal(
                holdout=bad)
        end
    end

    @testset "_need_errors" begin
        for (L, k) in ((50, 3), (80, 9), (90, 19), (95, 39), (99, 199),
                       (99.5, 399), (99.9, 1999), (1.5, 2))
            @test C._need_errors(Float64(L)) == k
        end
        @test C._need_errors(100 * (1 - 0.2)) == 9
        @test C._need_errors(80.00000000000001) == 9
        # Rounds to 95.0000 at four decimals, so 39.
        @test C._need_errors(94.99999999999999) == 39
        @test C._need_errors(99.999999) == 2_000_000 - 1
    end

    @testset "_max_level_text" begin
        for (K, s) in ((9, "80.00"), (20, "90.47"), (29, "93.33"), (39, "95.00"),
                       (40, "95.12"))
            @test C._max_level_text(K) == s
        end
        for K in 9:400
            @test C._need_errors(parse(Float64, C._max_level_text(K))) <= K
        end
    end

    @testset "_depth" begin
        @test C._depth(Recursive(), 60) == 52
        @test C._depth(Recursive(), 9) == 1
        @test C._depth(Direct(12), 60) == 12
        @test C._depth(Direct(12), 15) == 7
        @test C._depth(Direct(12), 20) == 12
        @test C._depth(Direct(12), 21) == 12
        @test C._depth(Direct(12), 19) == 11
    end

    @testset "_support_message" begin
        m = C._support_message
        lvl = "levels=(95,) needs at least 39 calibration errors per step, but " *
              "intervals=SplitConformal(holdout=40) gives 29 at forecast horizon 12 " *
              "(the origins that have 12 actuals), which supports central levels " *
              "up to 93.33. "
        @test m(40, 12, 32, [95.0], "forecast horizon") ==
              lvl * "Pass levels <= 93.33, forecast at most 2 steps, or fit with " *
                    "holdout >= 50."
        @test m(40, 12, 32, [95.0], "backtest horizon") ==
              replace(lvl, "forecast horizon" => "backtest horizon") *
              "Pass levels <= 93.33, use horizon <= 2, or fit with holdout >= 50."
        @test m(40, 12, 32, [80.0, 95.0, 90.0], "forecast horizon") ==
              m(40, 12, 32, [95.0], "forecast horizon")
        @test m(40, 12, 32, [80.0, 90.0], "forecast horizon") === nothing
        @test m(40, 12, 32, [99.5], "forecast horizon") ==
              "levels=(99.5,) needs at least 399 calibration errors per step, but " *
              "intervals=SplitConformal(holdout=40) gives 29 at forecast horizon 12 " *
              "(the origins that have 12 actuals), which supports central levels up " *
              "to 93.33. Pass levels <= 93.33, or fit with holdout >= 410."
        # c < need drops the "at most" clause.
        @test m(30, 1, 22, [95.0], "forecast horizon") ==
              "levels=(95,) needs at least 39 calibration errors per step, but " *
              "intervals=SplitConformal(holdout=30) gives 30 at forecast horizon 1 " *
              "(the origins that have 1 actual), which supports central levels up " *
              "to 93.54. Pass levels <= 93.54, or fit with holdout >= 39."
        # c == need still allows one step.
        @test m(39, 1, 31, [95.0], "forecast horizon") === nothing
        @test m(39, 2, 31, [95.0], "forecast horizon") ==
              "levels=(95,) needs at least 39 calibration errors per step, but " *
              "intervals=SplitConformal(holdout=39) gives 38 at forecast horizon 2 " *
              "(the origins that have 2 actuals), which supports central levels up " *
              "to 94.87. Pass levels <= 94.87, forecast at most 1 step, or fit " *
              "with holdout >= 40."

        dep = "needs at least 9 calibration errors per step, but " *
              "intervals=SplitConformal(holdout=60) gives 6 at that horizon (the " *
              "origins that have 55 actuals). "
        @test m(60, 55, 52, nothing, "forecast_samples horizon") ==
              "forecast_samples horizon 55 " * dep *
              "Forecast at most 52 steps, or fit with holdout >= 63."
        @test m(60, 55, 52, Float64[], "forecast horizon") ==
              "forecast horizon 55 " * dep *
              "Forecast at most 52 steps, or fit with holdout >= 63."
        @test m(60, 55, 52, Float64[], "backtest horizon") ==
              "backtest horizon 55 " * dep *
              "Use horizon <= 52, or fit with holdout >= 63."
        # A horizon beyond holdout has no errors, never fewer.
        @test m(60, 70, 52, nothing, "forecast horizon") ==
              "forecast horizon 70 needs at least 9 calibration errors per step, " *
              "but intervals=SplitConformal(holdout=60) gives 0 at that horizon " *
              "(the origins that have 70 actuals). Forecast at most 52 steps, or " *
              "fit with holdout >= 78."
        # Depth wins over the level rule.
        @test m(60, 55, 52, [95.0], "forecast horizon") ==
              m(60, 55, 52, nothing, "forecast horizon")
        @test m(60, 53, 52, [50.0], "forecast horizon") !== nothing

        # Nothing is refused at horizon 1 at a supported level.
        @test m(9, 1, 1, [80.0], "forecast horizon") === nothing
        @test m(39, 1, 31, [95.0], "backtest horizon") === nothing
        @test m(9, 1, 1, nothing, "forecast_samples horizon") === nothing
        @test m(9, 1, 1, Float64[], "backtest horizon") === nothing
        @test m(60, 52, 52, Float64[], "forecast horizon") === nothing
        # One step is singular in both messages.
        @test m(9, 2, 1, nothing, "forecast horizon") ==
              "forecast horizon 2 needs at least 9 calibration errors per step, " *
              "but intervals=SplitConformal(holdout=9) gives 8 at that horizon " *
              "(the origins that have 2 actuals). Forecast at most 1 step, or fit " *
              "with holdout >= 10."
    end
end

@testset "SplitConformal calibration" begin
    C = MachineLearningForecast
    LinAR, MeanModel = TestModels.LinAR, TestModels.MeanModel
    siv(c) = C.SplitConformal(holdout=c)
    mk(model; feats=FeatureSet(Lag(1)), strat=Recursive()) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1))
    # Dyadic values, so closed forms compare exactly.
    function mkdata(n)
        return (ds=collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(n - 1))),
                y=[8 + mod(7i, 16) / 4 + i / 8 for i in 1:n])
    end
    groups_of(fc, d) = C._series_groups(fc, C.normalize_table(d))
    cal(fc, d, c; kw...) = C._calibrate(siv(c), fc, groups_of(fc, d); kw...)
    # One origin's pseudo-state, as the calibration builds it.
    function pseudo(fc, p, o)
        yp = p.yraw[1:o]
        _, z, state = C._transform_table(fc, NamedTuple{(fc.target,)}((yp,)), yp)
        return C.SeriesState(p.id, p.yraw, p.t[o], p.t[1], o, z, state, 0)
    end

    @testset "pseudo-states through _forecast_states" begin
        n, c = 60, 12
        d = mkdata(n)
        feats = FeatureSet(Lag(1), RollingMean(3), Calendar(:dayofweek))
        for strat in (Recursive(), Direct(5)),
            model in (LinAR(0.5, 1.0), DecisionTreeRegressor(max_depth=3))

            fc = mk(model; feats, strat)
            depth = C._depth(strat, c)
            plain = fit(fc, (ds=d.ds[1:(n - c)], y=d.y[1:(n - c)]))
            usable, _ = C._prepare_all(fc, groups_of(fc, d))
            p = only(usable)

            pub = only(C._published_states(usable, [0]))
            full = only(fit(fc, d).series)
            for f in fieldnames(C.SeriesState)
                @test getfield(pub, f) == getfield(full, f)
                @test getfield(pub, f) == getfield(pseudo(fc, p, n), f)
            end

            os = (n - c):(n - 1)
            states = [pseudo(fc, p, o) for o in os]
            hs = [min(n - o, depth) for o in os]
            all_at_once = C._forecast_states(plain, states, hs,
                                             C._grids(states, fc.freq, hs), nothing)
            for k in eachindex(os)
                one = C._forecast_states(plain, [states[k]], [hs[k]],
                                         C._grids([states[k]], fc.freq, [hs[k]]), nothing)
                @test all_at_once[k] == only(one)
                fk = C.FittedForecaster(fc, plain.models, [states[k]], plain.feature_names)
                @test all_at_once[k] == forecast(fk, hs[k]).y_hat
            end

            head, E, st = cal(fc, d, c)
            @test all(getfield(only(st), f) == getfield(pub, f)
                      for f in fieldnames(C.SeriesState))
            @test forecast(head, depth).y_hat == forecast(plain, depth).y_hat
            @test E[1][1, :] == d.y[(n - c + 1):(n - c + depth)] .-
                                forecast(plain, depth).y_hat
            for (k, o) in enumerate(os)
                @test E[1][k, 1:hs[k]] == d.y[(o + 1):(o + hs[k])] .- all_at_once[k]
            end
        end
    end

    @testset "closed forms" begin
        n, c = 40, 12
        d = mkdata(n)
        y = d.y
        depth = C._depth(Recursive(), c)
        head, E = cal(mk(MeanModel()), d, c)
        @test head isa C.FittedForecaster
        @test E isa Vector{Matrix{Float64}}
        m = sum(y[2:(n - c)]) / (n - c - 1)
        for i in 1:c, j in 1:min(depth, c - i + 1)
            @test only(E)[i, j] == y[n - c + i - 1 + j] - m
        end

        _, E = cal(mk(MeanModel(); strat=Direct(3)), d, c)
        for i in 1:c, j in 1:min(3, c - i + 1)
            @test only(E)[i, j] == y[n - c + i - 1 + j] - sum(y[(1 + j):(n - c)]) /
                                                          (n - c - j)
        end

        _, E = cal(mk(LinAR(0.5, 1.0)), d, c)
        for i in 1:c
            f = y[n - c + i - 1]
            for j in 1:min(depth, c - i + 1)
                f = 0.5 * f + 1.0
                @test only(E)[i, j] == y[n - c + i - 1 + j] - f
            end
        end
    end

    @testset "triangle shape and the NaN corner" begin
        c = 12
        for strat in (Recursive(), Direct(3), Direct(20))
            _, E = cal(mk(MeanModel(); strat), mkdata(40), c)
            depth = C._depth(strat, c)
            E1 = only(E)
            @test size(E1) == (c, depth)
            for i in 1:c, j in 1:depth
                @test isnan(E1[i, j]) == (j > c - i + 1)
            end
            @test all(isfinite, E1[1, :])
            @test count(isnan, E1) == sum(max(depth - (c - i + 1), 0) for i in 1:c)
        end
        @test C._depth(Direct(20), 12) == 4
        @test C._depth(Direct(3), 12) == 3
    end

    @testset "Direct head rows and the lower holdout clause" begin
        pre = "intervals=SplitConformal(holdout=9) leaves 5 of the 14 rows to train " *
              "the calibration model, 4 after the feature set's 1 history rows, but " *
              "Direct(5) needs 5. "
        fc = mk(MeanModel(); strat=Direct(5))
        _, E = cal(fc, mkdata(15), 9)
        @test size(only(E)) == (9, 1)
        @test_throws ArgumentError(pre * "Use Direct(4) or a shorter max_horizon, or " *
                                   "provide at least 15 rows.") cal(fc, mkdata(14), 9)
        @test_throws ArgumentError(
            "intervals=SplitConformal(holdout=25) leaves 5 of the 30 rows to train the " *
            "calibration model, 4 after the feature set's 1 history rows, but Direct(5) " *
            "needs 5. Lower holdout to 24 or less, use Direct(4) or a shorter " *
            "max_horizon, or provide at least 31 rows.") cal(fc, mkdata(30), 25)
        _, E = cal(fc, mkdata(30), 24)
        @test size(only(E)) == (24, 5)
    end

    @testset "too few rows" begin
        fc = mk(MeanModel(); feats=FeatureSet(Lag(13)))
        @test_throws ArgumentError(
            "the data has 100 rows. intervals=SplitConformal(holdout=90) holds out the " *
            "last 90 rows to measure errors and leaves 10 to train the calibration " *
            "model, but the feature set needs more than 13 history rows. Provide at " *
            "least 104 rows or lower holdout to 86 or less.") cal(fc, mkdata(100), 90)
        @test_throws ArgumentError(
            "the data has 103 rows. intervals=SplitConformal(holdout=90) holds out the " *
            "last 90 rows to measure errors and leaves 13 to train the calibration " *
            "model, but the feature set needs more than 13 history rows. Provide at " *
            "least 104 rows or lower holdout to 89 or less.") cal(fc, mkdata(103), 90)
        _, E = cal(fc, mkdata(104), 90)
        @test size(only(E)) == (90, 82)
        # A holdout below 9 is invalid, so the clause is dropped.
        fc6 = mk(MeanModel(); feats=FeatureSet(Lag(6)))
        @test_throws ArgumentError(
            "the data has 14 rows. intervals=SplitConformal(holdout=9) holds out the " *
            "last 9 rows to measure errors and leaves 5 to train the calibration " *
            "model, but the feature set needs more than 6 history rows. Provide at " *
            "least 16 rows.") cal(fc6, mkdata(14), 9)
        @test_throws ArgumentError(
            "the data has 5 rows. intervals=SplitConformal(holdout=9) holds out the " *
            "last 9 rows to measure errors and leaves 0 to train the calibration " *
            "model, but the feature set needs more than 6 history rows. Provide at " *
            "least 16 rows.") cal(fc6, mkdata(5), 9)
        @test_throws ArgumentError(
            "the data has 16 rows. intervals=SplitConformal(holdout=10) holds out the " *
            "last 10 rows to measure errors and leaves 6 to train the calibration " *
            "model, but the feature set needs more than 6 history rows. Provide at " *
            "least 17 rows or lower holdout to 9 or less.") cal(fc6, mkdata(16), 10)
        # The row check runs before any fit.
        TestModels.FIT_COUNT[] = 0
        @test_throws ArgumentError cal(mk(TestModels.CountFit(); feats=FeatureSet(Lag(6))),
                                       mkdata(14), 9)
        @test TestModels.FIT_COUNT[] == 0
    end

    @testset "shortest series, horizon 1 and no history" begin
        # One training row in the head.
        n, c = 11, 9
        d = mkdata(n)
        head, E = cal(mk(MeanModel()), d, c)
        @test size(only(E)) == (9, 1)
        @test head.n_train == 2
        for i in 1:c
            @test only(E)[i, 1] == d.y[n - c + i] - d.y[2]
        end
        # c = 9 gives depth 1.
        _, E = cal(mk(MeanModel()), mkdata(60), 9)
        @test size(only(E)) == (9, 1)
        # No history: the head uses every row before the cutoff.
        fc0 = mk(MeanModel(); feats=FeatureSet(Calendar(:dayofweek)))
        d0 = mkdata(10)
        _, E = cal(fc0, d0, 9)
        @test size(only(E)) == (9, 1)
        @test only(E)[1, 1] == d0.y[2] - d0.y[1]
        @test_throws ArgumentError("the data has 9 rows. intervals=SplitConformal(" *
            "holdout=9) holds out the last 9 rows to measure errors and leaves 0 to " *
            "train the calibration model, but the feature set needs more than 0 history " *
            "rows. Provide at least 10 rows.") cal(fc0, mkdata(9), 9)
        # A long series.
        _, E = cal(mk(MeanModel()), mkdata(300), 60)
        @test size(only(E)) == (60, 52)
        @test all(isfinite, only(E)[1, :])
    end

    @testset "leakage" begin
        n, c = 40, 12
        for model in (LinAR(0.5, 1.0), DecisionTreeRegressor(max_depth=3)),
            strat in (Recursive(), Direct(3))

            fc = mk(model; feats=FeatureSet(Lag(1), Lag(2)), strat)
            d = mkdata(n)
            _, E0 = cal(fc, d, c)
            E0 = only(E0)
            depth = size(E0, 2)
            for r in (n - c + 3, n)
                y2 = copy(d.y)
                y2[r] += 8.0
                _, E1 = cal(fc, (ds=d.ds, y=y2), c)
                E1 = only(E1)
                for i in 1:c, j in 1:depth
                    o = n - c + i - 1
                    isnan(E0[i, j]) && continue
                    if o < r && o + j != r
                        @test E1[i, j] == E0[i, j]
                    elseif o < r
                        @test E1[i, j] ≈ E0[i, j] + 8.0
                    end
                end
            end
        end
        # Rows up to the cutoff train the head.
        d = mkdata(n)
        y3 = copy(d.y)
        y3[5] += 8.0
        fc = mk(MeanModel())
        @test only(cal(fc, (ds=d.ds, y=y3), c)[2]) != only(cal(fc, d, c)[2])
    end

    @testset "data and model are not mutated, results repeat" begin
        d = mkdata(40)
        d0 = deepcopy(d)
        for model in (LinAR(0.5, 1.0), DecisionTreeRegressor(max_depth=3))
            fc = mk(model)
            proto = deepcopy(model)
            tbl = C.normalize_table(d)
            groups = C._series_groups(fc, tbl)
            h1, E1 = C._calibrate(siv(12), fc, groups)
            h2, E2 = C._calibrate(siv(12), fc, groups)
            @test isequal(E1, E2)
            @test forecast(h1, 3).y_hat == forecast(h2, 3).y_hat
            @test d == d0
            @test tbl == C.normalize_table(d0)
            @test TestModels.MMI.is_same_except(fc.model, proto)
            @test h1.models[1].model !== fc.model
        end
    end

    @testset "chunk sizes agree" begin
        for model in (LinAR(0.5, 1.0), DecisionTreeRegressor(max_depth=3)),
            strat in (Recursive(), Direct(4))

            fc = mk(model; feats=FeatureSet(Lag(1), RollingMean(3)), strat)
            d = mkdata(50)
            c = 20
            ref = cal(fc, d, c)
            for chunk in (1, 7, c, 1000)
                h, E = cal(fc, d, c; chunk)
                @test isequal(E, ref[2])
                @test forecast(h, 4).y_hat == forecast(ref[1], 4).y_hat
            end
        end
    end

    @testset "one fit, no final fit" begin
        fc = mk(TestModels.CountFit())
        TestModels.FIT_COUNT[] = 0
        cal(fc, mkdata(40), 12)
        @test TestModels.FIT_COUNT[] == 1
        TestModels.FIT_COUNT[] = 0
        cal(mk(TestModels.CountFit(); strat=Direct(3)), mkdata(40), 12)
        @test TestModels.FIT_COUNT[] == 3
    end

    @testset "real learners" begin
        n, c = 80, 20
        d = mkdata(n)
        feats = FeatureSet(Lag(1), Lag(2), RollingMean(3), Calendar(:dayofweek))
        for model in (EvoTreeRegressor(nrounds=5), DecisionTreeRegressor(max_depth=3)),
            strat in (Recursive(), Direct(5))

            _, E = cal(mk(model; feats, strat), d, c)
            E = only(E)
            depth = C._depth(strat, c)
            @test size(E) == (c, depth)
            for i in 1:c, j in 1:depth
                @test isnan(E[i, j]) == (j > c - i + 1)
                j <= c - i + 1 && @test isfinite(E[i, j])
            end
        end
    end

    @testset "a non-finite forecast is refused" begin
        n = 60
        d = mkdata(n)
        msg(o, j, v) = "the calibration forecast from origin $(d.ds[o]) (row $o) is $v " *
                       "at step $j. A non-finite forecast cannot give an interval. " *
                       "Check the model's hyperparameters and the training data; a " *
                       "smaller holdout calibrates fewer steps."
        for chunk in (1, 5, 12)
            @test_throws ArgumentError(msg(n - 12, 1, NaN)) cal(
                mk(TestModels.NaNModel()), d, 12; chunk)
        end
        # Recursive: NaN once the lag passes the limit, at step 7.
        c = 16
        dl = (ds=d.ds, y=Float64.(1:n))
        for chunk in (1, 5, c)
            @test_throws ArgumentError(msg(n - c, 7, NaN)) cal(
                mk(TestModels.NaNAbove(Float64(n - c + 5))), dl, c; chunk)
        end
        # Direct: the step-7 model has L - 6 training targets.
        fc = mk(TestModels.NaNAtLength(n - c - 1 - 6); strat=Direct(8))
        @test_throws ArgumentError(msg(n - c, 7, NaN)) cal(fc, d, c)
        # c = 13 gives depth 5, so the step-7 model is never read.
        fc = mk(TestModels.NaNAtLength(n - 13 - 1 - 6); strat=Direct(8))
        _, E = cal(fc, d, 13)
        @test size(only(E)) == (13, 5)
        @test all(isfinite, E[1][1, :])
    end

    @testset "input errors come first" begin
        fc = mk(MeanModel())
        d = mkdata(40)
        @test_throws "time column :ds is empty" cal(fc, (ds=Date[], y=Float64[]), 12)
        @test_throws "time column :ds has 1 gap" cal(
            fc, (ds=d.ds[[1:10; 12:40]], y=d.y[[1:10; 12:40]]), 12)
        @test_throws "time column :ds not found in the data" cal(fc, (t=d.ds, y=d.y), 12)
        @test_throws "has element type String" cal(
            fc, (ds=string.(d.ds), y=d.y), 12)
        ym = Vector{Union{Missing,Float64}}(d.y)
        ym[30] = missing
        @test_throws ArgumentError(
            "target column :y contains missing values (first at row 30). Impute or " *
            "drop missing targets before fitting.") cal(fc, (ds=d.ds, y=ym), 12)
        @test_throws "target column :y not found" cal(fc, (ds=d.ds, z=d.y), 12)
        @test_throws "has element type String; expected a Real" cal(
            fc, (ds=d.ds, y=string.(d.y)), 12)
        # Too short with a missing target: our message comes first.
        ym = Vector{Union{Missing,Float64}}(d.y[1:12])
        ym[3] = missing
        @test_throws ArgumentError(
            "the data has 12 rows. intervals=SplitConformal(holdout=12) holds out the " *
            "last 12 rows to measure errors and leaves 0 to train the calibration " *
            "model, but the feature set needs more than 1 history rows. Provide at " *
            "least 14 rows or lower holdout to 10 or less.") cal(
            fc, (ds=d.ds[1:12], y=ym), 12)
    end

    @testset "errors from inside the calibration are named" begin
        n, c = 60, 12
        d = mkdata(n)
        pre = "intervals=SplitConformal(holdout=12), "
        fc = mk(TestModels.ThrowIn(:predict, false))
        @test_throws ArgumentError(
            pre * "calibrating origins at rows 48 to 54: boom") cal(fc, d, c; chunk=7)
        @test_throws ArgumentError(
            pre * "calibrating origins at rows 48 to 59: boom") cal(fc, d, c; chunk=c)
        @test_throws ArgumentError(
            pre * "calibrating origins at rows 48 to 48: boom") cal(fc, d, c; chunk=1)
        @test_throws ArgumentError(
            pre * "fitting the calibration model on the first 48 rows: boom") cal(
            mk(TestModels.ThrowIn(:fit, false)), d, c)
        # Other exceptions pass through unwrapped.
        @test_throws ErrorException cal(mk(TestModels.ThrowIn(:predict, true)), d, c)
        @test_throws ErrorException cal(mk(TestModels.ThrowIn(:fit, true)), d, c)
    end

    @testset "warnings appear once" begin
        fc = mk(TestModels.Clamped(0))
        @test_logs (:warn, "k must be ≥ 1; resetting k to 1.") cal(fc, mkdata(40), 12)
    end
end
