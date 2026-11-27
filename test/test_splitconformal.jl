# Seeded, so two fits of one tree are identical.
dtree() = DecisionTreeRegressor(max_depth=3, rng=StableRNG(1))

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
            model in (LinAR(0.5, 1.0), dtree())

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
              "the calibration model, 4 after the feature set's 1 history row, but " *
              "Direct(5) needs 5. "
        fc = mk(MeanModel(); strat=Direct(5))
        _, E = cal(fc, mkdata(15), 9)
        @test size(only(E)) == (9, 1)
        @test_throws ArgumentError(pre * "Use Direct(4) or a shorter max_horizon, or " *
                                   "provide at least 15 rows.") cal(fc, mkdata(14), 9)
        @test_throws ArgumentError(
            "intervals=SplitConformal(holdout=25) leaves 5 of the 30 rows to train the " *
            "calibration model, 4 after the feature set's 1 history row, but Direct(5) " *
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
        for model in (LinAR(0.5, 1.0), dtree()),
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
        for model in (LinAR(0.5, 1.0), dtree())
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
        for model in (LinAR(0.5, 1.0), dtree()),
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
        for model in (EvoTreeRegressor(nrounds=5), dtree()),
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
            "model, but the feature set needs more than 1 history row. Provide at " *
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

@testset "SplitConformal transforms and exogenous" begin
    C = MachineLearningForecast
    LinAR, MeanModel, EchoColumn = TestModels.LinAR, TestModels.MeanModel,
                                   TestModels.EchoColumn
    siv(c) = C.SplitConformal(holdout=c)
    mk(model; feats=FeatureSet(Lag(1)), strat=Recursive(), transform=NoTransform()) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1), transform)
    # Dyadic values, so closed forms compare exactly.
    function mkdata(n)
        return (ds=collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(n - 1))),
                y=[8 + mod(7i, 16) / 4 + i / 8 for i in 1:n],
                x=[100 + mod(5i, 8) / 2 + i / 4 for i in 1:n])
    end
    groups_of(fc, d) = C._series_groups(fc, C.normalize_table(d))
    cal(fc, d, c; kw...) = C._calibrate(siv(c), fc, groups_of(fc, d); kw...)
    usable_of(fc, d) = only(first(C._prepare_all(fc, groups_of(fc, d))))
    caught(f) = try f(); nothing catch err err end

    @testset "built-in transforms: original scale and prefix states" begin
        n, c = 60, 12
        d = mkdata(n)
        y = d.y
        trs = (MeanScale(), LastScale(3), Deseasonalize(4), LogRatio(3), Difference(2),
               TransformChain(LastScale(3), Difference(2)))
        depth = C._depth(Recursive(), c)
        for tr in trs, model in (LinAR(0.5, 1.0), dtree())
            fc = mk(model; transform=tr)
            head, E, st = cal(fc, d, c)
            p = usable_of(fc, d)
            @test only(st).step0 == only(head.series).step0 == 0
            for i in 1:c
                o = n - c + i - 1
                ps = C._pseudo_state(siv(c), fc, p, o, 0)
                z, state = C.fit_transform(tr, y[1:o])
                @test ps.z_history == z
                @test ps.transform_state == state
                @test (ps.n_train, ps.t_last, ps.t_start) == (o, d.ds[o], d.ds[1])
                h = min(c - i + 1, depth)
                fk = C.FittedForecaster(fc, head.models, [ps], head.feature_names)
                @test only(E)[i, 1:h] == y[(o + 1):(o + h)] .- forecast(fk, h).y_hat
                @test all(isnan, only(E)[i, (h + 1):end])
            end
        end
        # Closed forms: ŷ_j = a ŷ_(j-1) + b on the original scale.
        _, E = cal(mk(LinAR(0.5, 1.0); transform=Difference(1)), d, c)
        for i in 1:c
            o = n - c + i - 1
            f, step = y[o], 0.5 * (y[o] - y[o - 1]) + 1.0
            for j in 1:min(depth, c - i + 1)
                f += step
                @test only(E)[i, j] == y[o + j] - f
                step = 0.5 * step + 1.0
            end
        end
        for (tr, m) in ((MeanScale(), o -> sum(y[1:o]) / o),
                        (LastScale(3), o -> sum(y[(o - 2):o]) / 3))
            _, E = cal(mk(LinAR(0.5, 1.0); transform=tr), d, c)
            for i in 1:c
                o = n - c + i - 1
                f = y[o]
                for j in 1:min(depth, c - i + 1)
                    f = 0.5 * f + m(o)
                    @test only(E)[i, j] ≈ y[o + j] - f
                end
            end
        end
    end

    @testset "rows after the origin never reach its prediction" begin
        n, c = 60, 12
        d = mkdata(n)
        r = n - c + 5
        for tr in (MeanScale(), LastScale(3), Deseasonalize(4), LogRatio(3),
                   Difference(2), TransformChain(LastScale(3), Difference(2)))
            fc = mk(LinAR(0.5, 1.0); transform=tr)
            E0 = only(cal(fc, d, c)[2])
            y2 = copy(d.y)
            y2[r] += 8.0
            E1 = only(cal(fc, (ds=d.ds, y=y2), c)[2])
            for i in 1:c, j in 1:size(E0, 2)
                o = n - c + i - 1
                isnan(E0[i, j]) && continue
                if o < r && o + j != r
                    @test E1[i, j] == E0[i, j]
                elseif o < r
                    @test E1[i, j] ≈ E0[i, j] + 8.0
                end
            end
            @test E1[5, 1] != E0[5, 1]
        end
    end

    @testset "the head needs the transform's dropped rows" begin
        fc = mk(MeanModel(); transform=Difference(3))
        _, E = cal(fc, mkdata(17), 12)
        @test size(only(E)) == (12, 4)
        @test_throws ArgumentError(
            "the data has 16 rows. intervals=SplitConformal(holdout=12) holds out the " *
            "last 12 rows to measure errors and leaves 4 to train the calibration " *
            "model, but the feature set and transform need more than 4 rows. Provide " *
            "at least 17 rows or lower holdout to 11 or less.") cal(fc, mkdata(16), 12)
        TestModels.FIT_COUNT[] = 0
        @test_throws "the feature set and transform need" cal(
            mk(TestModels.CountFit(); transform=Difference(3)), mkdata(13), 9)
        @test TestModels.FIT_COUNT[] == 0
        # Direct names the transform's rows in its own message.
        fd = mk(MeanModel(); strat=Direct(5), transform=Difference(3))
        _, E = cal(fd, mkdata(18), 9)
        @test size(only(E)) == (9, 1)
        @test_throws ArgumentError(
            "intervals=SplitConformal(holdout=9) leaves 8 of the 17 rows to train the " *
            "calibration model, 4 after the feature set's 1 history row and the " *
            "transform's 3 dropped rows, but Direct(5) needs 5. Use Direct(4) or a " *
            "shorter max_horizon, or provide at least 18 rows.") cal(fd, mkdata(17), 9)
        @test_throws ArgumentError(
            "intervals=SplitConformal(holdout=25) leaves 5 of the 30 rows to train the " *
            "calibration model, 1 after the feature set's 1 history row and the " *
            "transform's 3 dropped rows, but Direct(5) needs 5. Lower holdout to 21 " *
            "or less, use Direct(1) or a shorter max_horizon, or provide at least 34 " *
            "rows.") cal(fd, mkdata(30), 25)
        _, E = cal(fd, mkdata(30), 21)
        @test size(only(E)) == (21, 5)
    end

    @testset "errors from the prefix state name one row" begin
        n, c = 60, 12
        d = mkdata(n)
        # The prefix to row 51 sums to zero; no longer one does.
        y = copy(d.y)
        y[51] = -sum(y[1:50])
        pre = "intervals=SplitConformal(holdout=12), "
        fc = mk(LinAR(0.5, 1.0); transform=MeanScale())
        for chunk in (1, 5, c)
            @test_throws ArgumentError(
                pre * "calibrating the origin at row 51: MeanScale() cannot scale a " *
                "series whose training mean is 0.0. Use a transform that does not " *
                "divide by it, e.g. NoTransform() or Difference(1).") cal(
                fc, (ds=d.ds, y=y), c; chunk)
        end
        # A model error names a range, never one row.
        fp = mk(TestModels.ThrowIn(:predict, false); transform=MeanScale())
        for (chunk, rows) in ((1, "48 to 48"), (7, "48 to 54"), (c, "48 to 59"))
            err = caught(() -> cal(fp, d, c; chunk))
            @test err isa ArgumentError
            @test err.msg == pre * "calibrating origins at rows $rows: boom"
            @test !occursin("the origin at row", err.msg)
        end
        # A series the transform refuses fails before calibrating.
        y = copy(d.y)
        y[50] = -1.0
        err = caught(() -> cal(mk(LinAR(0.5, 1.0); transform=LogRatio(3)),
                               (ds=d.ds, y=y), c))
        @test err isa ArgumentError
        @test occursin("LogRatio(3) takes logarithms of ratios", err.msg)
        @test !occursin("calibrating", err.msg)
        # Other exceptions pass through unwrapped.
        @test_throws ErrorException cal(
            mk(TestModels.ThrowIn(:predict, true); transform=MeanScale()), d, c)
    end

    @testset "origin o reads exogenous row o + s" begin
        n, c = 40, 12
        d = mkdata(n)
        y, x = d.y, d.x
        feats = FeatureSet(Lag(1), Exogenous(:x))
        depth = C._depth(Recursive(), c)
        fc = mk(EchoColumn(:x); feats)
        E0 = only(cal(fc, d, c)[2])
        for i in 1:c, j in 1:size(E0, 2)
            o = n - c + i - 1
            @test isnan(E0[i, j]) == (j > c - i + 1)
            j <= c - i + 1 && @test E0[i, j] == y[o + j] - x[o + j]
        end
        # The rows are the ones forecast builds from `future`.
        usable = [usable_of(fc, d)]
        exog = C._exogenous_history(fc, usable, groups_of(fc, d))
        @test length(only(exog)) == n
        for o in (n - c, n - 3), h in (1, min(3, n - o))
            ps = C._pseudo_state(siv(c), fc, only(usable), o, 0)
            future = (ds=d.ds[(o + 1):(o + h)], x=x[(o + 1):(o + h)])
            want = C._exogenous_rows(fc, [ps], C._grids([ps], fc.freq, [h]), future, [h])
            got = Vector{NamedTuple}[only(exog)[(o + 1):(o + h)]]
            @test got == want
            @test typeof(got) == typeof(want)
        end
        # Rows beyond an origin's reach do not move it.
        r = n - c + 6
        x2 = copy(x)
        x2[r] += 8.0
        E1 = only(cal(fc, (ds=d.ds, y=y, x=x2), c)[2])
        for i in 1:c, j in 1:depth
            o = n - c + i - 1
            isnan(E0[i, j]) && continue
            @test E1[i, j] == (o + j == r ? E0[i, j] - 8.0 : E0[i, j])
        end
        # Direct reads the target's row too.
        _, ED = cal(mk(EchoColumn(:x); feats, strat=Direct(3)), d, c)
        for i in 1:c, j in 1:min(3, c - i + 1)
            o = n - c + i - 1
            @test only(ED)[i, j] == y[o + j] - x[o + j]
        end
        # With a transform: the changes add up from the last value.
        _, ET = cal(mk(EchoColumn(:x); feats, transform=Difference(1)), d, c)
        for i in 1:c, j in 1:min(depth, c - i + 1)
            o = n - c + i - 1
            @test only(ET)[i, j] == y[o + j] - (y[o] + sum(x[(o + 1):(o + j)]))
        end
    end

    @testset "missing exogenous values" begin
        n, c = 40, 12
        d = mkdata(n)
        feats = FeatureSet(Lag(1), Exogenous(:x))
        fc = mk(EchoColumn(:x); feats)
        miss(rows) = (xm = Vector{Union{Missing,Float64}}(d.x);
                      xm[rows] .= missing; (ds=d.ds, y=d.y, x=xm))
        msg(r) = "feature column :x has a missing value at row $r (time $(d.ds[r])). " *
                 "Missing values in exogenous columns are not supported; impute or " *
                 "drop them before fitting."
        # The first mh rows are never read.
        _, E = cal(fc, miss([1]), c)
        @test isequal(E, cal(fc, d, c)[2])
        @test fit(fc, miss([1])) isa C.FittedForecaster
        # Holdout row, head row, both: the plain fit message.
        for rows in ([n - 5], [20], [n - 5, 20], [1, n - 5], [n]), chunk in (1, 1000)
            data = miss(rows)
            r = minimum(filter(>(1), rows))
            want = caught(() -> fit(fc, data))
            @test want isa ArgumentError
            @test want.msg == msg(r)
            @test_throws ArgumentError(msg(r)) cal(fc, data, c; chunk)
        end
        @test_throws ArgumentError(msg(20)) C.build_training_frame(
            feats, C.normalize_table(miss([20])), :y, :ds)
        # A transform moves mh: the dropped row is read by nobody.
        ft = mk(EchoColumn(:x); feats, transform=Difference(1))
        _, E = cal(ft, miss([2]), c)
        @test isequal(E, cal(ft, d, c)[2])
        @test fit(ft, miss([2])) isa C.FittedForecaster
        @test_throws ArgumentError(msg(3)) cal(ft, miss([3]), c)
        @test_throws ArgumentError(msg(3)) fit(ft, miss([3]))
        # A missing column gives the message of a plain fit.
        absent = (ds=d.ds, y=d.y)
        want = caught(() -> fit(fc, absent))
        @test want.msg == "features contain Exogenous(:x) but column :x is not " *
                          "present in the training data. Available columns: ds, y. " *
                          "Add column :x to the data, or remove it from the " *
                          "Exogenous feature."
        @test_throws ArgumentError(want.msg) cal(fc, absent, c)
    end

    @testset "bad exogenous values in the holdout" begin
        n, c = 40, 12
        d = mkdata(n)
        fc = mk(EchoColumn(:x); feats=FeatureSet(Lag(1), Exogenous(:x)))
        with(v) = (xv = Vector{Any}(d.x); xv[n - 5] = v; (ds=d.ds, y=d.y, x=xv))
        pre = "intervals=SplitConformal(holdout=12), calibrating origins at rows " *
              "28 to 39: exogenous column :x has "
        err = caught(() -> cal(fc, with(Inf), c; chunk=1000))
        @test err isa ArgumentError
        @test startswith(err.msg, pre * "the infinite value Inf at time $(d.ds[n - 5]) ")
        err = caught(() -> cal(fc, with("a"), c; chunk=1000))
        @test err isa ArgumentError
        @test startswith(err.msg, pre * "non-numeric value \"a\" (type String).")
        # NaN reaches the model, so the finite check names it.
        @test_throws ArgumentError(
            "the calibration forecast from origin $(d.ds[31]) (row 31) is NaN at " *
            "step 4. A non-finite forecast cannot give an interval. Check the " *
            "model's hyperparameters and the training data; a smaller holdout " *
            "calibrates fewer steps.") cal(fc, with(NaN), c)
    end

    @testset "several exogenous columns scan like a plain fit" begin
        n, c = 40, 12
        d = mkdata(n)
        fc = mk(EchoColumn(:x); feats=FeatureSet(Lag(1), Exogenous(:x, :w)))
        w = Vector{Union{Missing,Float64}}(d.x ./ 2)
        xm = Vector{Union{Missing,Float64}}(d.x)
        w[15] = missing
        xm[n - 5] = missing
        data = (ds=d.ds, y=d.y, x=xm, w=w)
        want = caught(() -> fit(fc, data))
        @test want isa ArgumentError
        @test_throws ArgumentError(want.msg) cal(fc, data, c)
        _, E = cal(fc, (ds=d.ds, y=d.y, x=d.x, w=d.x ./ 2), c)
        @test isequal(E, cal(mk(EchoColumn(:x); feats=FeatureSet(Lag(1), Exogenous(:x))),
                             d, c)[2])
    end

    @testset "real learners with a transform and an exogenous feature" begin
        n, c = 80, 20
        d = mkdata(n)
        feats = FeatureSet(Lag(1), Lag(2), RollingMean(3), Exogenous(:x))
        tr = TransformChain(LastScale(3), Difference(1))
        for model in (EvoTreeRegressor(nrounds=5), dtree()),
            strat in (Recursive(), Direct(5))

            _, E = cal(mk(model; feats, strat, transform=tr), d, c)
            E = only(E)
            depth = C._depth(strat, c)
            @test size(E) == (c, depth)
            for i in 1:c, j in 1:depth
                @test isnan(E[i, j]) == (j > c - i + 1)
                j <= c - i + 1 && @test isfinite(E[i, j])
            end
        end
    end
end

@testset "SplitConformal panels" begin
    C = MachineLearningForecast
    LinAR, MeanModel, EchoColumn = TestModels.LinAR, TestModels.MeanModel,
                                   TestModels.EchoColumn
    D0 = Date(2020, 1, 1)
    day(d) = D0 + Day(d - 1)
    siv(c) = C.SplitConformal(holdout=c)
    mk(model; feats=FeatureSet(Lag(1)), strat=Recursive(), transform=NoTransform(),
       id=:unique_id) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1), id, transform)
    # Dyadic values, so closed forms compare exactly.
    yv(d, k) = 8 + mod(7d + 3k, 16) / 4 + d / 8
    xv(d, k) = 100 + mod(5d + k, 8) / 2 + d / 4
    ys(k, r) = [yv(d, k) for d in r]
    function mkpanel(spans...; x=false)
        ids, ds, y, xs = Any[], Date[], Float64[], Union{Missing,Float64}[]
        for (k, (id, r)) in enumerate(spans), d in r
            push!(ids, id)
            push!(ds, day(d))
            push!(y, yv(d, k))
            push!(xs, xv(d, k))
        end
        base = (unique_id=identity.(ids), ds=ds, y=y)
        return x ? merge(base, (x=xs,)) : base
    end
    shuf(d, seed) = (p = shuffle(StableRNG(seed), 1:length(d.ds));
                     NamedTuple{keys(d)}(map(v -> v[p], values(d))))
    groups_of(fc, d) = C._series_groups(fc, C.normalize_table(d))
    cal(fc, d, c; kw...) = C._calibrate(siv(c), fc, groups_of(fc, d); kw...)
    caught(f) = try f(); nothing catch err err end
    ids_of(f) = [s.id for s in f.series]
    function pseudo(fc, p, o, step0=0)
        yp = p.yraw[1:o]
        _, z, state = C._transform_table(fc, NamedTuple{(fc.target,)}((yp,)), yp)
        return C.SeriesState(p.id, yp, p.t[o], p.t[1], o, z, state, step0)
    end
    # The head's mean over the targets of these days.
    headmean(parts, mh) = (v = reduce(vcat, [ys(k, (first(r) + mh):last(r))
                                              for (k, r) in parts]);
                           sum(v) / length(v))
    # Series k over days r, against a head that predicts m.
    function eref(k, r, m, c, depth)
        E = fill(NaN, c, depth)
        for i in 1:c, j in 1:min(depth, c - i + 1)
            E[i, j] = yv(first(r) + length(r) - c + i - 2 + j, k) - m
        end
        return E
    end

    @testset "a one-series panel equals the single series" begin
        c = 12
        feats = FeatureSet(Lag(1), RollingMean(3))
        for model in (LinAR(0.5, 1.0), MeanModel(), dtree()),
            strat in (Recursive(), Direct(3))

            fp = mk(model; feats, strat)
            fs = mk(model; feats, strat, id=nothing)
            hp, Ep, sp = cal(fp, mkpanel("a" => 1:60), c)
            hs, Es, ss = C._calibrate(siv(c), fs, C._series_groups(
                fs, C.normalize_table((ds=[day(d) for d in 1:60], y=ys(1, 1:60)))))
            @test isequal(Ep, Es)
            @test forecast(hp, 3).y_hat == forecast(hs, 3).y_hat
            @test sp[1].step0 == ss[1].step0 == 0
            @test sp[1].y_history == ss[1].y_history
        end
    end

    @testset "the head trains on rows before the earliest holdout start" begin
        c, depth = 12, 4
        spans = ("a" => 1:60, "b" => 10:70, "c" => 1:55)
        d = mkpanel(spans...)
        # Holdout starts: days 49, 59, 44; T* = day 44, set by "c".
        head, E, st = cal(mk(MeanModel()), d, c)
        @test ids_of(head) == ["a", "b", "c"]
        @test [s.n_train for s in head.series] == [43, 34, 43]
        @test [s.t_last for s in head.series] == fill(day(43), 3)
        m = headmean([(1, 1:43), (2, 10:43), (3, 1:43)], 1)
        @test head.models[1].fitresult == m
        for (k, (_, r)) in enumerate(spans)
            @test isequal(E[k], eref(k, r, m, c, depth))
            @test st[k].n_train == length(r)
        end
        # The series that sets T* is cut at its own end.
        @test head.series[3].n_train == 55 - c
        for chunk in (1, 5)
            @test isequal(cal(mk(MeanModel()), d, c; chunk)[2], E)
        end
        # The boundary: the setter keeps mh + 1 head rows.
        d2 = mkpanel("a" => 1:14, "b" => 1:30)
        head, E, _ = cal(mk(MeanModel()), d2, 12)
        @test [s.n_train for s in head.series] == [2, 2]
        @test head.models[1].fitresult == headmean([(1, 1:2), (2, 1:2)], 1)
    end

    @testset "an aligned panel equals the per-series cut" begin
        c, depth = 12, 3
        d = mkpanel("a" => 1:70, "b" => 15:70, "c" => 30:70)
        feats = FeatureSet(Lag(1), Lag(2), RollingMean(3))
        keep = d.ds .<= day(70 - c)
        cut = NamedTuple{keys(d)}(map(v -> v[keep], values(d)))
        for model in (LinAR(0.5, 1.0), MeanModel(), dtree(),
                      EvoTreeRegressor(nrounds=5)),
            strat in (Recursive(), Direct(3))

            fc = mk(model; feats, strat)
            head, E, st = cal(fc, d, c)
            ref = fit(fc, cut)
            @test forecast(head, depth).y_hat == forecast(ref, depth).y_hat
            @test [s.step0 for s in head.series] == [s.step0 for s in ref.series]
            @test [s.n_train for s in head.series] == [s.n_train for s in ref.series]
            usable, _ = C._prepare_all(fc, groups_of(fc, d))
            refst = C._published_states(usable, [s.step0 for s in ref.series])
            for f in fieldnames(C.SeriesState)
                @test all(getfield(a, f) == getfield(b, f) for (a, b) in zip(st, refst))
            end
            fa = C.FittedForecaster(fc, head.models, st, head.feature_names)
            fb = C.FittedForecaster(fc, ref.models, refst, ref.feature_names)
            @test forecast(fa, depth).y_hat == forecast(fb, depth).y_hat
            for (k, p) in enumerate(usable), i in 1:c
                o = length(p.t) - c + i - 1
                h = min(c - i + 1, depth)
                ps = pseudo(fc, p, o, ref.series[k].step0)
                fk = C.FittedForecaster(fc, ref.models, [ps], ref.feature_names)
                @test E[k][i, 1:h] == p.yraw[(o + 1):(o + h)] .- forecast(fk, h).y_hat
            end
        end
    end

    @testset "no leakage across series" begin
        c, depth = 12, 4
        spans = ("a" => 1:60, "b" => 10:70, "c" => 1:55)
        d = mkpanel(spans...)
        # Rows from T* = day 44 on, and rows before it, per series.
        late = ("a" => 50, "b" => 60, "c" => 54)
        early = ("a" => 20, "b" => 30, "c" => 40)
        row(id, dd) = findfirst(i -> d.unique_id[i] == id && d.ds[i] == day(dd),
                                eachindex(d.ds))
        bump(r) = (y = copy(d.y); y[r] += 8.0; merge(d, (y=y,)))
        fm = mk(MeanModel())
        head0, E0, _ = cal(fm, d, c)
        for (id, dd) in late
            head1, E1, _ = cal(fm, bump(row(id, dd)), c)
            @test head1.models[1].fitresult == head0.models[1].fitresult
            for (k, (sid, r)) in enumerate(spans), i in 1:c, j in 1:depth
                o = length(r) - c + i - 1
                hit = sid == id && first(r) + o + j - 1 == dd
                @test isequal(E1[k][i, j], hit ? E0[k][i, j] + 8.0 : E0[k][i, j])
            end
        end
        for (id, dd) in early
            head1, _ = cal(fm, bump(row(id, dd)), c)
            @test head1.models[1].fitresult != head0.models[1].fitresult
        end
        # A history reader moves only where it reads the row.
        fl = mk(LinAR(0.5, 1.0))
        _, L0, _ = cal(fl, d, c)
        for (id, dd) in late
            _, L1, _ = cal(fl, bump(row(id, dd)), c)
            for (k, (sid, r)) in enumerate(spans), i in 1:c, j in 1:depth
                o = length(r) - c + i - 1
                day_o = first(r) + o - 1
                moved = sid == id && (day_o == dd || day_o + j == dd)
                isnan(L0[k][i, j]) || @test (L1[k][i, j] != L0[k][i, j]) == moved
            end
        end
        # A trained head does not see rows from T* on.
        ft = mk(dtree(); feats=FeatureSet(Lag(1), Lag(2)))
        h0 = cal(ft, d, c)[1]
        for (id, dd) in late
            h1 = cal(ft, bump(row(id, dd)), c)[1]
            @test forecast(h1, 3).y_hat == forecast(h0, 3).y_hat
        end
        @test any(early) do (id, dd)
            forecast(cal(ft, bump(row(id, dd)), c)[1], 3).y_hat != forecast(h0, 3).y_hat
        end
    end

    @testset "late-starting series" begin
        c, depth = 12, 4
        feats = FeatureSet(Lag(3))
        fc = mk(MeanModel(); feats)
        # T* = day 49; "d" has 2 rows before it, "e" none (mh = 3).
        spans = ("a" => 1:60, "b" => 5:60, "d" => 47:62, "e" => 52:67)
        d = mkpanel(spans...)
        head, E, st = @test_logs cal(fc, d, c)
        @test ids_of(head) == ["a", "b"]
        @test [s.n_train for s in head.series] == [48, 44]
        m = headmean([(1, 1:48), (2, 5:48)], 3)
        @test head.models[1].fitresult == m
        @test [s.id for s in st] == ["a", "b", "d", "e"]
        for (k, (_, r)) in enumerate(spans)
            @test isequal(E[k], eref(k, r, m, c, depth))
        end
        @test [s.step0 for s in st] == [0, 4, 46, 51]
        @test all(s -> s.step0 >= 0, st)
        @test [s.step0 for s in head.series] == [0, 4]
        @test [s.n_train for s in st] == [60, 56, 16, 16]
        fh = C.FittedForecaster(fc, head.models, st, head.feature_names)
        @test forecast(fh, 2).y_hat == fill(m, 8)
        # A head fit failure counts the head rows only.
        @test_throws ArgumentError(
            "intervals=SplitConformal(holdout=12), fitting the calibration model on " *
            "the 92 rows before $(day(49)): boom") cal(
            mk(TestModels.ThrowIn(:fit, false); feats), d, c)
    end

    @testset "shuffled rows, first-appearance order" begin
        c, depth = 12, 4
        spans = ("c" => 1:60, "a" => 10:70, "b" => 1:55)
        d = mkpanel(spans...)
        for model in (MeanModel(), LinAR(0.5, 1.0))
            fc = mk(model)
            head, E, st = cal(fc, d, c)
            @test [s.id for s in st] == ["c", "a", "b"] == ids_of(head)
            for seed in (1, 2)
                head2, E2, st2 = cal(fc, shuf(d, seed), c)
                @test sort(ids_of(head2)) == ["a", "b", "c"]
                for (k, s) in enumerate(st)
                    k2 = findfirst(x -> x.id == s.id, st2)
                    @test isequal(E2[k2], E[k])
                    @test st2[k2].step0 == s.step0
                    @test st2[k2].y_history == s.y_history
                end
            end
        end
        # E follows the published order, not the sorted ids.
        m = headmean([(1, 1:43), (2, 10:43), (3, 1:43)], 1)
        _, E, st = cal(mk(MeanModel()), d, c)
        for (k, (id, r)) in enumerate(spans)
            @test st[k].id == id
            @test isequal(E[k], eref(k, r, m, c, depth))
        end
    end

    @testset "step0 of the pseudo-states" begin
        c = 12
        # T* = day 69; "c" has one row before it: not in the head.
        d = mkpanel("a" => 20:80, "b" => 30:80, "c" => 68:83)
        for tr in (NoTransform(), Difference(2))
            fc = mk(MeanModel(); transform=tr)
            head, _, st = cal(fc, d, c)
            usable, _ = C._prepare_all(fc, groups_of(fc, d))
            @test ids_of(head) == ["a", "b"]
            origin = usable[1].tz.ds[1]
            for (k, s) in enumerate(st)
                @test s.step0 == C._step0(fc, usable[k].tz, origin)
                @test s.step0 >= 0
            end
            @test [s.step0 for s in head.series] == [s.step0 for s in st[1:2]]
            @test st[3].step0 == 68 - 20
        end
    end

    @testset "transforms that drop undeclared rows" begin
        c = 12
        # Skipped on the full table: 15 - 14 leaves one row.
        fc = mk(LinAR(0.5, 1.0); transform=TestModels.UnderDeclared(14, 1))
        @test C.minhistory(fc) == 2
        d = mkpanel("a" => 1:60, "b" => 1:15)
        head, E, st = @test_logs (:warn, r"skipping 1 series with fewer than 3 rows") cal(
            fc, d, c)
        @test length(E) == 1
        @test [s.id for s in st] == ids_of(head) == ["a"]
        @test size(E[1]) == (c, 4)
        # Usable on the full table, skipped by the head's prep.
        fc = mk(LinAR(0.5, 1.0); transform=TestModels.UnderDeclared(5, 1))
        d = mkpanel("a" => 20:80, "b" => 65:90)
        # T* = day 69: "b" has 4 rows before it, but 4 - 5 < 0.
        head, E, st = @test_logs (:warn, r"skipping 1 series with fewer than 3 rows") cal(
            fc, d, c)
        @test ids_of(head) == ["a"]
        @test [s.id for s in st] == ["a", "b"]
        @test length(E) == 2
        @test all(isfinite, E[2][1, :])
        @test st[1].step0 == head.series[1].step0 == 0
        # The head's origin is day 25, a's first row after the drop.
        @test st[2].step0 == 70 - 25
    end

    @testset "short series" begin
        c = 12
        fc = mk(MeanModel(); feats=FeatureSet(Lag(2)))
        # n <= mh: skipped with one warning, not calibrated.
        d = mkpanel("a" => 1:60, "z" => 1:2, "b" => 1:50)
        head, E, st = @test_logs (:warn, r"skipping 1 series with fewer than 3 rows") cal(
            fc, d, c)
        @test [s.id for s in st] == ["a", "b"] == ids_of(head)
        @test length(E) == 2
        # mh < n < c + mh + 1 is refused, naming five ids.
        fc1 = mk(MeanModel())
        d = mkpanel("a" => 1:80, "u1" => 1:31, "u2" => 1:30, "u3" => 1:29,
                    "u4" => 1:28, "u5" => 1:27, "u6" => 1:12)
        names = join(("unique_id=\"u$k\" ($(32 - k) rows)" for k in 1:5), ", ")
        TestModels.FIT_COUNT[] = 0
        @test_throws ArgumentError(
            "6 series are too short for intervals=SplitConformal(holdout=30), which " *
            "holds out the last 30 rows of each series to measure errors and needs " *
            "at least 32 rows per series: $names, .... Provide longer series, leave " *
            "these series out, or lower holdout to 10 or less.") cal(fc1, d, 30)
        # Five are listed in full, without the ellipsis.
        d5 = mkpanel("a" => 1:80, "u1" => 1:31, "u2" => 1:30, "u3" => 1:29,
                     "u4" => 1:28, "u5" => 1:20)
        @test_throws ArgumentError(
            "5 series are too short for intervals=SplitConformal(holdout=30), which " *
            "holds out the last 30 rows of each series to measure errors and needs " *
            "at least 32 rows per series: unique_id=\"u1\" (31 rows), " *
            "unique_id=\"u2\" (30 rows), unique_id=\"u3\" (29 rows), " *
            "unique_id=\"u4\" (28 rows), unique_id=\"u5\" (20 rows). Provide longer " *
            "series, leave these series out, or lower holdout to 18 or less.") cal(
            fc1, d5, 30)
        # Lowering would go below 9, so the clause is dropped.
        d1 = mkpanel("a" => 1:80, "u" => 1:10, "z" => 1:1)
        @test_throws ArgumentError(
            "1 series is too short for intervals=SplitConformal(holdout=12), which " *
            "holds out the last 12 rows of each series to measure errors and needs " *
            "at least 14 rows per series: unique_id=\"u\" (10 rows). Provide longer " *
            "series or leave these series out.") cal(fc1, d1, 12)
        # The check runs before any fit or target read.
        ym = Vector{Union{Missing,Float64}}(d1.y)
        ym[end - 1] = missing
        @test_throws "1 series is too short" cal(
            mk(TestModels.CountFit()), merge(d1, (y=ym,)), 12)
        @test TestModels.FIT_COUNT[] == 0
        # The boundary: n = c + mh + 1 calibrates.
        _, E, _ = cal(fc1, mkpanel("a" => 1:80, "u" => 1:14), 12)
        @test length(E) == 2
        # A long series does not excuse a short one.
        @test_throws "1 series is too short" cal(
            fc1, mkpanel("a" => 1:500, "u" => 1:13), 12)
    end

    @testset "Direct needs rows in the series with most head rows" begin
        fc = mk(MeanModel(); strat=Direct(20))
        msg12(id) = "intervals=SplitConformal(holdout=12) trains the calibration " *
                    "model on rows before $(day(14)), the earliest holdout start " *
                    "(series unique_id=\"$id\"). That leaves 13 rows in the series " *
                    "with the most rows before it, 12 after the feature set's 1 " *
                    "history row, but Direct(20) needs 20. Use Direct(12) or a " *
                    "shorter max_horizon, or provide 8 more rows before " *
                    "$(day(14)) in that series."
        # T* = day 14, set by "b"; 13 rows before it in each.
        @test_throws ArgumentError(msg12("b")) cal(
            fc, mkpanel("a" => 1:30, "b" => 1:25), 12)
        # A tie names the first series in table order.
        @test_throws ArgumentError(msg12("z")) cal(
            fc, mkpanel("z" => 1:25, "a" => 1:25), 12)
        # The setter need not have the most rows before the cutoff.
        d = mkpanel("a" => 1:70, "b" => 1:60)
        fc = mk(MeanModel(); strat=Direct(40))
        for p in (d, mkpanel("a" => 1:70, "b" => 11:60))
            @test_throws ArgumentError(
                "intervals=SplitConformal(holdout=30) trains the calibration model " *
                "on rows before $(day(31)), the earliest holdout start (series " *
                "unique_id=\"b\"). That leaves 30 rows in the series with the most " *
                "rows before it, 29 after the feature set's 1 history row, but " *
                "Direct(40) needs 40. Lower holdout to 19 or less, use Direct(29) " *
                "or a shorter max_horizon, or provide 11 more rows before " *
                "$(day(31)) in that series.") cal(fc, p, 30)
        end
        _, E, _ = cal(fc, d, 19)
        @test size(E[1]) == (19, 11)
        # With a transform the message names its dropped rows.
        fd = mk(MeanModel(); strat=Direct(20), transform=Difference(2))
        err = caught(() -> cal(fd, mkpanel("a" => 1:30, "b" => 1:25), 12))
        @test err isa ArgumentError
        @test occursin("after the feature set's 1 history row and the transform's " *
                       "2 dropped rows", err.msg)
        # A series short of D after the cut warns once.
        fw = mk(MeanModel(); strat=Direct(10))
        d = mkpanel("a" => 1:60, "b" => 40:60)
        head, E, st = @test_logs (:warn, "Direct(10): 1 series with fewer than 10 " *
                                  "training rows is left out of the step models " *
                                  "beyond their row count: unique_id=\"b\" " *
                                  "(8 training rows)") cal(fw, d, 12)
        @test length(E) == 2
        @test [s.n_train for s in head.series] == [48, 9]
    end

    @testset "both learners on a ragged panel" begin
        c = 20
        d = mkpanel("a" => 1:90, "b" => 15:100, "c" => 30:85, "d" => 70:110)
        feats = FeatureSet(Lag(1), Lag(2), RollingMean(3), Calendar(:dayofweek))
        for model in (EvoTreeRegressor(nrounds=5), dtree()),
            strat in (Recursive(), Direct(4))

            head, E, st = cal(mk(model; feats, strat), d, c)
            depth = C._depth(strat, c)
            @test length(E) == 4 == length(st)
            for Ek in E
                @test all(isnan(Ek[i, j]) == (j > c - i + 1) for i in 1:c, j in 1:depth)
                @test all(isfinite(Ek[i, j]) for i in 1:c for j in 1:min(depth, c - i + 1))
            end
            @test all(isfinite, forecast(
                C.FittedForecaster(mk(model; feats, strat), head.models, st,
                                   head.feature_names), 4).y_hat)
        end
    end

    @testset "errors are named for a panel" begin
        c = 12
        d = mkpanel("a" => 1:60, "b" => 10:70)
        pre = "intervals=SplitConformal(holdout=12), calibrating origins "
        fp = mk(TestModels.ThrowIn(:predict, false))
        for (chunk, r) in ((1, "1 to 1"), (5, "1 to 5"), (c, "1 to 12"))
            @test_throws ArgumentError(
                pre * "$r of the 12 in each series (origin 1 is the last row before " *
                "its holdout): boom") cal(fp, d, c; chunk)
        end
        # Single series text stays rows.
        fs = mk(TestModels.ThrowIn(:predict, false); id=nothing)
        one = (ds=[day(i) for i in 1:60], y=ys(1, 1:60))
        @test_throws ArgumentError(pre * "at rows 48 to 54: boom") C._calibrate(
            siv(c), fs, C._series_groups(fs, C.normalize_table(one)); chunk=7)
        @test_throws ErrorException cal(mk(TestModels.ThrowIn(:predict, true)), d, c)
    end

    @testset "the failing series does not depend on the chunk size" begin
        c, n = 12, 60
        d = mkpanel("a" => 1:n, "b" => 1:n)
        y = copy(d.y)
        isb = d.unique_id .== "b"
        y[isb] .+= 25
        # "a" diverges at origin 3, "b" at origin 1.
        y[n - c + 2] = 100.0
        d = merge(d, (y=y,))
        fc = mk(TestModels.NaNAbove(30.0))
        o = n - c
        r = n + o
        msg = "series unique_id=\"b\": the calibration forecast from origin " *
              "$(day(o)) (row $r) is NaN at step 1. A non-finite forecast cannot " *
              "give an interval. Check the model's hyperparameters and the training " *
              "data; a smaller holdout calibrates fewer steps."
        for chunk in (1, 2, 5, c)
            @test_throws ArgumentError(msg) cal(fc, d, c; chunk)
        end
        # Both fail at origin 1: the first series is named.
        @test_throws "series unique_id=\"a\": the calibration forecast from origin" cal(
            mk(TestModels.NaNModel()), d, c)
    end

    @testset "_pseudo_state keeps the prefix as history" begin
        c = 12
        d = mkpanel("a" => 1:60, "b" => 10:70)
        fc = mk(LinAR(0.5, 1.0); transform=LastScale(3))
        usable, _ = C._prepare_all(fc, groups_of(fc, d))
        for p in usable, o in (48, 55, length(p.t))
            ps = C._pseudo_state(siv(c), fc, p, o, 0)
            @test ps.y_history == p.yraw[1:o]
            @test length(ps.y_history) == ps.n_train == o
        end
    end

    @testset "missing exogenous values in a panel" begin
        c, depth = 12, 4
        spans = ("a" => 1:50, "b" => 5:55)
        feats = FeatureSet(Lag(1), Exogenous(:x))
        fc = mk(EchoColumn(:x); feats)
        d = shuf(mkpanel(spans...; x=true), 5)
        row(id, dd) = findfirst(i -> d.unique_id[i] == id && d.ds[i] == day(dd),
                                eachindex(d.ds))
        miss(cells...) = (xm = copy(d.x);
                          foreach(cl -> xm[row(cl...)] = missing, cells);
                          merge(d, (x=xm,)))
        msg(id, dd) = "series unique_id=\"$id\": feature column :x has a missing " *
                      "value at row $(row(id, dd)) (time $(day(dd))). Missing values " *
                      "in exogenous columns are not supported; impute or drop them " *
                      "before fitting."
        # Rows interleave; each series reads its own x.
        head, E, st = cal(fc, d, c)
        for (k, (id, r)) in enumerate(spans)
            kk = findfirst(s -> s.id == id, st)
            n = length(r)
            for i in 1:c, j in 1:min(depth, c - i + 1)
                dd = first(r) + n - c + i - 2 + j
                @test E[kk][i, j] == yv(dd, k) - xv(dd, k)
            end
        end
        # The first mh rows are never read.
        dm = miss(("a", 1), ("b", 5))
        @test isequal(cal(fc, dm, c)[2], E)
        @test fit(fc, dm) isa C.FittedForecaster
        # A holdout row, a head row: the plain fit's message.
        for cells in ((("b", 50),), (("b", 20),), (("a", 45), ("b", 20)),
                      (("b", 50), ("a", 45)), (("a", 1), ("b", 50)))
            data = miss(cells...)
            want = caught(() -> fit(fc, data))
            @test want isa ArgumentError
            @test occursin("series unique_id=", want.msg)
            @test_throws ArgumentError(want.msg) cal(fc, data, c)
        end
        @test_throws ArgumentError(msg("b", 50)) cal(fc, miss(("b", 50)), c)
        # A missing column gives the message of a plain fit.
        absent = NamedTuple{(:unique_id, :ds, :y)}((d.unique_id, d.ds, d.y))
        want = caught(() -> fit(fc, absent))
        @test want isa ArgumentError
        @test_throws ArgumentError(want.msg) cal(fc, absent, c)
    end
end

@testset "SplitConformal forecaster" begin
    C = MachineLearningForecast
    LinAR, MeanModel = TestModels.LinAR, TestModels.MeanModel
    siv(c) = C.SplitConformal(holdout=c)
    mk(model; c=nothing, feats=FeatureSet(Lag(1)), strat=Recursive(), id=nothing,
       kw...) = Forecaster(model; features=feats, strategy=strat, freq=Day(1), id,
                           intervals=c === nothing ? nothing : siv(c), kw...)
    # Dyadic values, so closed forms compare exactly.
    function mkdata(n)
        return (ds=collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(n - 1))),
                y=[8 + mod(7i, 16) / 4 + i / 8 for i in 1:n])
    end
    q6(v, p) = quantile(v, p; alpha=0.0, beta=0.0)
    function smat(f, h)
        s = forecast_samples(f, h)
        K = maximum(s.sample)
        return permutedims(reshape(s.y_hat, K, h))
    end
    models() = (LinAR(0.5, 1.0), MeanModel(), EvoTreeRegressor(nrounds=5), dtree())
    triangle(f) = getfield(f, :calibration)

    @testset "y_hat is the head's forecast" begin
        n, c = 40, 12
        d = mkdata(n)
        m = sum(d.y[2:(n - c)]) / (n - c - 1)
        f = fit(mk(MeanModel(); c), d)
        @test forecast(f, 3).y_hat ≈ fill(m, 3)
        @test forecast(fit(mk(MeanModel()), d), 3).y_hat ≈
              fill(sum(d.y[2:n]) / (n - 1), 3)
        @test forecast(f, 3).y_hat != forecast(fit(mk(MeanModel()), d), 3).y_hat
        @test forecast(f, 3; levels=(80,)).y_hat == forecast(f, 3).y_hat

        f = fit(mk(MeanModel(); c, strat=Direct(3)), d)
        @test forecast(f, 3).y_hat ≈
              [sum(d.y[(1 + j):(n - c)]) / (n - c - j) for j in 1:3]

        for strat in (Recursive(), Direct(2))
            TestModels.FIT_COUNT[] = 0
            fit(mk(TestModels.CountFit(); c, strat), d)
            @test TestModels.FIT_COUNT[] == C.nmodels(strat)
        end

        # LinAR ignores training: the states are the full history.
        for strat in (Recursive(), Direct(4))
            a = fit(mk(LinAR(0.5, 1.0); c, strat), d)
            b = fit(mk(LinAR(0.5, 1.0); strat), d)
            @test forecast(a, 4).y_hat == forecast(b, 4).y_hat
            @test a.n_train == n && a.t_last == d.ds[end] && a.t_start == d.ds[1]
            @test a.y_history == d.y
        end

        # Trees publish models fitted before the holdout.
        n, c = 70, 20
        d = mkdata(n)
        for model in (EvoTreeRegressor(nrounds=5), dtree())
            plain = mk(model)
            f = fit(mk(model; c), d)
            head = fit(plain, (ds=d.ds[1:(n - c)], y=d.y[1:(n - c)]))
            fh = C.FittedForecaster(plain, head.models, f.series, head.feature_names)
            @test forecast(f, 5).y_hat == forecast(fh, 5).y_hat
            @test forecast(f, 5).y_hat != forecast(fit(plain, d), 5).y_hat
        end
    end

    @testset "levels give ordered interval columns" begin
        n, c, h = 90, 50, 5
        d = mkdata(n)
        for model in models(), strat in (Recursive(), Direct(5))
            f = fit(mk(model; c, strat), d)
            o = forecast(f, h; levels=(80, 95))
            @test keys(o) == (:ds, :y_hat, :lo_80, :hi_80, :lo_95, :hi_95)
            @test o.ds == forecast(f, h).ds
            @test o.y_hat == forecast(f, h).y_hat
            @test all(o.lo_95 .<= o.lo_80 .<= o.hi_80 .<= o.hi_95)
            r = forecast(f, h; levels=(95, 80))
            @test keys(r) == (:ds, :y_hat, :lo_95, :hi_95, :lo_80, :hi_80)
            @test r.lo_95 == o.lo_95 && r.hi_80 == o.hi_80
        end
    end

    @testset "intervals are the samples' type 6 quantiles" begin
        n, c, h = 90, 50, 5
        d = mkdata(n)
        for model in models(), strat in (Recursive(), Direct(5))
            f = fit(mk(model; c, strat), d)
            o = forecast(f, h; levels=(80, 95))
            S = smat(f, h)
            @test size(S) == (h, c - h + 1)
            for j in 1:h
                @test o.lo_80[j] ≈ q6(S[j, :], 0.1)
                @test o.hi_80[j] ≈ q6(S[j, :], 0.9)
                @test o.lo_95[j] ≈ q6(S[j, :], 0.025)
                @test o.hi_95[j] ≈ q6(S[j, :], 0.975)
            end
            E = triangle(f)[1]
            o1 = forecast(f, 1; levels=(80,))
            S1 = forecast(f, 1).y_hat[1] .+ E[1:c, 1]
            @test o1.lo_80[1] ≈ q6(S1, 0.1) && o1.hi_80[1] ≈ q6(S1, 0.9)
        end
    end

    @testset "at K = need(L) the interval is the sample range" begin
        d = mkdata(50)
        for model in models(), (c, h, L) in ((12, 4, 80), (21, 3, 90))
            f = fit(mk(model; c), d)
            o = forecast(f, h; levels=(L,))
            S = smat(f, h)
            @test size(S, 2) == C._need_errors(Float64(L))
            @test getproperty(o, Symbol("lo_", L)) ≈ vec(minimum(S; dims=2))
            @test getproperty(o, Symbol("hi_", L)) ≈ vec(maximum(S; dims=2))
        end
    end

    @testset "closed form: type 6 over the head's errors" begin
        n, c = 60, 40
        d = mkdata(n)
        y = d.y
        m = sum(y[2:(n - c)]) / (n - c - 1)
        f = fit(mk(MeanModel(); c), d)
        errs(K, j) = [y[n - c + i - 1 + j] - m for i in 1:K]
        for h in (1, 6, 12)
            K = c - h + 1
            o = forecast(f, h; levels=(80, 90))
            for j in 1:h
                S = m .+ errs(K, j)
                @test o.lo_80[j] ≈ q6(S, 0.1) && o.hi_90[j] ≈ q6(S, 0.95)
            end
        end
        # A longer call keeps fewer, older origins at every step.
        lo1 = forecast(f, 1; levels=(80,)).lo_80[1]
        lo12 = forecast(f, 12; levels=(80,)).lo_80[1]
        @test lo1 ≈ q6(m .+ errs(c, 1), 0.1)
        @test lo12 ≈ q6(m .+ errs(c - 11, 1), 0.1)
        @test lo1 != lo12
        @test size(triangle(f)[1]) == (c, C._depth(Recursive(), c))
    end

    @testset "forecast_samples" begin
        n, c, h = 50, 20, 6
        d = mkdata(n)
        K = c - h + 1
        for model in models()
            f = fit(mk(model; c), d)
            s = forecast_samples(f, h)
            @test keys(s) == (:ds, :sample, :y_hat)
            @test length(s.y_hat) == h * K
            @test s.sample == repeat(1:K; outer=h)
            @test s.ds == repeat(forecast(f, h).ds; inner=K)
            @test s == forecast_samples(f, h)
            E = triangle(f)[1]
            @test s.y_hat[1:K] == forecast(f, h).y_hat[1] .+ E[1:K, 1]
            Random.seed!(7)
            a = rand()
            Random.seed!(7)
            forecast_samples(f, h)
            forecast(f, h; levels=(80,))
            @test rand() == a
        end
        f = fit(mk(MeanModel(); c), d)
        s = forecast_samples(f, 1)
        @test s.sample == 1:c
    end

    @testset "a panel" begin
        c, h = 12, 3
        D0 = Date(2020, 1, 1)
        yv(d, k) = 8 + mod(7d + 3k, 16) / 4 + d / 8
        spans = (("a", 60), ("b", 50))
        pd = (unique_id=[id for (id, n) in spans for _ in 1:n],
              ds=[D0 + Day(d - 1) for (_, n) in spans for d in 1:n],
              y=[yv(d, k) for (k, (_, n)) in enumerate(spans) for d in 1:n])
        f = fit(mk(MeanModel(); c, id=:unique_id), pd)
        # The cutoff is day 39, series b's holdout start.
        m = sum(yv(d, k) for k in 1:2 for d in 2:38) / (2 * 37)
        o = forecast(f, h; levels=(80,))
        @test keys(o) == (:unique_id, :ds, :y_hat, :lo_80, :hi_80)
        @test o.unique_id == ["a", "a", "a", "b", "b", "b"]
        @test o.y_hat ≈ fill(m, 6)
        K = c - h + 1
        for (k, (id, n)) in enumerate(spans), j in 1:h
            S = m .+ [yv(n - c + i - 1 + j, k) - m for i in 1:K]
            r = findfirst(==(id), o.unique_id) + j - 1
            @test o.lo_80[r] ≈ q6(S, 0.1) && o.hi_80[r] ≈ q6(S, 0.9)
        end
        @test [size(E) for E in triangle(f)] == [(c, 4), (c, 4)]
        s = forecast_samples(f, h)
        @test keys(s) == (:unique_id, :ds, :sample, :y_hat)
        @test [count(==(id), s.unique_id) for id in ("a", "b")] == [h * K, h * K]
        @test s.sample == repeat(1:K; outer=2h)

        n = 70
        pd = (unique_id=[fill("a", n); fill("b", n - 10)],
              ds=[mkdata(n).ds; mkdata(n).ds[11:n]],
              y=[mkdata(n).y; 2 .* mkdata(n).y[11:n]])
        for model in models(), strat in (Recursive(), Direct(5))
            f = fit(mk(model; c=30, strat, id=:unique_id), pd)
            o = forecast(f, 5; levels=(80,))
            s = forecast_samples(f, 5)
            @test length(o.y_hat) == 10 && length(s.y_hat) == 10 * 26
            @test all(o.lo_80 .<= o.hi_80)
            for k in 1:2, j in 1:5
                S = reshape(s.y_hat[(26 * 5 * (k - 1) + 1):(26 * 5 * k)], 26, 5)'
                @test o.lo_80[5 * (k - 1) + j] ≈ q6(S[j, :], 0.1)
            end
        end
    end

    @testset "levels the holdout cannot support are refused" begin
        d = mkdata(80)
        f40 = fit(mk(MeanModel(); c=40), d)
        top = "levels=(95,) needs at least 39 calibration errors per step, but " *
              "intervals=SplitConformal(holdout=40) gives 29 at forecast horizon 12 " *
              "(the origins that have 12 actuals), which supports central levels up " *
              "to 93.33. Pass levels <= 93.33, forecast at most 2 steps, or fit with " *
              "holdout >= 50."
        @test_throws top forecast(f40, 12; levels=(95,))
        @test_throws top forecast(f40, 12; levels=(80, 95))
        @test_throws top forecast(f40, 12; levels=(95, 80))
        @test forecast(f40, 12; levels=(93,)) isa NamedTuple
        @test forecast(f40, 2; levels=(95,)) isa NamedTuple
        @test_throws "gives 38 at forecast horizon 3" forecast(f40, 3; levels=(95,))

        f30 = fit(mk(MeanModel(); c=30), d)
        @test_throws "levels=(95,) needs at least 39 calibration errors per step, but " *
                     "intervals=SplitConformal(holdout=30) gives 29 at forecast " *
                     "horizon 2 (the origins that have 2 actuals), which supports " *
                     "central levels up to 93.33. Pass levels <= 93.33, or fit with " *
                     "holdout >= 40." forecast(f30, 2; levels=(95,))
        @test_throws "gives 30 at forecast horizon 1 (the origins that have 1 " *
                     "actual), which supports central levels up to 93.54. Pass " *
                     "levels <= 93.54, or fit with holdout >= 39." forecast(
            f30, 1; levels=(95,))

        f60 = fit(mk(MeanModel(); c=60), d)
        deep = "horizon 55 needs at least 9 calibration errors per step, but " *
               "intervals=SplitConformal(holdout=60) gives 6 at that horizon (the " *
               "origins that have 55 actuals). Forecast at most 52 steps, or fit " *
               "with holdout >= 63."
        @test_throws "forecast_samples $deep" forecast_samples(f60, 55)
        @test_throws "forecast $deep" forecast(f60, 55; levels=(80,))
        @test_throws "forecast $deep" forecast(f60, 55; levels=(95, 50))
        @test forecast_samples(f60, 52) isa NamedTuple

        # One step is never refused at a supported level.
        f9 = fit(mk(MeanModel(); c=9), d)
        @test forecast(f9, 1; levels=(80,)) isa NamedTuple
        @test forecast_samples(f9, 1).sample == 1:9
        @test_throws "forecast horizon 2 needs at least 9 calibration errors per " *
                     "step, but intervals=SplitConformal(holdout=9) gives 8 at that " *
                     "horizon (the origins that have 2 actuals). Forecast at most 1 " *
                     "step, or fit with holdout >= 10." forecast(f9, 2; levels=(80,))

        # Without levels no horizon is refused.
        for f in (f9, f40, f60)
            @test length(forecast(f, 100).y_hat) == 100
        end
    end

    @testset "Direct's own horizon check comes first" begin
        d = mkdata(80)
        f = fit(mk(MeanModel(); c=40, strat=Direct(12)), d)
        text = "strategy=Direct(12) was fit with max_horizon=12 but forecast(h=24) " *
               "was requested. Refit with Direct(24) or use Recursive()."
        @test_throws text forecast(f, 24)
        @test_throws text forecast(f, 24; levels=(80,))
        @test_throws text forecast_samples(f, 24)
        @test forecast(f, 12; levels=(80,)) isa NamedTuple
        @test_throws "forecast horizon 13 needs" forecast(
            fit(mk(MeanModel(); c=20, strat=Direct(20)), d), 13; levels=(80,))
        @test_throws "backtest horizon=24 exceeds the Direct strategy's " *
                     "max_horizon=12. Use Direct(24) or reduce horizon." C._check_support(
            siv(40), mk(MeanModel(); c=40, strat=Direct(12)), (80.0,), 24,
            "backtest horizon")
    end

    @testset "backtest wording" begin
        fc = mk(MeanModel(); c=40)
        @test_throws "levels=(95,) needs at least 39 calibration errors per step, but " *
                     "intervals=SplitConformal(holdout=40) gives 29 at backtest horizon " *
                     "12 (the origins that have 12 actuals), which supports central " *
                     "levels up to 93.33. Pass levels <= 93.33, use horizon <= 2, or " *
                     "fit with holdout >= 50." backtest(
            fc, mkdata(100); horizon=12, windows=2, levels=(95,))
        @test_throws "backtest horizon 55 needs at least 9 calibration errors per " *
                     "step, but intervals=SplitConformal(holdout=60) gives 6 at that " *
                     "horizon (the origins that have 55 actuals). Use horizon <= 52, " *
                     "or fit with holdout >= 63." C._check_support(
            siv(60), mk(MeanModel(); c=60), Float64[], 55, "backtest horizon")
        @test C._check_support(siv(60), mk(MeanModel(); c=60), Float64[], 52,
                               "backtest horizon") === nothing
    end

    @testset "Phase 1 level messages are kept" begin
        f = fit(mk(MeanModel(); c=40), mkdata(80))
        @test_throws "an interval level is a percentage in (1, 100), got 0.95. Pass " *
                     "e.g. 95 for a 95% interval, not 0.95." forecast(
            f, 3; levels=0.95)
        @test_throws "levels is empty; pass at least one level, e.g. levels=(95,)." (
            forecast(f, 3; levels=()))
        @test_throws "levels repeats a level" forecast(f, 3; levels=(80, 80))
        d = mkdata(80)
        clash = fit(mk(MeanModel(); c=40, time=:lo_80), (lo_80=d.ds, y=d.y))
        @test_throws "the interval column :lo_80 would overwrite a column of the " *
                     "result. Rename your time or id column." forecast(
            clash, 3; levels=(80,))
    end

    @testset "the message texts" begin
        csp = ConformalSeasonalPool(12; freq=Month(1))
        bad = "intervals must be nothing, a SplitConformal(holdout=c) or a " *
              "PoolIntervals(m), got :csp. Pass intervals=SplitConformal(holdout=60) " *
              "for any model, or intervals=PoolIntervals(m) for the seasonal naive " *
              "one; ConformalSeasonalPool(m; freq) builds a whole Forecaster."
        @test_throws bad mk(MeanModel(); intervals=:csp)
        @test_throws "intervals must be nothing, a SplitConformal" mk(
            MeanModel(); intervals=0.95)
        @test mk(MeanModel(); c=12).intervals === siv(12)
        @test C.reconstruct(mk(MeanModel(); c=12); model=LinAR(1.0, 0.0)).intervals ==
              siv(12)
        none = fit(mk(MeanModel()), mkdata(40))
        who(w) = "$w needs an interval method, but this forecaster has none. Pass " *
                 "intervals=SplitConformal(holdout=60) for any model, or build the " *
                 "seasonal naive one with ConformalSeasonalPool(period; freq=...)."
        @test_throws who("forecast with levels") forecast(none, 3; levels=(80,))
        @test_throws who("forecast_samples") forecast_samples(none, 3)
        @test_throws who("backtest with levels or a sample-based metric") backtest(
            mk(MeanModel()), mkdata(40); horizon=3, windows=2, levels=(80,))
        @test_throws "For any other model pass intervals=SplitConformal(holdout=60)." *
                     " Use ConformalSeasonalPool(12; freq=...)" Forecaster(
            MeanModel(); features=FeatureSet(Lag(12)), freq=Month(1),
            intervals=C.PoolIntervals(12))
        tunable = "cannot reconstruct Forecaster with the field :intervals, which " *
                  "cannot be tuned: tune ranks candidates by the published point " *
                  "forecasts, and a SplitConformal forecaster publishes a model fitted " *
                  "without its last holdout rows, so interval settings change y_hat. " *
                  "Compare them with backtest(...; metrics=(crps, Coverage(95)))."
        @test_throws tunable C.reconstruct(mk(MeanModel(); c=12); intervals=nothing)
        @test occursin("so changing it changes `y_hat`", string(@doc(C.TUNABLE_FIELDS)))
        @test !occursin("leave `y_hat` unchanged", string(@doc(C.TUNABLE_FIELDS)))

        # The old texts are gone (split in this file).
        old = ["Build it with " * "ConformalSeasonalPool",
               "must be nothing " * "or a PoolIntervals",
               "models are not " * "available",
               "leave y_hat " * "unchanged"]
        for dir in (joinpath(pkgdir(C), "src"), @__DIR__), file in readdir(dir)
            endswith(file, ".jl") || continue
            text = read(joinpath(dir, file), String)
            @test !any(s -> occursin(s, text), old)
        end
    end

    @testset "_sample_matrices" begin
        n, c = 40, 12
        f = fit(mk(MeanModel(); c), mkdata(n))
        iv = f.spec.intervals
        preds = [[1.0, 2.0, 3.0]]
        S = @inferred C._sample_matrices(iv, f, preds)
        @test S isa Vector{Matrix{Float64}}
        E = triangle(f)[1]
        @test S[1] == preds[1] .+ transpose(E[1:(c - 2), 1:3])
        @test size(S[1]) == (3, c - 2)
        @test size(only(C._sample_matrices(iv, f, [Float64[]]))) == (0, c)
        @test_throws "forecast horizon 20 needs at least 9 calibration errors per " *
                     "step, but intervals=SplitConformal(holdout=12) gives 0 at that " *
                     "horizon (the origins that have 20 actuals). Forecast at most 4 " *
                     "steps, or fit with holdout >= 28." C._sample_matrices(
            iv, f, [fill(1.0, 20)])
    end

    @testset "the triangle is a hidden field" begin
        d = mkdata(40)
        public = (:spec, :models, :series, :feature_names, :y_history, :t_last,
                  :t_start, :n_train, :z_history, :transform_state)
        private = (:spec, :models, :series, :feature_names, :calibration, :y_history,
                   :t_last, :t_start, :n_train, :z_history, :transform_state)
        @test fieldnames(C.FittedForecaster) ==
              (:spec, :models, :series, :feature_names, :calibration)
        @test fieldnames(Forecaster) == (:model, :features, :strategy, :freq, :target,
                                         :time, :id, :transform, :intervals)
        csp = ConformalSeasonalPool(12; freq=Day(1))
        for fc in (mk(MeanModel()), mk(MeanModel(); c=12), csp)
            f = fit(fc, d)
            @test propertynames(f) == public
            @test propertynames(f, false) == public
            @test propertynames(f, true) == private
            @test f.calibration === triangle(f)
        end
        @test triangle(fit(mk(MeanModel()), d)) == Matrix{Float64}[]
        @test triangle(fit(csp, d)) isa Vector{Matrix{Float64}}
        @test isempty(triangle(fit(csp, d)))
        @test isempty(triangle(fit(mk(MeanModel(); id=:g), (g=fill("a", 40), d...))))
        for strat in (Recursive(), Direct(3))
            f = fit(mk(MeanModel(); c=12, strat), d)
            @test [size(E) for E in triangle(f)] == [(12, C._depth(strat, 12))]
        end
        # The four-argument constructor leaves it empty.
        f = fit(mk(MeanModel()), d)
        g = C.FittedForecaster(f.spec, f.models, f.series, f.feature_names)
        @test isempty(triangle(g)) && g.spec === f.spec
    end

    @testset "the spec carries no stale triangle" begin
        d, d2 = mkdata(60), mkdata(75)
        for model in models(), strat in (Recursive(), Direct(3))
            fc = mk(model; c=20, strat)
            f = fit(fc, d)
            @test f.spec === fc
            a, b = fit(f.spec, d2), fit(fc, d2)
            @test isequal(triangle(a), triangle(b))
            @test !isequal(triangle(a), triangle(f))
            @test forecast(a, 3).y_hat == forecast(b, 3).y_hat
            @test forecast_samples(a, 3) == forecast_samples(b, 3)
        end
    end

    @testset "backtest and tune through the spec" begin
        d = mkdata(60)
        for model in (LinAR(0.5, 1.0), dtree())
            fc = mk(model; c=12)
            f = fit(fc, d)
            kw = (horizon=3, windows=2)
            a = backtest(f.spec, d; kw..., levels=(80,))
            b = backtest(fc, d; kw..., levels=(80,))
            @test isequal(a.forecasts, b.forecasts) && isequal(a.scores, b.scores)
            @test isequal(backtest(f.spec, d; kw...).forecasts,
                          backtest(fc, d; kw...).forecasts)
            grid = (model=[LinAR(0.5, 1.0), LinAR(0.25, 1.0)],)
            r = tune(f.spec, d; grid, kw..., metric=rmse)
            s = tune(fc, d; grid, kw..., metric=rmse)
            @test isequal(r.table, s.table)
        end
    end

    @testset "fit mutates nothing" begin
        d = mkdata(50)
        pd = (unique_id=[fill("a", 50); fill("b", 44)], ds=[d.ds; d.ds[7:50]],
              y=[d.y; 2 .* d.y[7:50]])
        for (data, id) in ((d, nothing), (pd, :unique_id)), model in models()
            d0 = deepcopy(data)
            proto = deepcopy(model)
            fc = mk(model; c=12, id)
            f = fit(fc, data)
            @test data == d0
            @test TestModels.MMI.is_same_except(fc.model, proto)
            @test all(m -> m.model !== fc.model, f.models)
        end
    end

    @testset "show is unchanged" begin
        d = mkdata(40)
        f = fit(mk(MeanModel(); c=12), d)
        @test sprint(show, f) == sprint(show, fit(mk(MeanModel()), d))
        @test sprint(show, f) == "FittedForecaster(MeanModel, 1 features, Recursive(), " *
                                 "freq=1 day, trained on 40 rows ending $(d.ds[end]))"
        @test endswith(sprint(show, f.spec), ", intervals=SplitConformal(holdout=12))")
    end
end

@testset "SplitConformal backtest" begin
    C = MachineLearningForecast
    LinAR, MeanModel, CountFit = TestModels.LinAR, TestModels.MeanModel, TestModels.CountFit
    FIT = TestModels.FIT_COUNT
    D0 = Date(2020, 1, 1)
    day(d) = D0 + Day(d - 1)
    siv(c) = SplitConformal(holdout=c)
    mk(model; c=nothing, feats=FeatureSet(Lag(1)), strat=Recursive(), id=nothing,
       transform=NoTransform()) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1), id, transform,
                   intervals=c === nothing ? nothing : siv(c))
    pmk(model; kw...) = mk(model; id=:unique_id, kw...)
    models() = (LinAR(0.5, 1.0), MeanModel(), EvoTreeRegressor(nrounds=5), dtree())
    # Dyadic values, so closed forms compare exactly.
    yv(d, k) = 8 + mod(7d + 3k, 16) / 4 + d / 8
    xv(d, k) = 100 + mod(5d + k, 8) / 2 + d / 4
    ys(k, r) = [yv(d, k) for d in r]
    function mkpanel(spans...; x=false)
        ids, ds, y, xs = Any[], Date[], Float64[], Float64[]
        for (k, (id, r)) in enumerate(spans), d in r
            push!(ids, id)
            push!(ds, day(d))
            push!(y, yv(d, k))
            push!(xs, xv(d, k))
        end
        base = (unique_id=identity.(ids), ds=ds, y=y)
        return x ? merge(base, (x=xs,)) : base
    end
    mkdata(n) = (d = mkpanel(("a", 1:n)); (ds=d.ds, y=d.y))
    take(d, mask) = NamedTuple{keys(d)}(map(v -> v[mask], values(d)))
    headmean(parts, mh) = (v = reduce(vcat, [ys(k, (first(r) + mh):last(r))
                                              for (k, r) in parts]);
                           sum(v) / length(v))
    val(r, k, name) = only(r.scores.value[(r.scores.fold .== k) .&
                                          (r.scores.metric .== name)])
    caught(f) = try f(); nothing catch err err end

    @testset "levels on a single series equal the intervals of each fold's fit" begin
        d, h = mkdata(90), 3
        ms = (mae, crps, fair_crps, Coverage(95), IntervalWidth(95), pinball_loss)
        for model in models()
            fc = mk(model; c=45)
            r = backtest(fc, d; horizon=h, windows=3, levels=(95,), metrics=ms)
            @test keys(r.forecasts) == (:fold, :origin, :step, :ds, :y, :y_hat,
                                        :lo_95, :hi_95)
            @test summarize(r).metric == [:mae, :crps, :fair_crps, :coverage_95,
                                          :interval_width_95, :pinball_loss]
            for k in 1:3
                o = 78 + 3k
                f = fit(fc, (ds=d.ds[1:o], y=d.y[1:o]))
                fo = forecast(f, h; levels=(95,))
                sel = r.forecasts.fold .== k
                for c in (:y_hat, :lo_95, :hi_95)
                    @test r.forecasts[c][sel] == fo[c]
                end
                @test r.forecasts.y[sel] == d.y[(o + 1):(o + h)]
                _, dr = C._forecast_samples(f, [h], nothing)
                S, truth = only(dr), r.forecasts.y[sel]
                @test val(r, k, :crps) == crps(truth, S)
                @test val(r, k, :fair_crps) == fair_crps(truth, S)
                @test val(r, k, :pinball_loss) == pinball_loss(truth, S)
                @test val(r, k, :interval_width_95) == IntervalWidth(95)(truth, S)
                @test val(r, k, :coverage_95) == Coverage(95)(truth, S)
                @test val(r, k, :mae) == mae(truth, fo.y_hat)
            end
        end
    end

    @testset "levels on a panel equal the intervals of each fold's fit" begin
        d, h = mkpanel("a" => 1:90, "b" => 1:80), 3
        ms = (mae, crps, Coverage(95), IntervalWidth(95))
        for model in models()
            fc = pmk(model; c=45)
            r = backtest(fc, d; horizon=h, windows=2, levels=(95,), metrics=ms)
            @test keys(r.forecasts) == (:fold, :origin, :step, :unique_id, :ds, :y,
                                        :y_hat, :lo_95, :hi_95)
            @test all(ismissing, r.scores.origin)
            for k in 1:2
                cut = Dict("a" => 87 - 3 * (2 - k), "b" => 77 - 3 * (2 - k))
                keep = [Dates.value(d.ds[i] - D0) + 1 ≤ cut[d.unique_id[i]]
                        for i in eachindex(d.ds)]
                f = fit(fc, take(d, keep))
                fo = forecast(f, h; levels=(95,))
                sel = r.forecasts.fold .== k
                @test r.forecasts.unique_id[sel] == [fill("a", h); fill("b", h)]
                for c in (:y_hat, :lo_95, :hi_95)
                    @test r.forecasts[c][sel] == fo[c]
                end
                _, dr = C._forecast_samples(f, [h, h], nothing)
                S, truth = reduce(vcat, dr), r.forecasts.y[sel]
                @test val(r, k, :crps) == crps(truth, S)
                @test val(r, k, :coverage_95) == Coverage(95)(truth, S)
                @test val(r, k, :interval_width_95) == IntervalWidth(95)(truth, S)
            end
        end
    end

    @testset "y_hat does not depend on levels, and is the fold's head forecast" begin
        d, h, c = mkdata(90), 3, 20
        for model in models()
            fc = mk(model; c)
            a = backtest(fc, d; horizon=h, windows=3)
            b = backtest(fc, d; horizon=h, windows=3, levels=(80,))
            e = backtest(fc, d; horizon=h, windows=3, metrics=(mae, crps))
            @test a.forecasts.y_hat == b.forecasts.y_hat == e.forecasts.y_hat
            @test keys(a.forecasts) == (:fold, :origin, :step, :ds, :y, :y_hat)
        end
        # MeanModel: mean of the rows before the holdout.
        fc = mk(MeanModel(); c)
        for kw in ((;), (levels=(80,),))
            r = backtest(fc, d; horizon=h, windows=3, kw...)
            for k in 1:3
                o = 78 + 3k
                @test r.forecasts.y_hat[r.forecasts.fold .== k] ==
                      fill(headmean([(1, 1:(o - c))], 1), h)
            end
        end
        # A plain forecaster fits all the rows of the fold.
        plain = backtest(mk(MeanModel()), d; horizon=h, windows=3)
        @test plain.forecasts.y_hat[1:h] == fill(headmean([(1, 1:81)], 1), h)
        # A panel's head stops before the earliest holdout.
        pd = mkpanel("a" => 1:90, "b" => 1:70)
        for model in models()
            pfc = pmk(model; c)
            a = backtest(pfc, pd; horizon=h, windows=3)
            b = backtest(pfc, pd; horizon=h, windows=3, levels=(80,))
            @test a.forecasts.y_hat == b.forecasts.y_hat
        end
        pm = pmk(MeanModel(); c)
        r = backtest(pm, pd; horizon=h, windows=3)
        for k in 1:3
            # Series b ends first and sets the cutoff.
            cutb = 70 - h - (3 - k) * h
            want = headmean([(1, 1:(cutb - c)), (2, 1:(cutb - c))], 1)
            @test r.forecasts.y_hat[r.forecasts.fold .== k] == fill(want, 2h)
        end
    end

    @testset "refused before any fit" begin
        d = mkdata(100)
        fc = mk(CountFit(); c=40)
        FIT[] = 0
        @test_throws "levels=(95,) needs at least 39 calibration errors per step, but " *
                     "intervals=SplitConformal(holdout=40) gives 29 at backtest horizon " *
                     "12 (the origins that have 12 actuals), which supports central " *
                     "levels up to 93.33. Pass levels <= 93.33, use horizon <= 2, or fit " *
                     "with holdout >= 50." backtest(fc, d; horizon=12, windows=2,
                                                    levels=(95,))
        @test FIT[] == 0
        fc20 = mk(CountFit(); c=20)
        deep = "backtest horizon 13 needs at least 9 calibration errors per step, " *
               "but intervals=SplitConformal(holdout=20) gives 8 at that horizon " *
               "(the origins that have 13 actuals). Use horizon <= 12, or fit with " *
               "holdout >= 21."
        @test_throws deep backtest(fc20, d; horizon=13, windows=2, metrics=(mae, crps))
        @test FIT[] == 0
        pd = mkpanel("a" => 1:60, "b" => 1:60)
        @test_throws deep backtest(pmk(CountFit(); c=20), pd; horizon=13, windows=2,
                                   metrics=(Coverage(80),))
        @test FIT[] == 0
        # A point-only call has no depth rule.
        @test backtest(fc20, d; horizon=13, windows=2) isa BacktestResult
        @test FIT[] == 2
        # The row check refuses before the first fit.
        FIT[] = 0
        @test_throws "backtest fold 1 (origin" backtest(
            mk(CountFit(); c=60), mkdata(66); horizon=3, windows=2, step=5)
        @test FIT[] == 0
    end

    @testset "a backtest fits nm models per fold, levels add only the triangle" begin
        d = mkdata(60)
        pd = mkpanel("a" => 1:60, "b" => 1:50)
        for (strat, nm) in ((Recursive(), 1), (Direct(3), 3))
            for (fc, data) in ((mk(CountFit(); c=20, strat), d),
                               (pmk(CountFit(); c=20, strat), pd))
                for kw in ((;), (levels=(80,),), (metrics=(mae, crps),),
                           (levels=(80,), metrics=(crps,)))
                    FIT[] = 0
                    backtest(fc, data; horizon=3, windows=3, kw...)
                    @test FIT[] == 3nm
                end
            end
        end
    end

    @testset "nothing in a fold sees values after its window" begin
        d, h = mkdata(90), 3
        late = (ds=d.ds, y=[d.y[1:84]; 1000 .* d.y[85:90]])   # after fold 1
        ms = (mae, crps, Coverage(95))
        for model in (LinAR(0.5, 1.0), dtree())
            fc = mk(model; c=45)
            bt(x) = backtest(fc, x; horizon=h, windows=3, levels=(95,), metrics=ms)
            r1, r2 = bt(d), bt(late)
            f1 = r1.forecasts.fold .== 1
            for c in (:y, :y_hat, :lo_95, :hi_95)
                @test r1.forecasts[c][f1] == r2.forecasts[c][f1]
            end
            s1 = r1.scores.fold .== 1
            @test r1.scores.value[s1] == r2.scores.value[s1]
            @test any(r1.scores.value[.!s1] .!= r2.scores.value[.!s1])
        end
        # Ragged folds: the head stops before the holdout.
        c = 20
        pd = mkpanel("a" => 1:90, "b" => 1:70)
        pm = pmk(MeanModel(); c)
        # Series b's holdout starts at day 42 in fold 1.
        moved = merge(pd, (y=[pd.unique_id[i] == "a" && pd.ds[i] ≥ day(42) ?
                              1000.0 + pd.y[i] : pd.y[i] for i in eachindex(pd.y)],))
        a, b = (backtest(pm, x; horizon=h, windows=3) for x in (pd, moved))
        f1 = a.forecasts.fold .== 1
        @test a.forecasts.y_hat[f1] == b.forecasts.y_hat[b.forecasts.fold .== 1]
        @test a.forecasts.y_hat[f1] == fill(headmean([(1, 1:41), (2, 1:41)], 1), 2h)
        @test a.forecasts.y_hat != b.forecasts.y_hat
    end

    @testset "single series: the first fold must fit and calibrate" begin
        feats = FeatureSet(Lag(13))
        fc = mk(MeanModel(); c=60, feats)
        d = mkdata(70)
        recursive = "backtest fold 1 (origin $(day(62))): intervals=SplitConformal" *
                    "(holdout=60) needs at least 74 training rows to fit and calibrate, " *
                    "but the fold trains on 62. Move the first fold later (raise " *
                    "initial, or reduce windows, step or horizon), lower holdout to 48 " *
                    "or less, or provide more data."
        @test_throws recursive backtest(fc, d; horizon=3, windows=2, step=5)
        @test_throws recursive backtest(fc, d; horizon=3, initial=62, step=5)
        @test_throws recursive backtest(fc, d; horizon=3, windows=2, step=5,
                                        metrics=(mae, crps), levels=(80,))
        direct = "backtest fold 1 (origin $(day(80))): intervals=SplitConformal" *
                 "(holdout=60) with Direct(12) needs at least 85 training rows " *
                 "(60 + 13 + 12), but the fold trains on 80. Move the first fold " *
                 "later (raise initial, or reduce windows, step or horizon), lower " *
                 "holdout to 55 or less, use Direct(7), or provide more data."
        fd = mk(MeanModel(); c=60, feats, strat=Direct(12))
        @test_throws direct backtest(fd, mkdata(83); horizon=3, windows=1)
        @test_throws direct backtest(fd, mkdata(83); horizon=3, initial=80)
        # Without it the error would not name the fold.
        @test_throws "intervals=SplitConformal(holdout=60) leaves 20 of the 80 rows" fit(
            fd, take(mkdata(83), 1:80))
        # Same bound as fit: c + mh fails, c + mh + 1 runs.
        for (tr, f_) in ((NoTransform(), FeatureSet(Lag(1))),
                         (Difference(1), FeatureSet(Lag(1), Lag(3))))
            g = mk(MeanModel(); c=20, feats=f_, transform=tr)
            lim = 20 + C.minhistory(g)
            dd = mkdata(lim + 3)
            @test caught(() -> fit(g, take(dd, 1:lim))) isa ArgumentError
            @test fit(g, take(dd, 1:(lim + 1))) isa FittedForecaster
            mh = C.minhistory(g)
            full = "backtest fold 1 (origin $(day(lim))): intervals=SplitConformal" *
                   "(holdout=20) needs at least $(lim + 1) training rows to fit and " *
                   "calibrate, but the fold trains on $lim. Move the first fold later " *
                   "(raise initial, or reduce windows, step or horizon), lower holdout " *
                   "to $(lim - mh - 1) or less, or provide more data."
            @test_throws full backtest(g, dd; horizon=3, windows=1)
            @test backtest(g, mkdata(lim + 4); horizon=3, windows=1) isa BacktestResult
        end
        # Clauses that cannot hold are dropped.
        tiny = mk(MeanModel(); c=9, feats)
        @test_throws "but the fold trains on 20. Move the first fold later (raise " *
                     "initial, or reduce windows, step or horizon), or provide more " *
                     "data." backtest(tiny, mkdata(23); horizon=3, windows=1)
        dt = mk(MeanModel(); c=20, strat=Direct(5))
        @test_throws "(20 + 1 + 5), but the fold trains on 21. Move the first fold " *
                     "later (raise initial, or reduce windows, step or horizon), lower " *
                     "holdout to 15 or less, or provide more data." backtest(
            dt, mkdata(24); horizon=3, windows=1)
        @test_throws "(20 + 1 + 5), but the fold trains on 25. Move the first fold " *
                     "later (raise initial, or reduce windows, step or horizon), lower " *
                     "holdout to 19 or less, use Direct(4), or provide more data." backtest(
            dt, mkdata(28); horizon=3, windows=1)
        @test backtest(dt, mkdata(29); horizon=3, windows=1) isa BacktestResult
    end

    @testset "a panel fold leaves out a series too short to calibrate" begin
        c, h = 20, 3
        warn1 = "backtest fold 1: leaving out 1 series with fewer than 22 rows up to " *
                "the fold's cut, the minimum to calibrate with intervals=" *
                "SplitConformal(holdout=20); the fold trains on the other series: " *
                "unique_id=\"s\" (21 rows)"
        d = mkpanel("a" => 1:60, "b" => 1:50, "s" => 1:27)
        rest = mkpanel("a" => 1:60, "b" => 1:50)
        for model in models(), kw in ((;), (levels=(80,), metrics=(mae, crps)))
            fc = pmk(model; c)
            r = @test_logs (:warn, warn1) backtest(fc, d; horizon=h, windows=2, kw...)
            ref = backtest(fc, rest; horizon=h, windows=2, kw...)
            f1, g1 = r.forecasts.fold .== 1, ref.forecasts.fold .== 1
            @test r.forecasts.unique_id[f1] == [fill("a", h); fill("b", h)]
            @test "s" in r.forecasts.unique_id[r.forecasts.fold .== 2]
            for col in keys(r.forecasts)
                col == :fold && continue
                @test r.forecasts[col][f1] == ref.forecasts[col][g1]
            end
            @test r.scores.value[r.scores.fold .== 1] ==
                  ref.scores.value[ref.scores.fold .== 1]
        end
        # y_hat does not depend on levels either.
        fc = pmk(MeanModel(); c)
        a = @test_logs (:warn, warn1) backtest(fc, d; horizon=h, windows=2)
        b = @test_logs (:warn, warn1) backtest(fc, d; horizon=h, windows=2,
                                               levels=(80,))
        @test a.forecasts.y_hat == b.forecasts.y_hat
        # Fold 1's head: the mean before b's holdout start.
        cutb = 50 - h - h
        @test a.forecasts.y_hat[a.forecasts.fold .== 1] ==
              fill(headmean([(1, 1:(cutb - c)), (2, 1:(cutb - c))], 1), 2h)
        # Over five ids: five are named, then an ellipsis.
        many = mkpanel("a" => 1:60, "s1" => 1:24, "s2" => 1:23, "s3" => 1:24,
                       "s4" => 1:22, "s5" => 1:21, "s6" => 1:24)
        m = "backtest fold 1: leaving out 6 series with fewer than 22 rows up to the " *
            "fold's cut, the minimum to calibrate with intervals=SplitConformal(" *
            "holdout=20); the fold trains on the other series: unique_id=\"s1\" (21 " *
            "rows), unique_id=\"s2\" (20 rows), unique_id=\"s3\" (21 rows), " *
            "unique_id=\"s4\" (19 rows), unique_id=\"s5\" (18 rows), …"
        @test_logs (:warn, m) backtest(pmk(MeanModel(); c), many; horizon=h, windows=1)
        # Direct: the dropped series must not set the cutoff.
        for model in models()
            fc = pmk(model; c, strat=Direct(3))
            r = @test_logs (:warn, warn1) backtest(fc, d; horizon=h, windows=2)
            ref = backtest(fc, rest; horizon=h, windows=2)
            f1, g1 = r.forecasts.fold .== 1, ref.forecasts.fold .== 1
            @test r.forecasts.y_hat[f1] == ref.forecasts.y_hat[g1]
        end
        # A series absent from fold 1's cuts is not named there.
        @test_logs (:warn, r"^backtest fold 2: leaving out 1 series") backtest(
            pmk(MeanModel(); c), mkpanel("a" => 1:60, "s" => 1:5); horizon=h, windows=2)
    end

    @testset "the panel bound is c + mh + 1 rows up to the cut" begin
        c = 20
        for (tr, feats) in ((NoTransform(), FeatureSet(Lag(1))),
                            (Difference(1), FeatureSet(Lag(1), Lag(3))))
            fc = pmk(MeanModel(); c, feats, transform=tr)
            mh = C.minhistory(fc)
            need = c + mh + 1
            # windows=1, h=3: n rows train on n - 3 in fold 1.
            d = mkpanel("a" => 1:60, "lo" => 1:(need + 2), "ok" => 1:(need + 3))
            warn = "backtest fold 1: leaving out 1 series with fewer than $need rows " *
                   "up to the fold's cut, the minimum to calibrate with intervals=" *
                   "SplitConformal(holdout=20); the fold trains on the other series: " *
                   "unique_id=\"lo\" ($(need - 1) rows)"
            for kw in ((;), (levels=(80,),))
                r = @test_logs (:warn, warn) backtest(fc, d; horizon=3, windows=1, kw...)
                @test unique(r.forecasts.unique_id) == ["a", "ok"]
                rest = take(d, d.unique_id .!= "lo")
                ref = backtest(fc, rest; horizon=3, windows=1, kw...)
                @test r.forecasts.y_hat == ref.forecasts.y_hat
            end
        end
    end

    @testset "horizon 1: K = holdout and nested intervals" begin
        c, h = 45, 1
        d = mkdata(90)
        pd = mkpanel("a" => 1:90, "b" => 1:80)
        q6(v, p) = quantile(v, p; alpha=0.0, beta=0.0)
        for model in (LinAR(0.5, 1.0), EvoTreeRegressor(nrounds=5), dtree()),
            (fc, data) in ((mk(model; c), d), (pmk(model; c), pd))
            kw = (horizon=h, windows=3)
            r = backtest(fc, data; kw..., levels=(80, 95))
            @test r.forecasts.y_hat == backtest(fc, data; kw...).forecasts.y_hat
            f = r.forecasts
            for c_ in (:lo_80, :hi_80, :lo_95, :hi_95)
                @test all(isfinite, f[c_])
            end
            @test all(f.lo_95 .≤ f.lo_80 .≤ f.hi_80 .≤ f.hi_95)
            # Fold 1: type 6 quantiles of y_hat + E[1:c, 1].
            sel = f.fold .== 1
            cut = Dict("a" => 87, "b" => 77)
            fit1 = fc.id === nothing ? fit(fc, take(data, 1:87)) :
                   fit(fc, take(data, [data.ds[i] ≤ day(cut[data.unique_id[i]])
                                       for i in eachindex(data.ds)]))
            E = getfield(fit1, :calibration)
            yh = forecast(fit1, h).y_hat
            for (j, e) in enumerate(E)
                v = yh[j] .+ e[1:c, 1]
                @test length(v) == c
                @test f.lo_80[sel][j] ≈ q6(v, 0.1)
                @test f.hi_95[sel][j] ≈ q6(v, 0.975)
            end
        end
    end

    @testset "a FromStart fold names its origin; late starters are aligned" begin
        c, h = 20, 3
        d = mkpanel("a" => 1:70, "s" => 41:70)
        a_only = take(d, d.unique_id .== "a")
        warn1 = "backtest fold 1 (origin $(day(50))): leaving out 1 series with fewer " *
                "than 22 rows up to the fold's cut, the minimum to calibrate with " *
                "intervals=SplitConformal(holdout=20); the fold trains on the other " *
                "series: unique_id=\"s\" (10 rows)"
        for model in models()
            fc = pmk(model; c)
            r = @test_logs (:warn, warn1) backtest(fc, d; horizon=h, initial=50, step=15)
            @test unique(r.forecasts.fold) == [1, 2]
            f1 = r.forecasts.fold .== 1
            @test all(==("a"), r.forecasts.unique_id[f1])
            ref = fit(fc, take(a_only, a_only.ds .≤ day(50)))
            @test r.forecasts.y_hat[f1] == forecast(ref, h).y_hat
            # Fold 2: the late series has 25 rows, joins.
            f2 = r.forecasts.fold .== 2
            both = fit(fc, take(d, d.ds .≤ day(65)))
            @test r.forecasts.unique_id[f2] == [fill("a", h); fill("s", h)]
            @test r.forecasts.y_hat[f2] == forecast(both, h).y_hat
        end
        # A series with no rows up to the cut is not named.
        late = mkpanel("a" => 1:70, "s" => 56:70)
        @test_logs backtest(pmk(MeanModel(); c), late; horizon=h, initial=50, step=100)
        # A series that ended before the cut sets T*.
        ended = mkpanel("a" => 1:60, "e" => 1:40)
        for kw in ((;), (levels=(80,),))
            r = backtest(pmk(MeanModel(); c), ended; horizon=h, initial=50, step=100,
                         kw...)
            @test unique(r.forecasts.fold) == [1]
            @test r.forecasts.unique_id == fill("a", h)
            # e's holdout starts at day 21, a's at day 31.
            @test r.forecasts.y_hat == fill(headmean([(1, 1:20), (2, 1:20)], 1), h)
        end
    end

    @testset "a fold with no series that can calibrate is refused" begin
        d = mkpanel("a" => 1:30, "b" => 1:29)
        msg = "backtest fold 1: intervals=SplitConformal(holdout=25) needs at least 27 " *
              "rows in a series up to the fold's cut, but the longest has 24. Move the " *
              "first fold later (raise initial, or reduce windows, step or horizon), " *
              "lower holdout to 22 or less, or provide longer series."
        for model in (MeanModel(), dtree(), EvoTreeRegressor(nrounds=5))
            fc = pmk(model; c=25)
            @test_throws msg backtest(fc, d; horizon=3, windows=2)
            @test_throws msg backtest(fc, d; horizon=3, windows=2, levels=(80,))
        end
        # FromStart names the origin.
        @test_throws "backtest fold 1 (origin $(day(24))): intervals=" backtest(
            pmk(MeanModel(); c=25), d; horizon=3, initial=24)
        # Few rows left: no lower-holdout clause below 9.
        @test_throws "up to the fold's cut, but the longest has 10. Move the first " *
                     "fold later (raise initial, or reduce windows, step or horizon), " *
                     "or provide longer series." backtest(
            pmk(MeanModel(); c=9), mkpanel("a" => 1:13, "b" => 1:12); horizon=3,
            windows=1)
    end

    @testset "series with at most mh rows stay with the point fit" begin
        # b has one row in fold 1 (mh is 1), four in fold 2.
        d = mkpanel("a" => 1:60, "b" => 1:7)
        fc = pmk(MeanModel(); c=20)
        warn2 = "backtest fold 2: leaving out 1 series with fewer than 22 rows up to " *
                "the fold's cut, the minimum to calibrate with intervals=" *
                "SplitConformal(holdout=20); the fold trains on the other series: " *
                "unique_id=\"b\" (4 rows)"
        r = @test_logs (:warn, r"skipping 1 series with fewer than 2 rows") (
            :warn, warn2) backtest(fc, d; horizon=3, windows=2)
        @test unique(r.forecasts.unique_id) == ["a"]
    end

    @testset "the Direct head bound names the fold" begin
        c, h = 20, 3
        d = mkpanel("a" => 1:60, "b" => 1:28)
        fc = pmk(MeanModel(); c, strat=Direct(3))
        want = "backtest fold 1: intervals=SplitConformal(holdout=20) trains the " *
               "calibration model on rows before $(day(3)), the earliest holdout start " *
               "(series unique_id=\"b\"). That leaves 2 rows in the series with the " *
               "most rows before it, 1 after the feature set's 1 history row, but " *
               "Direct(3) needs 3. Lower holdout to 18 or less, use Direct(1) or a " *
               "shorter max_horizon, or provide 2 more rows before $(day(3)) in that " *
               "series."
        @test_throws want backtest(fc, d; horizon=h, windows=2)
        @test_throws want backtest(fc, d; horizon=h, windows=2, levels=(80,))
        # FromStart names the origin.
        ok = mkpanel("a" => 1:60, "b" => 1:60)
        @test backtest(fc, ok; horizon=h, initial=45, step=15) isa BacktestResult
        @test_throws "backtest fold 1 (origin $(day(24))): intervals=SplitConformal" (
            backtest(pmk(MeanModel(); c, strat=Direct(30)), ok; horizon=h, initial=24))
    end

    @testset "folds where a series has fewer than horizon steps" begin
        c, h = 20, 6
        feats = FeatureSet(Lag(1), Exogenous(:x))
        d = mkpanel("a" => 1:60, "b" => 1:52; x=true)
        for model in (LinAR(0.5, 1.0), EvoTreeRegressor(nrounds=5), dtree())
            fc = pmk(model; c, feats)
            r = backtest(fc, d; horizon=h, initial=48, step=3, levels=(80,))
            @test [count(==(k), r.forecasts.fold) for k in 1:3] == [10, 7, 6]
            @test all(isfinite, r.forecasts.lo_80) && all(isfinite, r.forecasts.hi_80)
            @test r.forecasts.y_hat == backtest(fc, d; horizon=h, initial=48,
                                                step=3).forecasts.y_hat
            # One K for every series, from the longest call.
            f = fit(fc, take(d, d.ds .≤ day(48)))
            nd = take(d, d.ds .> day(48))
            _, dr = C._forecast_samples(f, [h, 4], (unique_id=nd.unique_id, ds=nd.ds,
                                                    x=nd.x))
            @test size.(dr, 2) == [c - h + 1, c - h + 1]
            @test size.(dr, 1) == [h, 4]
            los, his = C._interval_bounds(reduce(vcat, dr), [80.0])
            f1 = r.forecasts.fold .== 1
            @test r.forecasts.lo_80[f1] == los[1] && r.forecasts.hi_80[f1] == his[1]
        end
        # No actual after the cut: the existing error.
        apart = mkpanel("a" => 1:50, "s" => 51:60; x=true)
        text = "backtest fold 1 (origin $(day(50))) produced no forecast that lines up " *
               "with an actual: no series trained on data up to the origin has data " *
               "after it. Choose initial and step so each origin falls inside the data " *
               "of a series."
        for model in (MeanModel(), dtree()), kw in ((;), (levels=(80,),)), fts in (
            FeatureSet(Lag(1)), feats)
            fc = pmk(model; c=12, feats=fts)
            @test_throws text backtest(fc, apart; horizon=3, initial=50, step=100, kw...)
        end
    end

    @testset "the hooks" begin
        fc = pmk(MeanModel(); c=20)
        d = mkpanel("a" => 1:60, "s" => 1:30)
        tbl = C.normalize_table(d)
        groups = C.panel_groups(fc, tbl)
        all_rows = collect(eachindex(tbl.y))
        iv = fc.intervals
        cut = Dict("a" => day(50), "s" => day(30))
        # Nothing to drop: the very same vector comes back.
        @test C._fold_train_rows(iv, fc, groups, cut, all_rows, 1, day(50)) === all_rows
        short = Dict("a" => day(50), "s" => day(21))
        rows = C._rows_before(short, tbl.unique_id, tbl.ds)
        kept = @test_logs (:warn, r"leaving out 1 series") C._fold_train_rows(
            iv, fc, groups, short, rows, 2, missing)
        @test kept isa Vector{Int}
        @test kept == [i for i in rows if tbl.unique_id[i] == "a"]
        @test rows == C._rows_before(short, tbl.unique_id, tbl.ds)
        @test_logs (:warn, r"^backtest fold 2 \(origin 2020-02-19\): leaving out") (
            C._fold_train_rows(iv, fc, groups, short, rows, 2, day(50)))
        sfc = mk(MeanModel(); c=20)
        t = mkdata(80).ds
        @test C._check_fold_rows(siv(20), sfc, [22, 30], t) === nothing
        @test_throws "backtest fold 1 (origin $(t[21])): intervals=SplitConformal" (
            C._check_fold_rows(siv(20), sfc, [21, 30], t))
    end

    @testset "a coverage by step table from a levels backtest" begin
        d = mkdata(120)
        fc = mk(dtree(); c=60, feats=FeatureSet(Lag(1), Lag(2)))
        r = backtest(fc, d; horizon=4, initial=80, step=4, levels=(80,))
        f = r.forecasts
        inside = f.lo_80 .≤ f.y .≤ f.hi_80
        by_step = [sum(inside[f.step .== j]) / count(==(j), f.step) for j in 1:4]
        @test length(by_step) == 4 && all(x -> 0 ≤ x ≤ 1, by_step)
        @test all(f.lo_80 .< f.hi_80)
    end

    @testset "plain and CSP backtests are unchanged" begin
        # Literals of b8ea2cb, equal at 69c57ef, --check-bounds=yes.
        t = collect(Date(2020, 1, 1):Month(1):Date(2022, 12, 1))
        y = [10.0 + mod1(i, 12) + i / 4 for i in 1:36]
        pfc = Forecaster(LinAR(0.5, 1.0); features=FeatureSet(Lag(1)), freq=Month(1),
                         id=:unique_id)
        pcsp = ConformalSeasonalPool(12; freq=Month(1), rng=StableRNG(9), id=:unique_id)
        ends = (unique_id=[fill("a", 36); fill("b", 8)], ds=[t; t[1:8]],
                y=[y; 2 .* y[1:8]])
        late = (unique_id=[fill("a", 36); fill("c", 10)], ds=[t; t[27:36]],
                y=[y; 2 .* y[27:36]])
        r = backtest(pfc, ends; horizon=3, windows=3)
        @test r.forecasts.y_hat == [10.875, 6.4375, 4.21875, 12.75, 7.375, 4.6875, 13.5,
                                    7.75, 4.875, 14.625, 8.3125, 5.15625, 17.25, 9.625,
                                    5.8125]
        @test r.forecasts.fold == [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        @test r.forecasts.unique_id == ["a", "a", "a", "a", "a", "a", "b", "b", "b",
                                        "a", "a", "a", "b", "b", "b"]
        @test r.scores.value ≈ [15.072916666666666, 15.53831160725536, 19.510416666666668,
                                20.22440834013033, 23.494791666666668, 24.429642945141122]
        r = backtest(pfc, late; horizon=3, initial=4, step=6)
        @test r.forecasts.y_hat == [8.5, 5.25, 3.625, 12.25, 7.125, 4.5625, 10.0, 6.0,
                                    4.0, 13.75, 7.875, 4.9375, 11.5, 6.75, 4.375, 22.0,
                                    12.0, 7.0]
        @test r.forecasts.fold == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5]
        @test r.forecasts.unique_id == [fill("a", 15); fill("c", 3)]
        @test r.scores.value ≈ [11.708333333333334, 12.095324509908776,
                                13.020833333333334, 13.48595913348398,
                                13.833333333333334, 14.269723192830336,
                                15.145833333333334, 15.552205430420472,
                                24.645833333333332, 26.922003469900478]
        r = @test_logs (:warn, r"unique_id=\"b\" \(2 rows\)") (
            :warn, r"unique_id=\"b\" \(5 rows\)") backtest(pcsp, ends; horizon=3,
                                                            windows=3, levels=(80,))
        @test r.forecasts.y_hat == [18.0, 19.25, 20.5, 21.75, 23.0, 24.25, 25.5, 26.75,
                                    28.0]
        @test r.forecasts.fold == [1, 1, 1, 2, 2, 2, 3, 3, 3]
        @test r.scores.value == fill(3.0, 6)
        r = @test_logs (:warn, r"unique_id=\"c\" \(6 rows\)") backtest(
            pcsp, late; horizon=3, initial=14, step=6, levels=(80,))
        @test r.forecasts.y_hat == [13.75, 15.0, 16.25, 21.25, 22.5, 23.75, 16.75, 18.0,
                                    19.25, 24.25, 25.5, 26.75]
        @test r.forecasts.fold == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        @test r.scores.value == fill(3.0, 8)
    end

    @testset "the docstrings describe the backtest of a SplitConformal forecaster" begin
        doc = replace(string(@doc(backtest)), r"\s+" => " ")
        @test occursin("`SplitConformal`", doc)
        @test occursin("every fold fits the published head", doc)
        @test occursin("`y_hat` does not depend on `levels`", doc)
        @test occursin("a fold that drops a series too short to calibrate trains on " *
                       "fewer series", doc)
        @test !occursin("Every fold draws from its own copy of the method's `rng`", doc)
        @test !occursin("Each fold trains on every row at or before the origin " *
                        "timestamp, forecasts", doc)
        sdoc = replace(string(@doc(SplitConformal)), r"\s+" => " ")
        @test occursin("A fold of a [`backtest`](@ref) that drops a series too short to " *
                       "calibrate trains on fewer series", sdoc)
    end
end

@testset "SplitConformal is exported and documented" begin
    @test :SplitConformal in names(MachineLearningForecast)
    doc = replace(string(@doc(SplitConformal)), r"\s+" => " ")
    @test occursin("The published forecast comes from the head model", doc)
    @test occursin("differs from a forecaster without `intervals`", doc)
    @test occursin("no coverage guarantee", doc)
    @test !occursin(r"\bvalid\b|\bguaranteed\b", doc)
    for d in (@doc(Forecaster), @doc(forecast), @doc(forecast_samples), @doc(SeriesState),
              @doc(MachineLearningForecast))
        @test occursin("SplitConformal", string(d))
    end
    fc = Forecaster(TestModels.LinAR(0.5, 1.0); features=FeatureSet(Lag(1)),
                    freq=Month(1), intervals=SplitConformal(holdout=60))
    @test occursin("intervals=SplitConformal(holdout=60)", sprint(show, fc))
end
