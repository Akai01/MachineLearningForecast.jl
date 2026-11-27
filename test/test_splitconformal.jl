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
