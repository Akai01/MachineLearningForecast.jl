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
