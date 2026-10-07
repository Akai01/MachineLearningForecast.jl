@testset "strategies" begin
    df = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19)), y=Float64.(1:50))

    @testset "Direct construction" begin
        @test_throws ArgumentError Direct()      # must explain why a horizon is needed
        err = try Direct() catch e; e end
        @test occursin("max_horizon", err.msg)
        @test occursin("one model per horizon step", err.msg)
        @test_throws ArgumentError Direct(0)
        @test_throws "Direct max_horizon must be ≥ 1, got 0. Pass the number of steps " *
                     "to forecast, e.g. Direct(28)." Direct(0)
        @test_throws ArgumentError Direct(-5)
        @test_throws "max_horizon must be ≥ 1, got -5" Direct(-5)
        @test Direct(3).max_horizon == 3
    end

    @testset "machine counts" begin
        fs = FeatureSet(Lag(1))
        fc_r = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Recursive(),
                          freq=Day(1))
        @test length(fit(fc_r, df).machines) == 1
        fc_d = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Direct(3),
                          freq=Day(1))
        @test length(fit(fc_d, df).machines) == 3
    end

    @testset "Direct(3) refuses h=5 with an actionable message" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        strategy=Direct(3), freq=Day(1))
        fitted = fit(fc, df)
        @test_throws ArgumentError forecast(fitted, 5)
        err = try forecast(fitted, 5) catch e; e end
        @test occursin("max_horizon=3", err.msg)
        @test occursin("forecast(h=5)", err.msg)
        @test occursin("Recursive()", err.msg)
        @test length(forecast(fitted, 2).y_hat) == 2
    end

    @testset "Direct target alignment" begin
        # Echoing y_lag_1, every machine predicts the last value.
        fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=FeatureSet(Lag(1)),
                        strategy=Direct(4), freq=Day(1))
        fcast = forecast(fit(fc, df), 4)
        @test all(fcast.y_hat .== 50.0)
        tiny = (ds=df.ds[1:4], y=df.y[1:4])
        fc_big = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(2)),
                            strategy=Direct(4), freq=Day(1))
        @test_throws ArgumentError fit(fc_big, tiny)
        @test_throws "not enough data for Direct(4): after dropping the feature set's 2 " *
                     "history rows, only 2 training rows remain" fit(fc_big, tiny)
        @test_throws "Provide at least 6 rows, reduce max_horizon" fit(fc_big, tiny)

        # MeanModel shows the shift: machine i's mean is (51+i)/2.
        fc_mean = Forecaster(TestModels.MeanModel(); features=FeatureSet(Lag(1)),
                             strategy=Direct(4), freq=Day(1))
        @test forecast(fit(fc_mean, df), 4).y_hat ≈ [(51 + i) / 2 for i in 1:4]
        # Distinct per step, so a wrong shift would fail.
        @test length(unique(forecast(fit(fc_mean, df), 4).y_hat)) == 4
    end

    @testset "Direct anchors time/exogenous columns to the target row" begin
        # Machine i must train on its target row's exogenous values.
        n = 60
        dd = (ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1)),
              y  = Float64.(1:n) .+ 1000,
              promo = Float64.(1:n))              # distinct exogenous values
        fc = Forecaster(TestModels.PairLookup(:promo);
                        features=FeatureSet(Lag(1), Exogenous(:promo)),
                        strategy=Direct(4), freq=Day(1))
        fitted = fit(fc, dd)
        future = (ds = collect(dd.ds[end] + Day(1):Day(1):dd.ds[end] + Day(4)),
                  promo = [10.0, 20.0, 30.0, 40.0])
        got = forecast(fitted, 4; new_data=future).y_hat
        # Correct anchoring maps promo v to 1000+v on every machine.
        @test got ≈ [1010.0, 1020.0, 1030.0, 1040.0]
        # The old, skewed anchoring produced 1000+v+(i-1) instead:
        @test got != [1010.0, 1021.0, 1032.0, 1043.0]

        # A TimeFeature must be recoverable at every step too.
        dw = (ds = dd.ds, y = Float64.(dayofweek.(dd.ds)))
        fc_cal = Forecaster(TestModels.PairLookup(:dayofweek);
                            features=FeatureSet(Lag(1), Calendar(:dayofweek)),
                            strategy=Direct(4), freq=Day(1))
        fut_ds = collect(dw.ds[end] + Day(1):Day(1):dw.ds[end] + Day(4))
        @test forecast(fit(fc_cal, dw), 4).y_hat ≈ Float64.(dayofweek.(fut_ds))
    end

    @testset "Direct(1): one machine and horizon 1" begin
        @test_throws MethodError Direct(2.5)
        fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=FeatureSet(Lag(1)),
                        strategy=Direct(1), freq=Day(1))
        @test forecast(fit(fc, df), 1).y_hat == [50.0]
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fs = FeatureSet(Lag(1), Lag(7))
            f1 = fit(Forecaster(m; features=fs, strategy=Direct(1), freq=Day(1)), df)
            @test length(f1.machines) == 1
            f3 = fit(Forecaster(m; features=fs, strategy=Direct(3), freq=Day(1)), df)
            @test forecast(f1, 1).y_hat == forecast(f3, 3).y_hat[1:1]
            @test_throws ArgumentError forecast(f1, 2)
            @test_throws "strategy=Direct(1) was fit with max_horizon=1 but " *
                         "forecast(h=2) was requested" forecast(f1, 2)
            tiny = (ds=df.ds[1:4], y=df.y[1:4])
            fct = Forecaster(m; features=FeatureSet(Lag(3)), strategy=Direct(1),
                             freq=Day(1))
            @test length(forecast(fit(fct, tiny), 1).y_hat) == 1
            @test_throws "the data has only 3 rows" fit(fct, (ds=tiny.ds[1:3],
                                                              y=tiny.y[1:3]))
        end
    end
end
