@testset "strategies" begin
    df = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19)), y=Float64.(1:50))

    @testset "Direct construction" begin
        @test_throws ArgumentError Direct()      # must explain why a horizon is needed
        err = try Direct() catch e; e end
        @test occursin("max_horizon", err.msg)
        @test occursin("one model per horizon step", err.msg)
        @test_throws ArgumentError Direct(0)
        @test_throws ArgumentError Direct(-5)
        @test Direct(3).max_horizon == 3
    end

    @testset "machine counts" begin
        fs = FeatureSet(Lag(1))
        fc_r = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Recursive(), freq=Day(1))
        @test length(fit(fc_r, df).machines) == 1
        fc_d = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Direct(3), freq=Day(1))
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
        # h ≤ max_horizon is fine
        @test length(forecast(fitted, 2).y_hat) == 2
    end

    @testset "Direct target alignment" begin
        # EchoColumn(:y_lag_1) predicts its lag-1 feature, so under Direct each
        # machine's prediction is exactly the last training value.
        fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=FeatureSet(Lag(1)),
                        strategy=Direct(4), freq=Day(1))
        fcast = forecast(fit(fc, df), 4)
        @test all(fcast.y_hat .== 50.0)
        # Direct needs at least minhistory + max_horizon rows
        tiny = (ds=df.ds[1:4], y=df.y[1:4])
        fc_big = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(2)),
                            strategy=Direct(4), freq=Day(1))
        @test_throws ArgumentError fit(fc_big, tiny)

        # ---- which TARGET does machine i actually see? -----------------------
        # MeanModel predicts the mean of its own training target, so the shift
        # is directly observable. Lag(1) drops row 1, leaving targets 2:50, and
        # machine i trains on targets (i+1):50 -> mean (51+i)/2.
        fc_mean = Forecaster(TestModels.MeanModel(); features=FeatureSet(Lag(1)),
                             strategy=Direct(4), freq=Day(1))
        @test forecast(fit(fc_mean, df), 4).y_hat ≈ [(51 + i) / 2 for i in 1:4]
        # sanity: those are genuinely different per step, so the test can fail
        @test length(unique(forecast(fit(fc_mean, df), 4).y_hat)) == 4
    end

    @testset "Direct anchors time/exogenous columns to the target row" begin
        # Regression test for a (step-1) train/serve skew: machine i must be
        # trained with the time/exogenous values of the row it predicts, since
        # that is what forecast() feeds it. PairLookup exposes the mapping.
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
        # Correct anchoring memorises promo v => y 1000+v for EVERY machine.
        @test got ≈ [1010.0, 1020.0, 1030.0, 1040.0]
        # The old, skewed anchoring produced 1000+v+(i-1) instead:
        @test got != [1010.0, 1021.0, 1032.0, 1043.0]

        # And the same must hold for a pure TimeFeature: y = dayofweek(ds)
        # is recoverable at every horizon step, not just step 1.
        dw = (ds = dd.ds, y = Float64.(dayofweek.(dd.ds)))
        fc_cal = Forecaster(TestModels.PairLookup(:dayofweek);
                            features=FeatureSet(Lag(1), Calendar(:dayofweek)),
                            strategy=Direct(4), freq=Day(1))
        fut_ds = collect(dw.ds[end] + Day(1):Day(1):dw.ds[end] + Day(4))
        @test forecast(fit(fc_cal, dw), 4).y_hat ≈ Float64.(dayofweek.(fut_ds))
    end
end
