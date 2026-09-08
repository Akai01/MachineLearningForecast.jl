# Guards added after an end-to-end audit: input validation, leakage refusal,
# reserved column names, and the exact-value coverage of the single-row
# (forecast-time) paths that batch materialization alone cannot pin.

@testset "validation and guards" begin
    df = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19)), y=Float64.(1:50))
    fs = FeatureSet(Lag(1))
    mk(; kwargs...) = Forecaster(TestModels.LinAR(1.0, 0.0);
                                 features=get(kwargs, :features, fs),
                                 strategy=Recursive(), freq=Day(1),
                                 target=get(kwargs, :target, :y),
                                 time=get(kwargs, :time, :ds))

    @testset "a feature emitting the target column is refused (total leak)" begin
        err = try mk(features=FeatureSet(Lag(1), Exogenous(:y))) catch e; e end
        @test err isa ArgumentError
        @test occursin("column named :y", err.msg)
        @test occursin("leak", err.msg)
        # a custom feature named after the target is caught too
        @test_throws ArgumentError mk(features=FeatureSet(Lag(1),
                                        CustomFeature(:y, length, 0)))
        # ...and naming the time column is refused with its own advice
        err2 = try mk(features=FeatureSet(Lag(1), Exogenous(:ds))) catch e; e end
        @test err2 isa ArgumentError
        @test occursin("Calendar", err2.msg)
    end

    @testset "time column may not shadow reserved result columns" begin
        for bad in (:origin, :step, :y_hat)
            err = try mk(target=:v, time=bad) catch e; e end
            @test err isa ArgumentError
            @test occursin("reserved", err.msg)
        end
        @test_throws ArgumentError mk(target=:v, time=:y)   # collides with folds.y
    end

    @testset "ragged tables are rejected by name and length" begin
        err = try fit(mk(), (ds=df.ds, y=df.y[1:49])) catch e; e end
        @test err isa ArgumentError
        @test occursin("unequal lengths", err.msg)
        @test occursin(":y has 49 rows", err.msg)
    end

    @testset "non-finite targets are rejected, not propagated" begin
        for bad in (NaN, Inf, -Inf)
            y = copy(df.y); y[10] = bad
            err = try fit(mk(), (ds=df.ds, y=y)) catch e; e end
            @test err isa ArgumentError
            @test occursin("non-finite", err.msg)
            @test occursin("row 10", err.msg)
        end
    end

    @testset "time column type must support + freq" begin
        err = try fit(mk(), (ds=string.(df.ds), y=df.y)) catch e; e end
        @test err isa ArgumentError
        @test occursin("does not support", err.msg)
        @test occursin("String", err.msg)
    end

    @testset "month-end series are gap-free (anchored grid)" begin
        me = [Date(2020, 1, 31) + Month(k) for k in 0:23]
        @test me[2] == Date(2020, 2, 29) && me[3] == Date(2020, 3, 31)
        # stepping from the previous row would see Feb 29 + Month(1) = Mar 29
        # and report a spurious gap; anchoring at t[1] does not.
        @test MachineLearningForecast.validate_time_column(me, :ds, Month(1)) === nothing
        fcm = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Recursive(),
                         freq=Month(1), target=:y, time=:ds)
        out = forecast(fit(fcm, (ds=me, y=Float64.(1:24))), 3)
        @test out.ds == [Date(2022, 1, 31), Date(2022, 2, 28), Date(2022, 3, 31)]
        # a genuine gap is still caught
        gapped = vcat(me[1:5], me[7:end])
        @test_throws ArgumentError MachineLearningForecast.validate_time_column(gapped, :ds, Month(1))
    end

    @testset "Calendar sub-daily parts need a DateTime column" begin
        err = try fit(mk(features=FeatureSet(Lag(1), Calendar(:hour))), df) catch e; e end
        @test err isa ArgumentError
        @test occursin("sub-daily", err.msg)
        @test occursin("hour", err.msg)
        # and they work when the column really is a DateTime
        hds = collect(DateTime(2022, 1, 1):Hour(1):DateTime(2022, 1, 1) + Hour(49))
        fch = Forecaster(TestModels.EchoColumn(:hour);
                         features=FeatureSet(Lag(1), Calendar(:hour)),
                         strategy=Recursive(), freq=Hour(1), target=:y, time=:ds)
        got = forecast(fit(fch, (ds=hds, y=Float64.(1:50))), 3)
        @test got.y_hat == Float64.(hour.(hds[end] .+ Hour.(1:3)))
    end

    @testset "new_data: duplicates and eltype mismatches are diagnosed" begin
        fce = Forecaster(TestModels.EchoColumn(:promo);
                         features=FeatureSet(Lag(1), Exogenous(:promo)),
                         strategy=Recursive(), freq=Day(1), target=:y, time=:ds)
        fitted = fit(fce, (ds=df.ds, y=df.y, promo=fill(1.0, 50)))
        grid = collect(df.ds[end] + Day(1):Day(1):df.ds[end] + Day(3))

        dup = (ds=[grid[1], grid[1], grid[2], grid[3]], promo=[1.0, 2.0, 3.0, 4.0])
        err = try forecast(fitted, 3; new_data=dup) catch e; e end
        @test err isa ArgumentError
        @test occursin("duplicate timestamps", err.msg)

        mism = (ds=DateTime.(grid), promo=[1.0, 2.0, 3.0])
        err2 = try forecast(fitted, 3; new_data=mism) catch e; e end
        @test err2 isa ArgumentError
        @test occursin("element type", err2.msg)   # not a bogus "missing timestamps"
        @test !occursin("missing", err2.msg)

        # out-of-order new_data joins correctly (join is by timestamp, not position)
        shuffled = (ds=grid[[3, 1, 2]], promo=[30.0, 10.0, 20.0])
        @test forecast(fitted, 3; new_data=shuffled).y_hat == [10.0, 20.0, 30.0]
    end

    @testset "FeatureSet accepts a plain vector of features" begin
        @test FeatureSet([Lag(k) for k in 1:3]) == FeatureSet(Lag(1), Lag(2), Lag(3))
        @test FeatureSet([Calendar(:dayofweek), Fourier(7, 1)]) isa FeatureSet
        @test FeatureSet(MachineLearningForecast.AbstractFeature[Lag(1)]) isa FeatureSet
    end

    @testset "single-row (forecast-time) paths: exact values" begin
        y = Float64[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        t = collect(Date(2021, 3, 1):Day(1):Date(2021, 3, 10))
        # Calendar: featurevalues must agree with the batch path at the same row
        cal = Calendar(:dayofweek, :month, :weekofyear)
        @test collect(MachineLearningForecast.featurevalues(cal, y, t[7], nothing)) ==
              Float64[dayofweek(t[7]), month(t[7]), week(t[7])]
        # Diff: y_{t-lag} - y_{t-lag-k} continuing the history
        @test only(MachineLearningForecast.featurevalues(Diff(3; lag=2), y, t[1], nothing)) ==
              y[10 + 1 - 2] - y[10 + 1 - 2 - 3]

        # ...and end-to-end through forecast(), which is what actually matters
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19))
        d2 = (ds=ds, y=Float64.(1:50))
        fc_dw = Forecaster(TestModels.EchoColumn(:dayofweek);
                           features=FeatureSet(Lag(1), Calendar(:dayofweek)),
                           strategy=Recursive(), freq=Day(1))
        @test forecast(fit(fc_dw, d2), 4).y_hat ==
              Float64.(dayofweek.(ds[end] .+ Day.(1:4)))
        fc_df = Forecaster(TestModels.EchoColumn(:y_diff_1_lag_1);
                           features=FeatureSet(Lag(1), Diff(1)),
                           strategy=Recursive(), freq=Day(1))
        # Closed-form oracle for the recursive feedback loop. The model echoes
        # Diff(1) = y_hist[end] - y_hist[end-1], and its own output is appended
        # to the history, so with y = 1:50:
        #   step 1: 50 - 49 =   1   (history ... 49, 50)
        #   step 2:  1 - 50 = -49   (history ... 50,  1)
        #   step 3: -49 -  1 = -50
        #   step 4: -50 - -49 =  -1
        @test forecast(fit(fc_df, d2), 4).y_hat == [1.0, -49.0, -50.0, -1.0]
    end

    @testset "mase is usable as a backtest/tune metric" begin
        fcb = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Recursive(),
                         freq=Day(1))
        res = backtest(fcb, df; horizon=7, initial=30, step=10, metrics=(mae, mase))
        @test :mase in res.metrics.metric
        vals = [res.metrics.value[i] for i in eachindex(res.metrics.fold)
                if res.metrics.metric[i] == :mase && res.metrics.fold[i] > 0]
        @test all(isfinite, vals) && all(>(0), vals)
        # the trait is what routes it; user metrics can opt in the same way
        @test MachineLearningForecast.needs_ytrain(mase)
        @test !MachineLearningForecast.needs_ytrain(mae)
    end
end
