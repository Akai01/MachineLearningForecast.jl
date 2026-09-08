@testset "forecaster" begin
    n = 60
    t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
    y = Float64.(1:n)
    df = (ds=t, y=y)

    @testset "construction validation" begin
        @test_throws ArgumentError Forecaster("not a model";
            features=FeatureSet(Lag(1)), freq=Day(1))
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0);
            features=FeatureSet(Lag(1)), freq=Day(1), target=:y, time=:y)
        # non-Deterministic model and unknown target scitype both warn, not error
        fc = @test_logs (:warn, r"not an MLJModelInterface.Deterministic") (:warn, r"target_scitype") Forecaster(
            TestModels.DummyProb(); features=FeatureSet(Lag(1)), freq=Day(1))
        @test fc isa Forecaster
    end

    @testset "recursive oracle: LinAR closed-form iteration" begin
        # ŷ_{n+1} = a*y_n + b, then iterated on its own predictions.
        a, b = 0.8, 2.0
        fc = Forecaster(TestModels.LinAR(a, b); features=FeatureSet(Lag(1)),
                        strategy=Recursive(), freq=Day(1))
        fitted = fit(fc, df)
        h = 10
        fcast = forecast(fitted, h)
        expected = Float64[]
        prev = y[end]
        for _ in 1:h
            prev = a * prev + b
            push!(expected, prev)
        end
        @test fcast.y_hat ≈ expected
        @test fcast.ds == [t[end] + Day(s) for s in 1:h]
        @test keys(fcast) == (:ds, :y_hat)
    end

    @testset "recursive oracle: naive ŷ = y_lag_1" begin
        fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=FeatureSet(Lag(1)),
                        strategy=Recursive(), freq=Day(1))
        fcast = forecast(fit(fc, df), 5)
        @test all(fcast.y_hat .== y[end])   # naive carries the last value forward
    end

    @testset "recursive feedback reaches rolling windows" begin
        # EchoColumn(:y_rollmean_2_lag_1) forecasts the mean of the last two
        # history values, which after step 1 includes a prediction.
        fc = Forecaster(TestModels.EchoColumn(:y_rollmean_2_lag_1);
                        features=FeatureSet(Lag(1), RollingMean(2)),
                        strategy=Recursive(), freq=Day(1))
        fcast = forecast(fit(fc, df), 3)
        e1 = (y[end-1] + y[end]) / 2
        e2 = (y[end] + e1) / 2
        e3 = (e1 + e2) / 2
        @test fcast.y_hat ≈ [e1, e2, e3]
    end

    @testset "Fourier continuity across the train/forecast boundary" begin
        f = Fourier(7, 1)
        name = MachineLearningForecast.outputnames(f)[1]   # the sin_1 column
        fc = Forecaster(TestModels.EchoColumn(name);
                        features=FeatureSet(Lag(1), f), strategy=Recursive(), freq=Day(1))
        fcast = forecast(fit(fc, df), 14)
        # training rows have index 0..n-1, so forecast step s has index n-1+s
        expected = [sin(2π * (n - 1 + s) / 7) for s in 1:14]
        @test fcast.y_hat ≈ expected
        # the series y_t = sin(2πt/7) continues with no phase jump:
        full = [sin(2π * k / 7) for k in 0:(n + 13)]
        @test fcast.y_hat ≈ full[(n + 1):(n + 14)]
    end

    @testset "exogenous handling" begin
        dfe = (ds=t, y=y, promo=Float64.(iseven.(1:n)))
        fc = Forecaster(TestModels.EchoColumn(:promo);
                        features=FeatureSet(Lag(1), Exogenous(:promo)),
                        strategy=Recursive(), freq=Day(1))
        fitted = fit(fc, dfe)

        # missing new_data → error naming the column and the needed range
        err = try forecast(fitted, 3) catch e; e end
        @test err isa ArgumentError
        @test occursin("Exogenous(:promo)", err.msg)
        @test occursin("new_data", err.msg)
        @test occursin(string(t[end] + Day(1)), err.msg)
        @test occursin(string(t[end] + Day(3)), err.msg)

        # missing column → error naming it
        bad_cols = (ds=t[end] .+ Day.(1:3), other=zeros(3))
        err = try forecast(fitted, 3; new_data=bad_cols) catch e; e end
        @test err isa ArgumentError && occursin(":promo", err.msg)

        # missing timestamps → error listing the first missing one
        short = (ds=t[end] .+ Day.(1:2), promo=zeros(2))
        err = try forecast(fitted, 3; new_data=short) catch e; e end
        @test err isa ArgumentError && occursin(string(t[end] + Day(3)), err.msg)

        # missing values inside new_data → error
        holed = (ds=t[end] .+ Day.(1:3), promo=[1.0, missing, 0.0])
        @test_throws ArgumentError forecast(fitted, 3; new_data=holed)

        # correct join alignment: rows out of order and with extras still align by time
        future = (ds=[t[end] + Day(3), t[end] + Day(1), t[end] + Day(2), t[end] + Day(9)],
                  promo=[30.0, 10.0, 20.0, 99.0])
        fcast = forecast(fitted, 3; new_data=future)
        @test fcast.y_hat == [10.0, 20.0, 30.0]   # EchoColumn(:promo)

        # exogenous in training must be present
        fc2 = Forecaster(TestModels.LinAR(1.0, 0.0);
                         features=FeatureSet(Lag(1), Exogenous(:absent)), freq=Day(1))
        @test_throws ArgumentError fit(fc2, dfe)
    end

    @testset "new_data without exogenous features warns and is ignored" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)), freq=Day(1))
        fitted = fit(fc, df)
        extra = (ds=t[end] .+ Day.(1:2), promo=zeros(2))
        fcast = @test_logs (:warn, r"ignored") forecast(fitted, 2; new_data=extra)
        @test length(fcast.y_hat) == 2
    end

    @testset "time column validation" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)), freq=Day(1))
        gap = (ds=[t[1:10]; t[12:20]], y=Float64.(1:19))
        err = try fit(fc, gap) catch e; e end
        @test err isa ArgumentError && occursin("gap", err.msg)
        @test occursin(string(t[10]), err.msg)   # first gap after this timestamp
        dup = (ds=[t[1:10]; [t[10]]; t[11:19]], y=Float64.(1:20))
        err = try fit(fc, dup) catch e; e end
        @test err isa ArgumentError && occursin("duplicate", err.msg)
        unsorted = (ds=reverse(t), y=y)
        err = try fit(fc, unsorted) catch e; e end
        @test err isa ArgumentError && occursin("not sorted", err.msg)
        # wrong frequency counts as gaps
        fc_w = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)), freq=Week(1))
        @test_throws ArgumentError fit(fc_w, df)
        # missing time column
        @test_throws ArgumentError fit(fc, (when=t, y=y))
        # missing target values
        ym = Vector{Union{Missing,Float64}}(y)
        ym[7] = missing
        @test_throws ArgumentError fit(fc, (ds=t, y=ym))
    end

    @testset "table genericity: columntable and rowtable agree" begin
        fc = Forecaster(TestModels.LinAR(0.9, 1.0);
                        features=FeatureSet(Lag(1), RollingMean(3), Calendar(:dayofweek)),
                        strategy=Recursive(), freq=Day(1))
        ref = forecast(fit(fc, df), 5)
        rt = Tables.rowtable(df)               # Vector of NamedTuples
        @test rt isa Vector{<:NamedTuple}
        @test forecast(fit(fc, rt), 5).y_hat == ref.y_hat
        dt = Tables.dictcolumntable(df)        # yet another table flavor
        @test forecast(fit(fc, dt), 5).y_hat == ref.y_hat
        @test_throws ArgumentError fit(fc, 42)   # not a table
        # the forecast output is itself a Tables.jl table
        @test Tables.istable(ref)
        @test Tables.columntable(ref) == ref
    end

    @testset "misc" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)), freq=Day(1))
        fitted = fit(fc, df)
        @test_throws ArgumentError forecast(fitted, 0)
        @test_throws ArgumentError forecast(fitted, -2)
        # user data is not mutated by fit
        df2 = deepcopy(df)
        fit(fc, df2)
        @test df2 == df
        # pretty one-liners
        @test occursin("LinAR", sprint(show, fc))
        @test occursin("Recursive()", sprint(show, fc))
        @test occursin("trained on 60 rows", sprint(show, fitted))
        # DateTime time column with hourly frequency
        th = collect(DateTime(2022, 1, 1):Hour(1):DateTime(2022, 1, 3, 11))
        dfh = (ds=th, y=Float64.(1:length(th)))
        fch = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)), freq=Hour(1))
        fh = forecast(fit(fch, dfh), 3)
        @test fh.ds == [th[end] + Hour(s) for s in 1:3]
    end
end
