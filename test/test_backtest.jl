@testset "metrics" begin
    y = [10.0, 20.0, 30.0]
    ŷ = [12.0, 18.0, 33.0]
    @test mae(y, ŷ) ≈ (2 + 2 + 3) / 3
    @test rmse(y, ŷ) ≈ sqrt((4 + 4 + 9) / 3)
    @test mape(y, ŷ) ≈ (2 / 10 + 2 / 20 + 3 / 30) / 3
    @test smape(y, ŷ) ≈ (2 * 2 / 22 + 2 * 2 / 38 + 2 * 3 / 63) / 3
    @test mae(y, y) == 0.0
    @test smape(y, y) == 0.0

    # mape with zeros warns and returns Inf
    @test (@test_logs (:warn, r"mape is undefined") mape([0.0, 1.0], [1.0, 1.0])) == Inf

    # smape 0/0 terms contribute 0
    @test smape([0.0, 1.0], [0.0, 1.0]) == 0.0

    # mase: hand-computed
    y_train = [1.0, 3.0, 1.0, 3.0, 1.0]   # naive m=1 error: mean(|2,2,2,2|) = 2
    @test mase([10.0, 12.0], [11.0, 11.0]; y_train=y_train) ≈ 1.0 / 2
    # seasonal m=2: |y_t - y_{t-2}| = 0 everywhere → Inf with a warning
    @test (@test_logs (:warn, r"mase is undefined") mase(y, ŷ; y_train=y_train, m=2)) == Inf
    @test_throws ArgumentError mase(y, ŷ; y_train=[1.0], m=1)
    @test_throws "mase needs length(y_train) > m; got length(y_train)=1 with m=1. " *
                 "Pass a longer y_train or a smaller m." mase(y, ŷ; y_train=[1.0], m=1)
    @test_throws ArgumentError mase(y, ŷ; y_train=y_train, m=0)
    @test_throws "mase seasonality m must be ≥ 1, got 0. Pass the season length in " *
                 "steps, e.g. m=7 for weekly seasonality on daily data, or m=1 (the " *
                 "default)." mase(y, ŷ; y_train=y_train, m=0)

    # argument validation
    @test_throws ArgumentError mae([1.0], [1.0, 2.0])
    @test_throws "metric inputs must have equal length, got length(y)=1 and " *
                 "length(ŷ)=2. Pass one forecast per actual, aligned element by " *
                 "element." mae([1.0], [1.0, 2.0])
    @test_throws ArgumentError rmse(Float64[], Float64[])
    @test_throws "metric inputs are empty: got length(y)=0 and length(ŷ)=0. Pass at " *
                 "least one actual and its forecast." rmse(Float64[], Float64[])
    for f in (mae, rmse, mape, smape)
        @test_throws "got length(y)=2 and length(ŷ)=1" f([1.0, 2.0], [1.0])
        @test_throws "metric inputs are empty" f(Float64[], Float64[])
    end
end

@testset "backtest" begin
    n = 100
    t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
    y = Float64.(1:n)
    df = (ds=t, y=y)
    fs = FeatureSet(Lag(1))
    # EchoColumn(:y_lag_1) is the naive forecaster: constant y[origin] over the horizon.
    fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=fs,
                    strategy=Recursive(), freq=Day(1))

    @testset "fold structure and exact values" begin
        res = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(mae, rmse))
        # origins: 80, 90 (100 would need rows 101:105)
        @test res isa BacktestResult
        @test keys(res.folds) == (:origin, :step, :ds, :y, :y_hat)
        @test length(res.folds.step) == 2 * 5
        @test res.folds.step == [1:5; 1:5]
        @test res.folds.origin == [fill(t[80], 5); fill(t[90], 5)]
        @test res.folds.ds == [t[81:85]; t[91:95]]
        @test res.folds.y == [y[81:85]; y[91:95]]
        # naive forecast: y[80] and y[90] carried forward
        @test res.folds.y_hat == [fill(80.0, 5); fill(90.0, 5)]
        # per-fold metrics: |y - origin| averaged: mean(1,2,3,4,5) = 3
        m = res.metrics
        perfold = m.fold .> 0
        @test count(perfold) == 4   # 2 folds × 2 metrics
        @test all(m.value[perfold .& (m.metric .== :mae)] .≈ 3.0)
        # overall summary: fold 0, mean over folds, missing origin
        overall = m.fold .== 0
        @test count(overall) == 2
        @test all(ismissing, m.origin[overall])
        @test only(m.value[overall .& (m.metric .== :mae)]) ≈ 3.0
        @test only(m.value[overall .& (m.metric .== :rmse)]) ≈ sqrt(mean([1, 4, 9, 16, 25]))
        # both tables are Tables.jl-compatible
        @test Tables.istable(res.folds) && Tables.istable(res.metrics)
        # show is compact and informative
        s = sprint(show, MIME"text/plain"(), res)
        @test occursin("2 folds", s) && occursin("mae", s)
    end

    @testset "exogenous columns are sliced from held-out data" begin
        dfe = (ds=t, y=y, promo=Float64.(1:n) .* 10)
        fce = Forecaster(TestModels.EchoColumn(:promo);
                         features=FeatureSet(Lag(1), Exogenous(:promo)),
                         strategy=Recursive(), freq=Day(1))
        res = backtest(fce, dfe; horizon=3, initial=90, step=100, metrics=(mae,))
        # forecast echoes future promo values
        @test res.folds.y_hat == dfe.promo[91:93]
    end

    @testset "argument validation" begin
        @test_throws ArgumentError backtest(fc, df; horizon=0, initial=50)
        @test_throws "backtest horizon must be ≥ 1, got 0. Pass the number of steps each " *
                     "fold forecasts, e.g. horizon=28." backtest(fc, df; horizon=0,
                                                                  initial=50)
        @test_throws ArgumentError backtest(fc, df; horizon=5, initial=0)
        @test_throws "backtest initial must be ≥ 1, got 0. Pass the size of the first " *
                     "training window, e.g. initial=730." backtest(fc, df; horizon=5,
                                                                    initial=0)
        @test_throws ArgumentError backtest(fc, df; horizon=5, initial=50, step=0)
        @test_throws "backtest step must be ≥ 1, got 0. Pass how many steps each fold " *
                     "moves the origin, e.g. step=5, which equals horizon (the " *
                     "default)." backtest(fc, df; horizon=5, initial=50, step=0)
        # no complete folds
        @test_throws ArgumentError backtest(fc, df; horizon=30, initial=90)
        @test_throws "no complete backtest folds: data has 100 rows, but the first fold " *
                     "needs initial + horizon = 120. Provide more data" backtest(
            fc, df; horizon=30, initial=90)
        # initial must exceed minhistory
        fc_deep = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(30)), freq=Day(1))
        @test_throws ArgumentError backtest(fc_deep, df; horizon=5, initial=30)
        @test_throws "backtest initial=30 must exceed the feature set's minimum history " *
                     "(30 rows) so the first training window has at least one usable " *
                     "row. Pass initial=31 or more, or reduce lags/windows." backtest(
            fc_deep, df; horizon=5, initial=30)
        # Direct max_horizon < backtest horizon
        fc_d = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Direct(3), freq=Day(1))
        @test_throws ArgumentError backtest(fc_d, df; horizon=5, initial=80)
        @test_throws "backtest horizon=5 exceeds the Direct strategy's max_horizon=3. " *
                     "Use Direct(5) or reduce horizon." backtest(fc_d, df; horizon=5,
                                                                initial=80)
        # ... but works when compatible
        res = backtest(fc_d, df; horizon=3, initial=90, metrics=(mae,))
        @test maximum(res.metrics.fold) == 3   # origins 90, 93, 96
    end

    @testset "metrics must be a tuple or vector of functions" begin
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fcm = Forecaster(m; features=fs, freq=Day(1))
            for bad in (mae, (:mae,), ("mae", "rmse"), Dict(:mae => mae), [mae, :rmse])
                @test_throws ArgumentError backtest(fcm, df; horizon=5, initial=80,
                                                    metrics=bad)
                @test_throws "got $(repr(bad)). Pass e.g. metrics=(mae, rmse)" backtest(
                    fcm, df; horizon=5, initial=80, metrics=bad)
            end
        end
        # a vector, named tuple and closure still work
        r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=[mae, rmse])
        @test r.metrics.metric == [:mae, :rmse, :mae, :rmse, :mae, :rmse]
        r2 = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(a=mae,))
        @test r2.metrics.value == r.metrics.value[r.metrics.metric .== :mae]
        r3 = backtest(fc, df; horizon=5, initial=80, step=10,
                      metrics=((a, b) -> mae(a, b),))
        @test r3.metrics.value == r2.metrics.value
    end

    @testset "edge cases: horizon 1 and bad data" begin
        r1 = backtest(fc, df; horizon=1, initial=95, metrics=(mae,))
        @test r1.folds.step == ones(Int, 5)
        @test r1.folds.y_hat == y[95:99] && r1.folds.y == y[96:100]
        @test all(==(1.0), r1.metrics.value)
        @test_throws TypeError backtest(fc, df; horizon=1.5, initial=95)
        ym = Vector{Union{Missing,Float64}}(y); ym[55] = missing
        yn = copy(y); yn[55] = NaN
        gapped = (ds=[t[1:30]; t[32:end]], y=y[1:99])
        bad = [(ds=Date[], y=Float64[]) => "time column :ds is empty",
               (ds=t[1:3], y=y[1:3]) => "no complete backtest folds: data has 3 rows",
               (ds=t, y=ym) => "contains missing values (first at row 55)",
               (ds=t, y=yn) => "the non-finite value NaN at row 55",
               gapped => "time column :ds has 1 gap for freq=1 day",
               (ds=t, y=string.(y)) => "target column :y has element type String",
               (ds=string.(t), y=y) => "has element type String, which does not support",
               42 => "expected a Tables.jl-compatible table"]
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fcm = Forecaster(m; features=FeatureSet(Lag(1), Lag(2)), freq=Day(1))
            r = backtest(fcm, df; horizon=1, initial=95)
            want = [only(forecast(fit(fcm, (ds=t[1:o], y=y[1:o])), 1).y_hat)
                    for o in 95:99]
            @test r.folds.y_hat == want
            for (data, msg) in bad
                @test_throws ArgumentError backtest(fcm, data; horizon=1, initial=5)
                @test_throws msg backtest(fcm, data; horizon=1, initial=5)
            end
        end
    end
end
