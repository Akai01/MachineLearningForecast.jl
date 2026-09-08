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
    @test_throws ArgumentError mase(y, ŷ; y_train=y_train, m=0)

    # argument validation
    @test_throws ArgumentError mae([1.0], [1.0, 2.0])
    @test_throws ArgumentError rmse(Float64[], Float64[])
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
        @test_throws ArgumentError backtest(fc, df; horizon=5, initial=0)
        @test_throws ArgumentError backtest(fc, df; horizon=5, initial=50, step=0)
        # no complete folds
        @test_throws ArgumentError backtest(fc, df; horizon=30, initial=90)
        # initial must exceed minhistory
        fc_deep = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(30)), freq=Day(1))
        @test_throws ArgumentError backtest(fc_deep, df; horizon=5, initial=30)
        # Direct max_horizon < backtest horizon
        fc_d = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Direct(3), freq=Day(1))
        @test_throws ArgumentError backtest(fc_d, df; horizon=5, initial=80)
        # ... but works when compatible
        res = backtest(fc_d, df; horizon=3, initial=90, metrics=(mae,))
        @test maximum(res.metrics.fold) == 3   # origins 90, 93, 96
    end
end
