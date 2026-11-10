@testset "metrics" begin
    y = [10.0, 20.0, 30.0]
    ŷ = [12.0, 18.0, 33.0]
    @test mae(y, ŷ) ≈ (2 + 2 + 3) / 3
    @test rmse(y, ŷ) ≈ sqrt((4 + 4 + 9) / 3)
    @test mape(y, ŷ) ≈ (2 / 10 + 2 / 20 + 3 / 30) / 3
    @test smape(y, ŷ) ≈ (2 * 2 / 22 + 2 * 2 / 38 + 2 * 3 / 63) / 3
    @test mae(y, y) == 0.0
    @test smape(y, y) == 0.0

    @test (@test_logs (:warn, r"mape is undefined") mape([0.0, 1.0], [1.0, 1.0])) == Inf

    # smape 0/0 terms contribute 0
    @test smape([0.0, 1.0], [0.0, 1.0]) == 0.0

    y_train = [1.0, 3.0, 1.0, 3.0, 1.0]   # naive m=1 error: mean(|2,2,2,2|) = 2
    @test mase([10.0, 12.0], [11.0, 11.0]; y_train=y_train) ≈ 1.0 / 2
    # seasonal m=2: y_t == y_{t-2} everywhere, so Inf
    @test (@test_logs (:warn, r"mase is undefined") mase(y, ŷ; y_train=y_train, m=2)) == Inf
    @test_throws ArgumentError mase(y, ŷ; y_train=[1.0], m=1)
    @test_throws "mase needs length(y_train) > m; got length(y_train)=1 with m=1. " *
                 "Pass a longer y_train or a smaller m." mase(y, ŷ; y_train=[1.0], m=1)
    @test_throws ArgumentError mase(y, ŷ; y_train=y_train, m=0)
    @test_throws "mase seasonality m must be ≥ 1, got 0. Pass the season length in " *
                 "steps, e.g. m=7 for weekly seasonality on daily data, or m=1 (the " *
                 "default)." mase(y, ŷ; y_train=y_train, m=0)

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
