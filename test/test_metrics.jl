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
    for bad in (true, false)
        @test_throws "mase seasonality m must be a whole number, not the Bool $bad. " *
                     "Pass the season length in steps, e.g. m=7 for weekly " *
                     "seasonality on daily data." mase(y, ŷ; y_train=y_train, m=bad)
    end
    @test_throws "mase seasonality m must be ≥ 1, got 0. Pass the season length in " *
                 "steps, e.g. m=7 for weekly seasonality on daily data, or m=1 (the " *
                 "default)." mase(y, ŷ; y_train=y_train, m=0)

    @testset "rmsse" begin
        yr = [10.0, 14.0]
        ŷr = [11.0, 11.0]
        # Naive m=1 steps 1, 2, 3: mean square 14/3; rmse² = 5.
        @test rmsse(yr, ŷr; y_train=[1.0, 2.0, 4.0, 7.0]) ≈ sqrt(15 / 14)
        # Lag-2 steps 3, 5, 7: mean square 83/3.
        @test rmsse(yr, ŷr; y_train=[1.0, 2.0, 4.0, 7.0, 11.0], m=2) ≈ sqrt(15 / 83)
        tr = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
        sc = mean(abs2(tr[i] - tr[i - 1]) for i in 2:6)
        @test rmsse(yr, ŷr; y_train=tr) ≈ rmse(yr, ŷr) / sqrt(sc)
        @test rmsse(yr, ŷr; y_train=tr) ≈ rmsse(yr, ŷr; y_train=tr, m=1)
        # Scale-free.
        @test rmsse(1000 .* yr, 1000 .* ŷr; y_train=1000 .* tr) ≈
              rmsse(yr, ŷr; y_train=tr)
        @test rmsse(yr, yr; y_train=tr) == 0.0
        # Ranks methods within a series as rmse does.
        @test (rmsse(yr, ŷr; y_train=tr) < rmsse(yr, [12.0, 11.0]; y_train=tr)) ==
              (rmse(yr, ŷr) < rmse(yr, [12.0, 11.0]))
        @test MachineLearningForecast.needs_ytrain(rmsse)
        @test (@test_logs (:warn, r"rmsse is undefined for a constant") rmsse(
            yr, ŷr; y_train=[3.0, 3.0, 3.0])) == Inf
        @test (@test_logs (:warn, r"rmsse is undefined") rmsse(
            yr, ŷr; y_train=[1.0, 2.0, 1.0, 2.0, 1.0], m=2)) == Inf

        @test_throws ArgumentError rmsse(yr, ŷr; y_train=[1.0], m=1)
        @test_throws "rmsse needs length(y_train) > m; got length(y_train)=2 with m=2. " *
                     "Pass a longer y_train or a smaller m." rmsse(
            yr, ŷr; y_train=[1.0, 2.0], m=2)
        @test_throws "rmsse seasonality m must be ≥ 1, got 0. Pass the season length " *
                     "in steps, e.g. m=7 for weekly seasonality on daily data, or m=1 " *
                     "(the default)." rmsse(yr, ŷr; y_train=tr, m=0)
        for bad in (true, false)
            @test_throws "rmsse seasonality m must be a whole number, not the Bool " *
                         "$bad. Pass the season length in steps, e.g. m=7 for weekly " *
                         "seasonality on daily data." rmsse(yr, ŷr; y_train=tr, m=bad)
        end
        @test_throws "metric inputs must have equal length, got length(y)=1 and " *
                     "length(ŷ)=2." rmsse([1.0], [1.0, 2.0]; y_train=tr)
        @test_throws "metric inputs are empty" rmsse(Float64[], Float64[]; y_train=tr)
    end

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
