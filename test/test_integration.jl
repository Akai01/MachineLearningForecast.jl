# End-to-end integration with real learners (EvoTrees gradient boosting and a
# DecisionTree random forest) on a synthetic seasonal series: the model must
# beat a seasonal-naive baseline on backtest sMAPE. The series has weekly and
# annual seasonality plus a mild trend — the weekly part is what seasonal-naive
# nails, the annual drift is what it structurally cannot track but Fourier
# features can.
@testset "integration" begin
    rng = StableRNG(2024)
    n = 730
    ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 1) + Day(n - 1))
    y = 10 .+ 3 .* sin.(2π .* (1:n) ./ 7) .+ 4 .* sin.(2π .* (1:n) ./ 365.25) .+
        0.001 .* (1:n) .+ 0.25 .* randn(rng, n)
    df = (ds=ts, y=y)

    horizon, initial, step = 14, 600, 56   # folds at origins 600, 656, 712
    features = FeatureSet(Lag(7), Lag(14), RollingMean(7), Calendar(:dayofweek),
                          Fourier(7, 2), Fourier(365.25, 2))

    # seasonal-naive baseline over the same folds: ŷ_{o+s} = y[o + s - 7⌈s/7⌉]
    snaive(o, h, m) = [y[o + s - m * cld(s, m)] for s in 1:h]
    origins = initial:step:(n - horizon)
    snaive_smape = mean(smape(y[(o+1):(o+horizon)], snaive(o, horizon, 7)) for o in origins)

    overall_smape(res) = only(res.metrics.value[(res.metrics.fold .== 0) .&
                                                (res.metrics.metric .== :smape)])

    @testset "EvoTrees beats seasonal naive" begin
        model = EvoTreeRegressor(nrounds=200, eta=0.05, max_depth=4, seed=123)
        fc = Forecaster(model; features=features, strategy=Recursive(), freq=Day(1))
        res = backtest(fc, df; horizon=horizon, initial=initial, step=step,
                       metrics=(smape, mae))
        @test overall_smape(res) < snaive_smape
    end

    @testset "DecisionTree random forest beats seasonal naive" begin
        model = RandomForestRegressor(n_trees=30, rng=StableRNG(1))
        fc = Forecaster(model; features=features, strategy=Recursive(), freq=Day(1))
        res = backtest(fc, df; horizon=horizon, initial=initial, step=step,
                       metrics=(smape,))
        @test overall_smape(res) < snaive_smape
    end

    @testset "Direct strategy end-to-end with a real learner" begin
        model = EvoTreeRegressor(nrounds=30, eta=0.2, seed=123)
        fc = Forecaster(model; features=features, strategy=Direct(14), freq=Day(1))
        fitted = fit(fc, df)
        @test length(fitted.machines) == 14
        fcast = forecast(fitted, 14)
        @test all(isfinite, fcast.y_hat)
        # sanity: forecasts stay in a plausible band around the signal level
        @test all(abs.(fcast.y_hat .- mean(y)) .< 8)
    end

    @testset "exogenous end-to-end (the README example shape)" begin
        # y depends strongly on a future-known covariate
        promo = Float64.(rand(rng, Bool, n))
        y2 = 10 .+ 3 .* sin.(2π .* (1:n) ./ 7) .+ 5 .* promo .+ 0.3 .* randn(rng, n)
        df2 = (ds=ts, y=y2, promo=promo)
        model = EvoTreeRegressor(nrounds=50, eta=0.2, seed=123)
        fc = Forecaster(model;
                        features=FeatureSet(Lag(1), Lag(7), Calendar(:dayofweek),
                                            Exogenous(:promo)),
                        strategy=Recursive(), freq=Day(1))
        fitted = fit(fc, df2)
        future = (ds=ts[end] .+ Day.(1:14),
                  promo=Float64.([ones(7); zeros(7)]))
        fcast = forecast(fitted, 14; new_data=future)
        # promo weeks must forecast visibly higher than non-promo weeks
        @test mean(fcast.y_hat[1:7]) - mean(fcast.y_hat[8:14]) > 2.5
    end
end
