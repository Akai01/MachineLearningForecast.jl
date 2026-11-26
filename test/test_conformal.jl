@testset "conformal seasonal pools" begin
    C = MachineLearningForecast
    monthly = collect(Date(2015, 1, 1):Month(1):Date(2019, 12, 1))
    noise = randn(StableRNG(1), 60)
    yseas = [50 + 10 * sin(2π * i / 12) + 0.2i + noise[i] for i in 1:60]

    @testset "SeasonalNaive repeats the last season" begin
        t = collect(Date(2020, 1, 1):Month(1):Date(2022, 12, 1))
        y = [10.0 + mod1(i, 12) + i / 100 for i in 1:36]
        fc = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1))
        f = fit(fc, (ds=t, y=y))
        # Past one season the latest same-phase value repeats.
        @test forecast(f, 26).y_hat == [y[36 + j - 12 * cld(j, 12)] for j in 1:26]
        @test forecast(f, 1).y_hat == [y[25]]
        @test fc.model.period == 12

        # One season and one row: the fallback still applies.
        f13 = fit(fc, (ds=t[1:13], y=y[1:13]))
        @test forecast(f13, 14).y_hat == [y[13 + j - 12 * cld(j, 12)] for j in 1:14]

        # Exogenous columns do not change the forecast.
        x = collect(1.0:38.0)
        fx = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12), Exogenous(:x)),
                        freq=Month(1))
        tf = collect(Date(2023, 1, 1):Month(1):Date(2023, 2, 1))
        @test forecast(fit(fx, (ds=t, y=y, x=x[1:36])), 2;
                       future=(ds=tf, x=x[37:38])).y_hat == y[25:26]

        # A row table works when called through MLJ directly.
        rows = [(y_lag_12=1.0,), (y_lag_12=2.5,)]
        @test C.MLJModelInterface.predict(SeasonalNaive(12), nothing, rows) == [1.0, 2.5]

        # Direct is not seasonal naive: Lag(12) is one value.
        fd = fit(Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)),
                            strategy=Direct(3), freq=Month(1)), (ds=t, y=y))
        @test forecast(fd, 3).y_hat == fill(y[25], 3)

        # A ragged panel: each series repeats its own last season.
        ids = [fill("a", 36); fill("b", 30)]
        panel = (unique_id=ids, ds=[t; t[7:36]], y=[y; 2 .* y[7:36]])
        pfc = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1),
                         id=:unique_id)
        out = forecast(fit(pfc, panel), 3)
        @test out.y_hat == [y[25:27]; 2 .* y[25:27]]

        # Backtest folds equal the manual seasonal naive forecast.
        r = backtest(fc, (ds=t, y=y); horizon=6, windows=2)
        @test r.forecasts.y_hat ==
              reduce(vcat, [[y[o + s - 12 * cld(s, 12)] for s in 1:6] for o in (24, 30)])

        # Short history, a gap, a NaN and a missing are refused.
        @test_throws "not enough data" fit(fc, (ds=t[1:12], y=y[1:12]))
        gap = [t[1:10]; t[12:36]]
        @test_throws "has 1 gap" fit(fc, (ds=gap, y=y[1:35]))
        @test_throws "non-finite value NaN" fit(fc, (ds=t, y=[y[1:35]; NaN]))
        @test_throws "contains missing values" fit(
            fc, (ds=t, y=Union{Missing,Float64}[y[1:35]; missing]))
    end

    @testset "SeasonalNaive errors are actionable" begin
        t = collect(Date(2020, 1, 1):Month(1):Date(2022, 12, 1))
        y = [10.0 + mod1(i, 12) for i in 1:36]
        bad = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(1)), freq=Month(1))
        @test_throws "SeasonalNaive(12) predicts the feature column :y_lag_12, which " *
                     "the feature set does not produce. Add Lag(12) to the FeatureSet, " *
                     "e.g. FeatureSet(Lag(12))." forecast(fit(bad, (ds=t, y=y)), 2)
        @test_throws "SeasonalNaive needs a period ≥ 1, got 0. Pass the season length " *
                     "in steps, e.g. SeasonalNaive(12) for monthly data." SeasonalNaive(0)
        @test_throws "SeasonalNaive needs a period ≥ 1, got -3." SeasonalNaive(-3)
        @test SeasonalNaive(1).period == 1
        @test_throws "SeasonalNaive period must be a whole number, not the Bool true. " *
                     "Pass the season length in steps, e.g. SeasonalNaive(12) for " *
                     "monthly data." SeasonalNaive(true)
        @test_throws "SeasonalNaive period must be a whole number, got 12.0. Pass the " *
                     "season length in steps, e.g. SeasonalNaive(12) for monthly " *
                     "data." SeasonalNaive(12.0)
        @test_throws "must be a whole number, got \"12\"." SeasonalNaive("12")
        @test SeasonalNaive(Int32(4)).period === 4
        @test C.MLJModelInterface.predict(SeasonalNaive(2), nothing,
                                          (y_lag_2=[1.0, 2.0],)) == [1.0, 2.0]
        @test C.MLJModelInterface.predict(SeasonalNaive(2), nothing,
                                          (y_lag_2=[1, 2],)) isa Vector{Float64}
        MMI = C.MLJModelInterface
        @test MMI.input_scitype(SeasonalNaive) == MMI.Table(MMI.Continuous)
        @test MMI.target_scitype(SeasonalNaive) == AbstractVector{MMI.Continuous}
    end
end
