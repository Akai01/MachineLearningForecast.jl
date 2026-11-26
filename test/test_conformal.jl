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

    @testset "signed seasonal-difference pool" begin
        y = [t + 10.0 * (t % 4 == 0) for t in 1:20]
        @test @inferred(C._seasonal_differences(y, 4, 0.5)) == fill(4.0, 10)
        @test length(C._seasonal_differences(y, 4, 1.0)) == 16
        @test isempty(C._seasonal_differences(y[1:4], 4, 0.5))
        @test C._seasonal_differences([1.0, 4.0, 2.0], 1, 1.0) == [3.0, -2.0]
        # floor(0.05 * 13) = 0: no calibration value at all.
        @test isempty(C._seasonal_differences(y[1:13], 4, 0.05))
        @test isempty(C._seasonal_differences(Float64[], 4, 0.5))
        # R comes from the latest ρT steps, not the first.
        @test C._seasonal_differences(Float64.(1:10) .^ 2, 1, 0.3) == [15.0, 17.0, 19.0]
    end

    @testset "seasonal pool and recency weights" begin
        y = Float64.(1:10)
        vals, cw = @inferred C._seasonal_pool(y, 1, 3, 0.0)
        @test vals == [2.0, 5.0, 8.0] && cw == [1.0, 2.0, 3.0]
        vals, cw = C._seasonal_pool(y, 1, 3, 0.5)
        # The newest value has weight 1.
        @test diff([0; cw]) ≈ exp.(-0.5 .* [6, 3, 0])
        # The pool sits at the phase of T + h.
        @test first(C._seasonal_pool(y, 2, 3, 0.0)) == [3.0, 6.0, 9.0]
        @test isempty(first(C._seasonal_pool(Float64.(1:5), 1, 12, 0.0)))
        # No seasonality: every value is in the pool.
        @test first(C._seasonal_pool(y, 4, 1, 0.0)) == y
        @test @inferred(C._draw(StableRNG(1), [1.0, 2.0, 3.0])) in 1:3
        # A large rate must not underflow every weight to zero.
        yl = Float64.(1:400)
        _, cwl = C._seasonal_pool(yl, 1, 12, 100.0)
        @test cwl[end] > 0 && issorted(cwl)
        Sl = C._pool_samples(StableRNG(9), yl, [-1e6], [0.0],
                             C.PoolIntervals(12; weight=C.FixedWeight(1.0),
                                             recency=100.0, samples=200))
        @test all(==(389.0), Sl)
    end

    @testset "mixture weights follow Algorithm 1" begin
        @test @inferred(C._mixture_weight(C.AdaptiveWeight(), 1, 50)) == 0.0
        @test C._mixture_weight(C.AdaptiveWeight(), 12, 2) == 0.3
        @test C._mixture_weight(C.AdaptiveWeight(), 12, 3) == 0.5
        @test C._mixture_weight(C.FixedWeight(), 1, 0) == 0.5
        @test C._mixture_weight(C.FixedWeight(0.2), 12, 9) == 0.2
        @test C.FixedWeight(1).w === 1.0
        @test_throws("FixedWeight needs a seasonal-pool weight in [0, 1], got 1.2. " *
                     "Pass e.g. FixedWeight(0.5), the paper's CSP-Fixed.",
                     C.FixedWeight(1.2))
        @test_throws "got NaN" C.FixedWeight(NaN)
        @test_throws("FixedWeight weight must be a number in [0, 1], not the Bool " *
                     "true. Pass e.g. FixedWeight(0.5), the paper's CSP-Fixed.",
                     C.FixedWeight(true))
        @test_throws("FixedWeight needs a seasonal-pool weight in [0, 1], got \"a\".",
                     C.FixedWeight("a"))
        @test sprint(show, C.AdaptiveWeight()) == "AdaptiveWeight()"
    end

    @testset "the sampler mixes the two pools in proportion w_h" begin
        y = 100 .+ Float64.(1:48)
        cp = C.PoolIntervals(12; samples=20_000, rng=StableRNG(1))
        S = C._pool_samples(copy(cp.rng), y, fill(-1e6, 3), [0.0], cp)
        @test size(S) == (3, 20_000)
        # Four same-season values, so the Adaptive weight is 0.5.
        @test all(abs(mean(S[j, :] .> 0) - 0.5) < 0.02 for j in 1:3)
        @test all(v -> v == -1e6 || v in y, S)

        # Two same-season values (T = 30): weight 0.3.
        S3 = C._pool_samples(StableRNG(5), y[1:30], fill(-1e6, 1), [0.0], cp)
        @test abs(mean(S3[1, :] .> 0) - 0.3) < 0.02
        # Exactly three same-season values (T = 36): back to 0.5.
        S4 = C._pool_samples(StableRNG(6), y[1:36], fill(-1e6, 1), [0.0], cp)
        @test abs(mean(S4[1, :] .> 0) - 0.5) < 0.02

        # m = 1: the Adaptive weight is 0.
        S1 = C._pool_samples(StableRNG(2), y, fill(-1e6, 2), [0.0],
                             C.PoolIntervals(1; samples=500))
        @test all(==(-1e6), S1)

        # Weight 0: centre plus a residual, uniformly.
        R = [1.0, 2.0, 3.0]
        S0 = C._pool_samples(StableRNG(3), y, [10.0, 20.0], R,
                             C.PoolIntervals(12; weight=C.FixedWeight(0.0),
                                             samples=30_000))
        @test all(in((11.0, 12.0, 13.0)), S0[1, :])
        @test all(in((21.0, 22.0, 23.0)), S0[2, :])
        @test all(abs(mean(S0[1, :] .== 10.0 + r) - 1 / 3) < 0.02 for r in R)

        # Weight 1: seasonal values only, recent ones more often.
        S2 = C._pool_samples(StableRNG(4), y, [-1e6], [0.0],
                             C.PoolIntervals(12; weight=C.FixedWeight(1.0), recency=0.05,
                                             samples=40_000))
        _, cw = C._seasonal_pool(y, 1, 12, 0.05)
        p = diff([0; cw]) ./ cw[end]
        @test p[end] > p[1]
        @test all(abs(mean(S2[1, :] .== v) - p[k]) < 0.015
                  for (k, v) in enumerate(y[1:12:48]))
    end

    @testset "sampling is reproducible from a copied rng" begin
        cp = C.PoolIntervals(12; rng=StableRNG(7))
        yy = 10 .+ sin.(2π .* (1:60) ./ 12)
        R = C._seasonal_differences(yy, 12, 0.5)
        centre = [yy[60 + j - 12] for j in 1:6]
        a = C._pool_samples(copy(cp.rng), yy, centre, R, cp)
        @test a == C._pool_samples(copy(cp.rng), yy, centre, R, cp)
        @test a != C._pool_samples(StableRNG(8), yy, centre, R, cp)
        @test @inferred(C._pool_samples(copy(cp.rng), yy, centre, R, cp)) == a
        # The default Xoshiro rng is inferred too.
        cx = C.PoolIntervals(12)
        Sx = @inferred C._pool_samples(copy(cx.rng), yy, centre, R, cx)
        @test Sx isa Matrix{Float64}
        # No horizons: an empty matrix, not an error.
        @test size(C._pool_samples(copy(cp.rng), yy, Float64[], R, cp)) == (0, 100)
    end

    @testset "edge cases of the pools" begin
        # An empty seasonal pool falls back to the residual pool.
        S = C._pool_samples(StableRNG(1), Float64.(1:5), [100.0], [1.0, 2.0],
                            C.PoolIntervals(12; weight=C.FixedWeight(1.0), samples=50))
        @test all(in((101.0, 102.0)), S)
        # An empty residual pool is only an error when it is needed.
        short = Float64.(1:12)
        @test size(C._pool_samples(StableRNG(1), short, fill(1.0, 2), Float64[],
                                   C.PoolIntervals(12; weight=C.FixedWeight(1.0),
                                                   samples=5))) == (2, 5)
        @test_throws("the residual pool for step 1 is empty: the last calibration=0.5 " *
                     "of the 12 training values hold no seasonal difference y[t] - " *
                     "y[t - 12]. Raise calibration or provide a longer series.",
                     C._pool_samples(StableRNG(1), short, fill(1.0, 2), Float64[],
                                     C.PoolIntervals(12)))
    end

    @testset "invalid sampler settings" begin
        @test_throws("PoolIntervals needs a seasonal period ≥ 1 in steps of freq, got " *
                     "0. Pass e.g. 12 for monthly data, or 1 for no seasonality.",
                     C.PoolIntervals(0))
        @test_throws("PoolIntervals period must be a whole number, not the Bool true.",
                     C.PoolIntervals(true))
        @test_throws("PoolIntervals period must be a whole number, got 12.0. Pass the " *
                     "season length in steps, e.g. 12 for monthly data, or 1 for no " *
                     "seasonality.", C.PoolIntervals(12.0))
        @test_throws("PoolIntervals period must be a whole number, got \"12\".",
                     C.PoolIntervals("12"))
        @test C.PoolIntervals(Int32(4)).period === 4
        @test_throws("calibration must be in (0, 1], the share of each series' history " *
                     "whose seasonal differences form the residual pool, got 0. Pass " *
                     "e.g. calibration=0.5.", C.PoolIntervals(12; calibration=0))
        @test_throws "calibration must be in (0, 1]" C.PoolIntervals(12; calibration=1.5)
        @test_throws "calibration must be in (0, 1]" C.PoolIntervals(12; calibration=NaN)
        @test_throws "calibration must be in (0, 1]" C.PoolIntervals(12; calibration="a")
        @test_throws("calibration must be a number in (0, 1], not the Bool true. Pass " *
                     "e.g. calibration=0.5.", C.PoolIntervals(12; calibration=true))
        @test_throws("recency must be a finite rate ≥ 0 (the decay per step of the " *
                     "seasonal-pool weights), got -1. Pass e.g. recency=0.01, or 0 to " *
                     "weigh the pool equally.", C.PoolIntervals(12; recency=-1))
        @test_throws "recency must be a finite rate" C.PoolIntervals(12; recency=Inf)
        @test_throws "recency must be a finite rate" C.PoolIntervals(12; recency="a")
        @test_throws("recency must be a number ≥ 0, not the Bool false. Pass e.g. " *
                     "recency=0.01, or 0 to weigh the pool equally.",
                     C.PoolIntervals(12; recency=false))
        @test C.PoolIntervals(12; recency=0).recency === 0.0
        @test_throws("samples must be a whole number ≥ 1, got 0. Pass e.g. samples=100.",
                     C.PoolIntervals(12; samples=0))
        @test_throws("samples must be a whole number, not the Bool true. Pass the " *
                     "number of draws per horizon, e.g. samples=100.",
                     C.PoolIntervals(12; samples=true))
        @test_throws("samples must be a whole number, got 2.5.",
                     C.PoolIntervals(12; samples=2.5))
        @test_throws("weight must be an AdaptiveWeight() or a FixedWeight(w), got " *
                     "\"half\". Pass e.g. weight=FixedWeight(0.5).",
                     C.PoolIntervals(12; weight="half"))
        @test sprint(show, C.PoolIntervals(12)) == "PoolIntervals(12, AdaptiveWeight())"
        @test sprint(show, C.PoolIntervals(12; weight=C.FixedWeight(0.3))) ==
              "PoolIntervals(12, FixedWeight(0.3))"
    end

    @testset "Monte Carlo: coverage on an iid seasonal series" begin
        rng = StableRNG(11)
        hits = map(1:200) do rep
            y = [10 + 3 * sin(2π * t / 12) + randn(rng) for t in 1:132]
            hist, fut = y[1:120], y[121:132]
            cp = C.PoolIntervals(12; rng=StableRNG(rep))
            centre = [hist[120 + j - 12] for j in 1:12]
            S = C._pool_samples(copy(cp.rng), hist, centre,
                                C._seasonal_differences(hist, 12, 0.5), cp)
            Coverage(95)(fut, S)
        end
        # No guarantee; this only catches a broken sampler.
        @test mean(hits) > 0.9
    end

    csp(; kw...) = ConformalSeasonalPool(12; freq=Month(1), rng=StableRNG(9), kw...)

    @testset "ConformalSeasonalPool builds the paired forecaster" begin
        fc = csp()
        @test fc isa Forecaster && fc.model isa SeasonalNaive && fc.model.period == 12
        @test fc.strategy isa Recursive && fc.transform isa NoTransform
        @test C.outputnames(fc.features) == [:y_lag_12]
        @test fc.intervals isa C.PoolIntervals && fc.intervals.period == 12
        @test fc.intervals.weight isa C.AdaptiveWeight && fc.intervals.samples == 100
        @test !C.ispanel(fc) && C.ispanel(csp(id=:unique_id))
        @test ConformalSeasonalPool(12; freq=Month(1), target=:v, time=:t).time === :t
        tuned = ConformalSeasonalPool(12; freq=Month(1), weight=C.FixedWeight(0.5),
                                      calibration=0.4, recency=0.1, samples=7)
        @test tuned.intervals.weight isa C.FixedWeight && tuned.intervals.weight.w == 0.5
        @test tuned.intervals.samples == 7
        @test tuned.intervals.calibration == 0.4 && tuned.intervals.recency == 0.1
        @test occursin(", intervals=PoolIntervals(12, AdaptiveWeight())", sprint(show, fc))
        plain = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1))
        @test plain.intervals === nothing && !occursin("intervals", sprint(show, plain))
        # Training-free: fit keeps the series, trains nothing.
        f = fit(fc, (ds=monthly, y=yseas))
        @test only(f.models) isa C.TrainedModel && f.n_train == 60
        @test_throws UndefKeywordError ConformalSeasonalPool(12)
        @test_throws "calibration must be in (0, 1]" csp(calibration=2)
        # The period is checked under this function's own name.
        pfix = "Pass the season length in steps, e.g. 12 for monthly data, or 1 for " *
               "no seasonality."
        @test_throws("ConformalSeasonalPool needs a seasonal period ≥ 1 in steps of " *
                     "freq, got 0. $pfix", ConformalSeasonalPool(0; freq=Month(1)))
        @test_throws("ConformalSeasonalPool period must be a whole number, got 12.0. " *
                     pfix, ConformalSeasonalPool(12.0; freq=Month(1)))
        @test_throws("ConformalSeasonalPool period must be a whole number, not the " *
                     "Bool true. $pfix", ConformalSeasonalPool(true; freq=Month(1)))
        @test_throws "weight must be an AdaptiveWeight() or a FixedWeight(w)" csp(
            weight="half")
        rmsg = "rng must be a random number generator, got 1. Pass e.g. " *
               "rng=StableRNG(42) or rng=Random.Xoshiro(42)."
        @test_throws rmsg ConformalSeasonalPool(12; freq=Month(1), rng=1)
        @test_throws rmsg C.PoolIntervals(12; rng=1)
    end

    @testset "a hand-built Forecaster is checked against the pairing" begin
        iv = C.PoolIntervals(12)
        hand(; kw...) = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)),
                                   freq=Month(1), intervals=iv, kw...)
        @test hand().intervals === iv
        @test hand(id=:u).id === :u
        fix = "Use ConformalSeasonalPool(12; freq=...) to build the paired forecaster, " *
              "or Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), " *
              "strategy=Recursive(), freq=..., intervals=PoolIntervals(12))."
        # The pairing refuses every other model.
        for model in (EvoTreeRegressor(nrounds=5), DecisionTreeRegressor(max_depth=2),
                      TestModels.LinAR(1.0, 0.0))
            @test_throws "intervals=PoolIntervals(12) is the paper's Conformal Seasonal " *
                         "Pools, defined around the seasonal naive forecast, but the " *
                         "model is $(nameof(typeof(model))). Intervals around other " *
                         "models are not available. $fix" Forecaster(
                model; features=FeatureSet(Lag(12)), freq=Month(1), intervals=iv)
        end
        @test_throws "intervals=PoolIntervals(12) needs SeasonalNaive(12), got " *
                     "SeasonalNaive(7): the pools are built around the seasonal naive " *
                     "forecast of that period. $fix" Forecaster(
            SeasonalNaive(7); features=FeatureSet(Lag(7)), freq=Month(1), intervals=iv)
        for (feats, got) in ((FeatureSet(Lag(1)), "[:y_lag_1]"),
                             (FeatureSet(Lag(12), Lag(1)), "[:y_lag_12, :y_lag_1]"),
                             (FeatureSet(Lag(12), Exogenous(:x)), "[:y_lag_12, :x]"))
            @test_throws "intervals=PoolIntervals(12) needs the feature set to be " *
                         "exactly FeatureSet(Lag(12)), which feeds SeasonalNaive(12), " *
                         "but the feature set produces $got. $fix" Forecaster(
                SeasonalNaive(12); features=feats, freq=Month(1), intervals=iv)
        end
        @test_throws "intervals=PoolIntervals(12) needs strategy=Recursive(), got " *
                     "$(Direct(3)): under Direct, Lag(12) is not in phase with the " *
                     "target at every step, so the centre is not the seasonal naive " *
                     "forecast. $fix" hand(strategy=Direct(3))
        @test_throws "intervals=PoolIntervals(12) needs no target transform, got " *
                     "$(LastScale(12)): the paper's pools and centre are on the " *
                     "original scale. $fix" hand(transform=LastScale(12))
        # reconstruct runs the constructor, so it is checked.
        @test_throws "needs SeasonalNaive(12)" C.reconstruct(csp(); model=SeasonalNaive(7))
        # Anything but nothing or a PoolIntervals is refused.
        for bad in (:csp, 0.95, csp())
            @test_throws "intervals must be nothing or a PoolIntervals(m), got " *
                         "$(repr(bad)). Pass intervals=PoolIntervals(m) here; " *
                         "ConformalSeasonalPool(m; freq) builds a whole Forecaster." hand(
                intervals=bad)
        end
    end

    @testset "intervals are not tunable and reconstruct keeps them" begin
        fc = csp()
        @test C.reconstruct(fc).intervals === fc.intervals
        @test C.reconstruct(fc; model=SeasonalNaive(12)).intervals === fc.intervals
        @test !(:intervals in C.TUNABLE_FIELDS) && :intervals in fieldnames(Forecaster)
        msg(lead, noun) = "$lead the $noun :intervals, which cannot be tuned: tune " *
                          "ranks candidates by point-forecast metrics and interval " *
                          "settings leave y_hat unchanged. Compare them with " *
                          "backtest(...; metrics=(crps, Coverage(95)))."
        @test_throws msg("cannot reconstruct Forecaster with", "field") C.reconstruct(
            fc; intervals=nothing)
        @test_throws msg("grid has", "key") tune(
            fc, (ds=monthly, y=yseas); grid=(intervals=[fc.intervals],), horizon=6,
            initial=36)
        # A point metric tunes a CSP spec and keeps its intervals.
        r = tune(fc, (ds=monthly, y=yseas); grid=(model=[SeasonalNaive(12)],),
                 horizon=6, windows=2, metric=smape)
        @test r.best.intervals === fc.intervals && isfinite(only(r.table.mean_score))
        # A grid that breaks the pairing fails every candidate, naming the model.
        @test_throws ["all 1 tuning candidates failed to evaluate",
                      "candidate 1 (model=EvoTreeRegressor(",
                      "ArgumentError: intervals=PoolIntervals(12) is the paper's " *
                      "Conformal Seasonal Pools, defined around the seasonal naive " *
                      "forecast, but the model is EvoTreeRegressor. Intervals around " *
                      "other models are not available."] tune(
            fc, (ds=monthly, y=yseas); grid=(model=[EvoTreeRegressor(nrounds=5)],),
            horizon=6, windows=2, metric=smape)
    end

    @testset "the docs state that there is no coverage guarantee" begin
        for doc in (string(@doc(C.PoolIntervals)), string(@doc(ConformalSeasonalPool)))
            @test occursin("no coverage guarantee", doc)
            @test !occursin(r"\bvalid\b|\bguaranteed\b", doc)
        end
    end

    q6(x, p) = quantile(x, p; alpha=0, beta=0)

    @testset "forecast gives intervals from the paper's sampler" begin
        spec = csp()
        f = fit(spec, (ds=monthly, y=yseas))
        out = forecast(f, 18; levels=(80, 95))
        @test keys(out) == (:ds, :y_hat, :lo_80, :hi_80, :lo_95, :hi_95)
        @test out.y_hat == forecast(f, 18).y_hat
        @test out.y_hat == [yseas[60 + j - 12 * cld(j, 12)] for j in 1:18]
        @test all(out.lo_95 .≤ out.lo_80 .≤ out.hi_80 .≤ out.hi_95)
        S = C._pool_samples(copy(spec.intervals.rng), yseas, out.y_hat,
                            C._seasonal_differences(yseas, 12, 0.5), spec.intervals)
        @test out.lo_95 == [q6(S[j, :], 0.025) for j in 1:18]
        @test out.hi_95 == [q6(S[j, :], 0.975) for j in 1:18]
        @test out.lo_80 == [q6(S[j, :], 0.1) for j in 1:18]
        @test out.hi_80 == [q6(S[j, :], 0.9) for j in 1:18]
        # Type 6 bounds, not type 7.
        @test out.lo_95 != [quantile(S[j, :], 0.025) for j in 1:18]
        # Levels keep their given order.
        rev = forecast(f, 18; levels=(95, 80))
        @test keys(rev) == (:ds, :y_hat, :lo_95, :hi_95, :lo_80, :hi_80)
        @test rev.lo_80 == out.lo_80 && rev.hi_95 == out.hi_95

        smp = forecast_samples(f, 18)
        @test keys(smp) == (:ds, :sample, :y_hat)
        @test length(smp.y_hat) == 18 * 100 && smp.sample[1:3] == 1:3
        @test smp.ds[1:100] == fill(out.ds[1], 100)
        @test smp.y_hat[smp.ds .== out.ds[3]] == S[3, :]
        @test smp.sample[smp.ds .== out.ds[3]] == 1:100

        # The spec's rng is never advanced.
        @test forecast(f, 18; levels=(95,)) == forecast(f, 18; levels=(95,))
        @test forecast_samples(f, 4) == forecast_samples(f, 4)
        r0 = rand(copy(spec.intervals.rng))
        forecast(f, 6; levels=(95,)); forecast_samples(f, 6)
        @test rand(copy(spec.intervals.rng)) == r0
        # The method copies the rng it is given.
        r = Random.Xoshiro(7)
        fr = fit(csp(rng=r), (ds=monthly, y=yseas))
        a = forecast(fr, 6; levels=(95,))
        rand(r)
        @test forecast(fr, 6; levels=(95,)) == a
        fd = fit(csp(rng=Random.default_rng()), (ds=monthly, y=yseas))
        @test forecast_samples(fd, 3) == forecast_samples(fd, 3)
        @test forecast(f, 6; levels=95) == forecast(f, 6; levels=(95,))
        @test forecast(f, 6; levels=[95.0]) == forecast(f, 6; levels=(95,))
        # Horizon 1 and a fractional level.
        one = forecast(f, 1; levels=(99.5,))
        @test keys(one) == (:ds, :y_hat, :lo_99_5, :hi_99_5) && length(one.y_hat) == 1
        @test C._check_levels(nothing) == Float64[]
    end

    @testset "forecast without levels does not sample" begin
        # An empty pool matters only when drawing.
        short = (unique_id=[fill("a", 60); fill("z", 13)], ds=[monthly; monthly[1:13]],
                 y=[yseas; yseas[1:13]])
        z = fit(csp(id=:unique_id, calibration=0.05), short)
        @test length(forecast(z, 3).y_hat) == 6
        @test_throws("series unique_id=\"z\": the residual pool for step 1 is empty",
                     forecast(z, 3; levels=(95,)))
        plain = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1))
        fp = fit(plain, (ds=monthly, y=yseas))
        @test forecast(fp, 3) == forecast(fp, 3; levels=nothing)
        @test keys(forecast(fp, 3)) == (:ds, :y_hat)
    end

    @testset "panel intervals: one rng copy per series" begin
        ids = [fill("b", 60); fill("a", 48)]
        panel = (unique_id=ids, ds=[monthly; monthly[13:60]],
                 y=[yseas; 3 .* yseas[13:60]])
        spec = csp(id=:unique_id)
        out = forecast(fit(spec, panel), 4; levels=(95,))
        @test keys(out) == (:unique_id, :ds, :y_hat, :lo_95, :hi_95)
        @test out.unique_id == [fill("b", 4); fill("a", 4)]
        ya, yb = 3 .* yseas[13:60], yseas
        # Each series gets its own copy of the rng.
        Sa = C._pool_samples(copy(spec.intervals.rng), ya, out.y_hat[5:8],
                             C._seasonal_differences(ya, 12, 0.5), spec.intervals)
        Sb = C._pool_samples(copy(spec.intervals.rng), yb, out.y_hat[1:4],
                             C._seasonal_differences(yb, 12, 0.5), spec.intervals)
        @test out.lo_95 == [[q6(Sb[j, :], 0.025) for j in 1:4];
                            [q6(Sa[j, :], 0.025) for j in 1:4]]
        @test out.hi_95 == [[q6(Sb[j, :], 0.975) for j in 1:4];
                            [q6(Sa[j, :], 0.975) for j in 1:4]]
        # Draws do not depend on the other series.
        isa_ = ids .== "a"
        only_a = forecast(fit(spec, map(c -> c[isa_], panel)), 4; levels=(95,))
        @test only_a.lo_95 == out.lo_95[5:8] && only_a.hi_95 == out.hi_95[5:8]
        p = shuffle(StableRNG(2), 1:length(ids))
        shuffled = forecast(fit(spec, map(c -> c[p], panel)), 4; levels=(95,))
        key(o) = Dict((o.unique_id[i], o.ds[i]) => (o.y_hat[i], o.lo_95[i], o.hi_95[i])
                      for i in eachindex(o.ds))
        @test key(shuffled) == key(out)
        smp = forecast_samples(fit(spec, panel), 4)
        @test keys(smp) == (:unique_id, :ds, :sample, :y_hat)
        @test length(smp.y_hat) == 2 * 4 * 100
        @test smp.unique_id == repeat(out.unique_id; inner=100)
        @test smp.y_hat[1:100] == Sb[1, :] && smp.y_hat[401:500] == Sa[1, :]
        # Backtest shortens each series' horizon on its own.
        tbl, draws = C._forecast_samples(fit(spec, panel), [2, 5], nothing)
        @test size.(draws) == [(2, 100), (5, 100)] && length(tbl.y_hat) == 7
        @test tbl.unique_id == [fill("b", 2); fill("a", 5)]
        # Past one season too.
        long = forecast(fit(spec, panel), 14; levels=(95,))
        @test long.lo_95[15:28] ==
              forecast(fit(spec, map(c -> c[isa_], panel)), 14; levels=(95,)).lo_95

        # A one-series panel equals one series.
        one = (unique_id=fill("s", 60), ds=monthly, y=yseas)
        po = forecast(fit(csp(id=:unique_id), one), 6; levels=(80, 95))
        so = forecast(fit(csp(), (ds=monthly, y=yseas)), 6; levels=(80, 95))
        @test po.y_hat == so.y_hat && po.lo_80 == so.lo_80 && po.hi_80 == so.hi_80
        @test po.lo_95 == so.lo_95 && po.hi_95 == so.hi_95
        ps = forecast_samples(fit(csp(id=:unique_id), one), 6)
        @test ps.y_hat == forecast_samples(fit(csp(), (ds=monthly, y=yseas)), 6).y_hat
    end

    @testset "adaptive weight end to end: 0.3 with two same-season values" begin
        T = 30
        ya = [50 + 10 * sin(2π * i / 12) + noise[i] for i in 1:T]
        fa = fit(ConformalSeasonalPool(12; freq=Month(1), samples=2000,
                                       rng=StableRNG(5)), (ds=monthly[1:T], y=ya))
        smp = forecast_samples(fa, 12)
        share(j) = begin
            vals = ya[mod1(T + j, 12):12:T]
            mean(in(vals), smp.y_hat[(1:2000) .+ 2000 * (j - 1)])
        end
        # Steps 1-6 see two values, steps 7-12 three.
        @test length(ya[mod1(T + 1, 12):12:T]) == 2 && length(ya[mod1(T + 7, 12):12:T]) == 3
        @test abs(mean(share.(1:6)) - 0.3) < 0.02
        @test abs(mean(share.(7:12)) - 0.5) < 0.02
    end

    @testset "ConformalSeasonalPool(1) end to end" begin
        f1 = fit(ConformalSeasonalPool(1; freq=Month(1), rng=StableRNG(3)),
                 (ds=monthly, y=yseas))
        out = forecast(f1, 5; levels=(90,))
        @test out.y_hat == fill(yseas[end], 5)
        R = C._seasonal_differences(yseas, 1, 0.5)
        @test R == diff(yseas)[end - 29:end]
        smp = forecast_samples(f1, 5)
        # No season: last value plus a difference.
        @test all(in(Set(yseas[end] .+ R)), smp.y_hat)
        @test all(out.lo_90 .< yseas[end] .+ maximum(R))
    end

    @testset "future is ignored by forecast_samples without Exogenous features" begin
        f = fit(csp(), (ds=monthly, y=yseas))
        tf = collect(Date(2020, 1, 1):Month(1):Date(2020, 4, 1))
        base = forecast_samples(f, 4)
        with = @test_logs (:warn, r"`future` was passed") forecast_samples(
            f, 4; future=(ds=tf, promo=[1.0, 2.0, 3.0, 4.0]))
        @test with == base
        lv = forecast(f, 4; levels=(95,))
        @test lv.lo_95 == [q6(with.y_hat[with.ds .== tf[j]], 0.025) for j in 1:4]
        @test (@test_logs (:warn, r"`future` was passed") forecast(
            f, 4; levels=(95,), future=(ds=tf,))) == lv
    end

    @testset "interval errors are actionable" begin
        plain = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1))
        fp = fit(plain, (ds=monthly, y=yseas))
        no(who) = "$who needs an interval method, but this forecaster has none. Build " *
                  "it with ConformalSeasonalPool(period; freq=...), or pass " *
                  "intervals=PoolIntervals(period) with a SeasonalNaive(period) model."
        @test_throws no("forecast with levels") forecast(fp, 3; levels=(95,))
        @test_throws no("forecast_samples") forecast_samples(fp, 3)

        f = fit(csp(), (ds=monthly, y=yseas))
        @test_throws "an interval level is a percentage in (1, 100), got 0.95. Pass e.g. " *
                     "95 for a 95% interval, not 0.95." forecast(f, 3; levels=(0.95,))
        @test_throws("levels is empty; pass at least one level, e.g. levels=(95,).",
                     forecast(f, 3; levels=()))
        @test_throws "levels repeats a level, got (95, 95); pass each level once, e.g. " *
                     "levels=(80, 95)." forecast(f, 3; levels=(95, 95))
        @test_throws "levels repeats a level" forecast(f, 3; levels=(95, 95.0))
        @test_throws "levels must be percentages, got (\"95\",). Pass e.g. " *
                     "levels=(80, 95)." forecast(f, 3; levels=("95",))
        @test_throws "levels must be percentages, got :a." forecast(f, 3; levels=:a)
        @test_throws "forecast horizon h must be ≥ 1, got 0" forecast_samples(f, 0)
        @test_throws("forecast horizon h must be a whole number, not the Bool true",
                     forecast_samples(f, true))

        # Calibration below one value: empty pool.
        short = (unique_id=[fill("a", 60); fill("z", 13)], ds=[monthly; monthly[1:13]],
                 y=[yseas; yseas[1:13]])
        z = fit(csp(id=:unique_id, calibration=0.05), short)
        @test_throws("series unique_id=\"z\": the residual pool for step 1",
                     forecast_samples(z, 3))

        # Fit rejects gaps, NaN, Inf and missing.
        gap = [monthly[1:20]; monthly[22:60]]
        @test_throws "has 1 gap" fit(csp(), (ds=gap, y=yseas[1:59]))
        @test_throws "non-finite value NaN" fit(csp(), (ds=monthly, y=[yseas[1:59]; NaN]))
        @test_throws "non-finite value Inf" fit(csp(), (ds=monthly, y=[yseas[1:59]; Inf]))
        @test_throws "contains missing values" fit(
            csp(), (ds=monthly, y=Union{Missing,Float64}[yseas[1:59]; missing]))

        # :sample is reserved; intervals may not clash.
        @test_throws "time=:sample collides with a column name reserved by forecast()/" *
                     "backtest() results (:origin, :step, :y, :y_hat, :sample, :fold). " *
                     "Rename the time column in your data." csp(time=:sample)
        @test_throws "id=:sample collides" csp(id=:sample)
        clash = fit(csp(time=:lo_95), (lo_95=monthly, y=yseas))
        @test_throws "the interval column :lo_95 would overwrite a column of the " *
                     "result. Rename your time or id column." forecast(
            clash, 3; levels=(95,))
    end

    @testset "backtest scores samples and adds interval columns" begin
        spec = csp()
        data = (ds=monthly, y=yseas)
        ms = (mae, crps, Coverage(95), IntervalWidth(95))
        r = backtest(spec, data; horizon=6, windows=3, levels=(95,), metrics=ms)
        @test keys(r.forecasts) == (:fold, :origin, :step, :ds, :y, :y_hat, :lo_95, :hi_95)
        @test summarize(r).metric == [:mae, :crps, :coverage_95, :interval_width_95]
        @test unique(r.forecasts.origin) == monthly[[42, 48, 54]]
        for k in 1:3
            o = 36 + 6k
            fk = fit(spec, map(c -> c[1:o], data))
            manual = forecast(fk, 6; levels=(95,))
            sel = r.forecasts.fold .== k
            truth = yseas[o .+ (1:6)]
            @test r.forecasts.y[sel] == truth
            @test r.forecasts.y_hat[sel] == manual.y_hat
            @test r.forecasts.lo_95[sel] == manual.lo_95
            @test r.forecasts.hi_95[sel] == manual.hi_95
            # Every fold starts from a fresh copy of the same rng.
            S = only(C._forecast_samples(fk, [6], nothing)[2])
            val(name) = only(r.scores.value[(r.scores.fold .== k) .&
                                            (r.scores.metric .== name)])
            @test val(:crps) == crps(truth, S)
            @test val(:coverage_95) == Coverage(95)(truth, S)
            @test val(:interval_width_95) == IntervalWidth(95)(truth, S)
            @test val(:mae) == mae(truth, manual.y_hat)
            # The metric and the columns come from the same samples.
            @test val(:coverage_95) == mean(manual.lo_95 .≤ truth .≤ manual.hi_95)
        end
        two = backtest(spec, data; horizon=6, windows=2, levels=(80, 95), metrics=(crps,))
        @test keys(two.forecasts)[end - 3:end] == (:lo_80, :hi_80, :lo_95, :hi_95)
        # Levels add columns and change no point result.
        plain = backtest(spec, data; horizon=6, windows=2)
        withlv = backtest(spec, data; horizon=6, windows=2, levels=(95,))
        @test keys(plain.forecasts) == (:fold, :origin, :step, :ds, :y, :y_hat)
        @test plain.forecasts.y_hat == withlv.forecasts.y_hat
        @test plain.scores == withlv.scores
        # Horizon 1.
        one = backtest(spec, data; horizon=1, windows=2, levels=(95,), metrics=(crps,))
        @test all(isfinite, one.scores.value) && length(one.forecasts.y_hat) == 2
        # A metric of your own on samples.
        mine = backtest(spec, data; horizon=6, windows=2, metrics=(TestModels.spread,))
        @test summarize(mine).metric == [:spread] && all(>(0), mine.scores.value)
        # A level given as a scalar and a NamedTuple of metrics.
        sc = backtest(spec, data; horizon=6, windows=2, levels=95,
                      metrics=(score=crps, cover=Coverage(95)))
        @test keys(sc.forecasts)[end - 1:end] == (:lo_95, :hi_95)
        @test summarize(sc).metric == [:score, :cover]
    end

    @testset "ConformalSeasonalPool(1) backtest" begin
        s1 = ConformalSeasonalPool(1; freq=Month(1), rng=StableRNG(3))
        data = (ds=monthly, y=yseas)
        r = backtest(s1, data; horizon=3, windows=2, levels=(90,),
                     metrics=(crps, Coverage(90)))
        for (k, o) in enumerate((54, 57))
            fk = fit(s1, map(c -> c[1:o], data))
            manual = forecast(fk, 3; levels=(90,))
            sel = r.forecasts.fold .== k
            @test r.forecasts.y_hat[sel] == fill(yseas[o], 3)
            @test r.forecasts.lo_90[sel] == manual.lo_90
            @test r.forecasts.hi_90[sel] == manual.hi_90
            S = only(C._forecast_samples(fk, [3], nothing)[2])
            sk = r.scores.fold .== k
            @test r.scores.value[sk] ==
                  [crps(yseas[o .+ (1:3)], S), Coverage(90)(yseas[o .+ (1:3)], S)]
        end
    end

    @testset "a panel backtest keeps the samples of the scored rows" begin
        # "b" ends at row 54: no fold-2 actuals for it.
        ids = [fill("b", 54); fill("a", 60)]
        panel = (unique_id=ids, ds=[monthly[1:54]; monthly],
                 y=[2 .* yseas[1:54]; yseas])
        spec = csp(id=:unique_id)
        r = backtest(spec, panel; horizon=6, initial=48, step=6, levels=(95,),
                     metrics=(crps, Coverage(95)))
        @test keys(r.forecasts) ==
              (:fold, :origin, :step, :unique_id, :ds, :y, :y_hat, :lo_95, :hi_95)
        sel = r.forecasts.fold .== 2
        @test count(==(1), r.forecasts.fold) == 12 && count(sel) == 6
        @test all(==("a"), r.forecasts.unique_id[sel])
        val(k, name) = only(r.scores.value[(r.scores.fold .== k) .&
                                           (r.scores.metric .== name)])
        # Fold 2: "b" comes first and loses all its rows.
        rows2 = panel.ds .≤ monthly[54]
        f2 = fit(spec, map(c -> c[rows2], panel))
        out2, draws = C._forecast_samples(f2, [6, 6], nothing)
        @test out2.unique_id == [fill("b", 6); fill("a", 6)]
        Sa = draws[2]
        truth = yseas[55:60]
        @test val(2, :crps) == crps(truth, Sa)
        @test val(2, :coverage_95) == Coverage(95)(truth, Sa)
        los, his = C._interval_bounds(Sa, [95.0])
        @test r.forecasts.lo_95[sel] == only(los) && r.forecasts.hi_95[sel] == only(his)
        @test r.forecasts.y_hat[sel] == out2.y_hat[7:12]
        # Fold 1 pools the rows of both series, in series order.
        f1 = fit(spec, map(c -> c[panel.ds .≤ monthly[48]], panel))
        _, d1 = C._forecast_samples(f1, [6, 6], nothing)
        truth1 = [2 .* yseas[49:54]; yseas[49:54]]
        @test val(1, :crps) == crps(truth1, reduce(vcat, d1))
        @test val(1, :coverage_95) == Coverage(95)(truth1, reduce(vcat, d1))
    end

    @testset "a one-series panel backtest equals the single series" begin
        data = (ds=monthly, y=yseas)
        pan = (unique_id=fill("a", 60), ds=monthly, y=yseas)
        ms = (mae, crps, Coverage(80), IntervalWidth(80))
        one = backtest(csp(), data; horizon=6, windows=3, levels=(80, 95), metrics=ms)
        many = backtest(csp(id=:unique_id), pan; horizon=6, windows=3, levels=(80, 95),
                        metrics=ms)
        for c in keys(one.forecasts)
            @test one.forecasts[c] == many.forecasts[c]
        end
        @test many.forecasts.unique_id == fill("a", 18)
        @test one.scores == many.scores
    end

    @testset "a ragged panel with windows: columns, scores and draws agree" begin
        t96 = collect(Date(2012, 1, 1):Month(1):Date(2019, 12, 1))
        noise96 = randn(StableRNG(2), 96)
        ya = [30 + 8 * sin(2π * i / 12) + 0.1i + noise96[i] for i in 1:96]
        # C ends 6 months before A, so the fold origins differ.
        panel = (unique_id=[fill("A", 96); fill("C", 54)], ds=[t96; monthly[1:54]],
                 y=[ya; yseas[1:54]])
        spec = csp(id=:unique_id)
        ms = (mae, Mase(12), crps, fair_crps, pinball_loss, Coverage(95),
              IntervalWidth(80))
        r = backtest(spec, panel; horizon=6, windows=3, levels=(80, 95), metrics=ms)
        @test all(ismissing, r.scores.origin)
        @test keys(r.forecasts) == (:fold, :origin, :step, :unique_id, :ds, :y, :y_hat,
                                    :lo_80, :hi_80, :lo_95, :hi_95)
        @test length(r.forecasts.y) == 3 * 2 * 6
        @test summarize(r).metric == [:mae, :mase_12, :crps, :fair_crps, :pinball_loss,
                                      :coverage_95, :interval_width_80]
        val(k, name) = only(r.scores.value[(r.scores.fold .== k) .&
                                           (r.scores.metric .== name)])
        for k in 1:3
            cutA, cutC = 90 - 6 * (3 - k), 48 - 6 * (3 - k)
            sel = r.forecasts.fold .== k
            @test r.forecasts.unique_id[sel] == [fill("A", 6); fill("C", 6)]
            @test r.forecasts.y[sel] == [ya[cutA .+ (1:6)]; yseas[cutC .+ (1:6)]]
            @test r.forecasts.ds[sel] == [t96[cutA .+ (1:6)]; monthly[cutC .+ (1:6)]]
            train = [i ≤ 96 ? i ≤ cutA : i - 96 ≤ cutC for i in eachindex(panel.y)]
            fk = fit(spec, map(c -> c[train], panel))
            _, d = C._forecast_samples(fk, [6, 6], nothing)
            S = reduce(vcat, d)
            truth = r.forecasts.y[sel]
            los, his = C._interval_bounds(S, [80.0, 95.0])
            @test r.forecasts.lo_80[sel] == los[1] && r.forecasts.hi_80[sel] == his[1]
            @test r.forecasts.lo_95[sel] == los[2] && r.forecasts.hi_95[sel] == his[2]
            @test val(k, :crps) == crps(truth, S)
            @test val(k, :fair_crps) == fair_crps(truth, S)
            @test val(k, :pinball_loss) == pinball_loss(truth, S)
            @test val(k, :interval_width_80) == IntervalWidth(80)(truth, S)
            # The score and the columns come from the same draws.
            share = mean(r.forecasts.lo_95[sel] .≤ truth .≤ r.forecasts.hi_95[sel])
            @test val(k, :coverage_95) == share == Coverage(95)(truth, S)
        end
        sm = summarize(r)
        for (i, name) in enumerate(sm.metric)
            @test sm.mean[i] ≈ mean(r.scores.value[r.scores.metric .== name])
        end
    end

    @testset "intervals use only the fold's training window" begin
        spec = csp()
        later = [yseas[1:48]; 1000 .* yseas[49:60]]
        ms = (mae, crps, Coverage(95))
        bt(y) = backtest(spec, (ds=monthly, y=y); horizon=6, windows=3, levels=(95,),
                         metrics=ms)
        r1, r2 = bt(yseas), bt(later)
        f1 = r1.forecasts.fold .== 1
        for c in (:y, :y_hat, :lo_95, :hi_95)
            @test r1.forecasts[c][f1] == r2.forecasts[c][f1]
        end
        s1 = r1.scores.fold .== 1
        @test r1.scores.value[s1] == r2.scores.value[s1]
        @test any(r1.scores.value[.!s1] .!= r2.scores.value[.!s1])
        # Fold 2 trains up to row 48: only its actuals move.
        f2 = r1.forecasts.fold .== 2
        for c in (:y_hat, :lo_95, :hi_95)
            @test r1.forecasts[c][f2] == r2.forecasts[c][f2]
        end
        @test r1.forecasts.y[f2] != r2.forecasts.y[f2]
        f3 = r1.forecasts.fold .== 3
        @test r1.forecasts.y_hat[f3] == r2.forecasts.y_hat[f3]
        @test r1.forecasts.hi_95[f3] != r2.forecasts.hi_95[f3]
        # Panel: series "a" changes after its own fold-1 window.
        ids = [fill("b", 60); fill("a", 60)]
        ya = yseas .+ 5
        pan(y) = (unique_id=ids, ds=[monthly; monthly], y=[yseas; y])
        ps = csp(id=:unique_id)
        btp(y) = backtest(ps, pan(y); horizon=6, windows=3, levels=(95,), metrics=ms)
        p1, p2 = btp(ya), btp(vcat(ya[1:48], 1000 .* yseas[49:60]))
        g1 = p1.forecasts.fold .== 1
        for c in (:y, :y_hat, :lo_95, :hi_95)
            @test p1.forecasts[c][g1] == p2.forecasts[c][g1]
        end
        @test p1.scores.value[p1.scores.fold .== 1] ==
              p2.scores.value[p2.scores.fold .== 1]
        @test any(p1.scores.value[p1.scores.fold .!= 1] .!=
                  p2.scores.value[p2.scores.fold .!= 1])
        g3 = p1.forecasts.fold .== 3
        @test p1.forecasts.hi_95[g3] != p2.forecasts.hi_95[g3]
    end

    @testset "backtest needs an interval method for levels and sample metrics" begin
        data = (ds=monthly, y=yseas)
        plain = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)), freq=Month(1))
        who = "backtest with levels or a sample-based metric needs an interval method, " *
              "but this forecaster has none. Build it with ConformalSeasonalPool(period; " *
              "freq=...), or pass intervals=PoolIntervals(period) with a " *
              "SeasonalNaive(period) model."
        @test_throws who backtest(plain, data; horizon=6, windows=2, metrics=(crps,))
        @test_throws who backtest(plain, data; horizon=6, windows=2, levels=(95,))
        @test_throws who backtest(plain, data; horizon=6, windows=2,
                                  metrics=(TestModels.spread,))
        pplain = Forecaster(SeasonalNaive(12); features=FeatureSet(Lag(12)),
                            freq=Month(1), id=:unique_id)
        pan = (unique_id=fill("a", 60), ds=monthly, y=yseas)
        @test_throws who backtest(pplain, pan; horizon=6, windows=2, levels=(95,))
        # Point metrics on a plain forecaster still run.
        @test backtest(plain, data; horizon=6, windows=2) isa BacktestResult
        spec = csp()
        @test_throws "an interval level is a percentage in (1, 100), got 0.95" backtest(
            spec, data; horizon=6, windows=2, levels=(0.95,))
        @test_throws "levels is empty" backtest(spec, data; horizon=6, windows=2,
                                                levels=())
        @test_throws "which is not Point(), Scaled() or Samples()." backtest(
            spec, data; horizon=6, windows=2, metrics=(TestModels.badtrait,))
        # tune refuses a sample metric before any fit.
        @test_throws "tune ranks candidates by a point-forecast metric" tune(
            spec, data; grid=(model=[SeasonalNaive(12)],), horizon=6, windows=2,
            metric=crps)
    end

    @testset "an empty residual pool names the fold" begin
        # 5 % of 14 values leaves no seasonal difference.
        thin = csp(calibration=0.05)
        @test_throws("backtest fold 1 (origin 2016-02-01): the residual pool for " *
                     "step 1 is empty",
                     backtest(thin, (ds=monthly, y=yseas); horizon=3, initial=14,
                              levels=(95,)))
        @test_throws("backtest fold 1 (origin 2016-02-01): the residual pool",
                     backtest(thin, (ds=monthly, y=yseas); horizon=3, initial=14,
                              metrics=(crps,)))
        ids = [fill("a", 60); fill("z", 14)]
        panel = (unique_id=ids, ds=[monthly; monthly[1:14]], y=[yseas; yseas[1:14]])
        # Each series is cut at its own end, so no single origin.
        @test_throws("backtest fold 1: series unique_id=\"z\": the residual pool for " *
                     "step 1 is empty",
                     backtest(csp(id=:unique_id, calibration=0.05), panel;
                              horizon=1, windows=1, levels=(95,)))
        @test_throws("backtest fold 1 (origin 2016-02-01): series unique_id=\"a\": the " *
                     "residual pool for step 1 is empty",
                     backtest(csp(id=:unique_id, calibration=0.05), panel;
                              horizon=3, initial=14, levels=(95,)))
        # Without levels or sample metrics nothing is sampled.
        @test backtest(thin, (ds=monthly, y=yseas); horizon=3, initial=14) isa
              BacktestResult
        @test backtest(csp(id=:unique_id, calibration=0.05), panel; horizon=1,
                       windows=1) isa BacktestResult
        @test C._in_fold(() -> 1, 2, missing) == 1
        @test_throws "backtest fold 2: boom" C._in_fold(
            () -> throw(ArgumentError("boom")), 2, missing)
        @test_throws ErrorException C._in_fold(() -> error("x"), 2, missing)
    end
end
