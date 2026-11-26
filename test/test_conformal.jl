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
    end

    @testset "the docs state that there is no coverage guarantee" begin
        for doc in (string(@doc(C.PoolIntervals)), string(@doc(ConformalSeasonalPool)))
            @test occursin("no coverage guarantee", doc)
            @test !occursin(r"\bvalid\b|\bguaranteed\b", doc)
        end
    end
end
