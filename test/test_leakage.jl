@testset "leakage" begin
    M = MachineLearningForecast
    n = 36
    rng = StableRNG(2027)
    t = collect(Date(2022, 1, 3):Day(1):(Date(2022, 1, 3) + Day(n - 1)))
    y = 10 .+ cumsum(randn(rng, n))
    promo = Float64.(rand(rng, 0:1, n))

    # Reads only h, so only its argument can leak.
    customs = [CustomFeature(:hist_mean, mean, 1),
               CustomFeature(:last_step, h -> h[end] - h[end - 1], 2),
               CustomFeature(:recent_max, h -> maximum(h[max(1, end - 3):end]), 1)]
    # parent(h) is the whole series in training only.
    peek(h) = (p = parent(h); length(p) > length(h) ? p[length(h) + 1] : 0.0)
    leaky = CustomFeature(:peek, peek, 0)
    # Leaks through a minimum, so only a low rewrite exposes it.
    function peekmin(h)
        w = minimum(h[max(1, end - 9):end])
        return h isa SubArray ? min(w, parent(h)[length(h) + 1]) : w
    end
    leaky_min = CustomFeature(:peekmin, peekmin, 1)
    # A one-sided rewrite hides a leak through min or max.
    junk = (1e6, -1e6)
    feats = [Lag(1), Lag(5), RollingMean(3), RollingStd(4; lag=2), RollingMin(2),
             RollingMax(3; lag=3), Diff(2; lag=2), Diff(1; lag=1),
             Calendar(:dayofweek, :month), Fourier(7, 2), Exogenous(:promo), customs...]

    batch(f, yv) = columns(f, yv, t, (ds=t, y=yv, promo=promo))
    row(f, yv, i) = rowvalues(f, yv[1:(i - 1)], t[i], i - 1, (promo=promo[i],))

    function leaks(f, signs=junk)
        acc = batch(f, y)
        mh = M.minhistory(f)
        truncated, rewritten = Int[], Int[]
        for t0 in mh:(n - 1)
            isequal(row(f, y, t0 + 1), [c[t0 + 1] for (_, c) in acc]) ||
                push!(truncated, t0)
            changed = false
            for v in signs
                y2 = copy(y)
                y2[(t0 + 1):end] .= v
                alt = batch(f, y2)
                changed |= !all(isequal(a[1:(t0 + 1)], b[1:(t0 + 1)])
                                for ((_, a), (_, b)) in zip(acc, alt))
            end
            changed && push!(rewritten, t0)
        end
        return (; truncated, rewritten)
    end

    @testset "feature at every cut: $(repr(f))" for f in feats
        r = leaks(f)
        @test r.truncated == Int[]
        @test r.rewritten == Int[]
    end

    @testset "the feature checks catch a feature that peeks" begin
        r = leaks(leaky)
        @test r.truncated == collect(0:(n - 1))
        @test r.rewritten == collect(0:(n - 1))
        @test only(last.(batch(leaky, y))) == y
        mh = M.minhistory(leaky_min)
        r = leaks(leaky_min)
        @test r.rewritten == collect(mh:(n - 1))
    end

    mh_all = M.minhistory(FeatureSet(feats...))
    spy() = TestModels.KeepPredict(Ref(Any[]))

    function table(series, ranges; panel, withy=true)
        cols = (ds=reduce(vcat, [s.t[r] for (s, r) in zip(series, ranges)]),
                promo=reduce(vcat, [s.promo[r] for (s, r) in zip(series, ranges)]))
        withy && (cols = merge(cols,
            (y=reduce(vcat, [s.y[r] for (s, r) in zip(series, ranges)]),)))
        panel || return cols
        ids = reduce(vcat, [fill(k, length(r)) for (k, r) in enumerate(ranges)])
        return merge((id=ids,), cols)
    end

    forecaster(model, fs, strat, panel) = Forecaster(model; features=fs, strategy=strat,
        freq=Day(1), id=panel ? :id : nothing)

    function skewed_cuts(fs, strat, series, cuts; panel)
        mh = M.minhistory(fs)
        full = fit(forecaster(TestModels.KeepTrain(), fs, strat, panel),
                   table(series, [1:length(s.y) for s in series]; panel))
        nX = [length(s.y) - mh for s in series]
        hmax = strat isa Direct ? strat.max_horizon : 1
        bad = Int[]
        for (c, cut) in enumerate(cuts)
            pre = fit(forecaster(spy(), fs, strat, panel),
                      table(series, [1:k for k in cut]; panel))
            forecast(pre, 3; future=table(series, [(k + 1):(k + 3) for k in cut];
                                          panel, withy=false))
            ok = true
            for i in 1:hmax
                X = full.models[strat isa Direct ? i : 1].fitresult.X
                call = pre.models[strat isa Direct ? i : 1].model.calls[][1]
                shift = strat isa Direct ? i - 1 : 0
                for k in eachindex(series)
                    j = sum(nX[1:(k - 1)] .- shift) + cut[k] + 1 - mh
                    ok &= all(isequal(call[name][panel ? k : 1], X[name][j])
                              for name in keys(X))
                end
            end
            ok || push!(bad, c)
        end
        return bad
    end

    function rewritten_cuts(fs, strat, series, cuts; panel)
        mh = M.minhistory(fs)
        ranges = [1:length(s.y) for s in series]
        spec(m) = forecaster(m, fs, strat, panel)
        full = fit(spec(TestModels.KeepTrain()), table(series, ranges; panel))
        nX = [length(s.y) - mh for s in series]
        hmax = strat isa Direct ? strat.max_horizon : 1
        bad = Int[]
        for (c, cut) in enumerate(cuts)
            ok = true
            for v in junk
                alt_series = [(t=s.t, promo=s.promo,
                               y=[s.y[1:k]; fill(v, length(s.y) - k)])
                              for (s, k) in zip(series, cut)]
                alt = fit(spec(TestModels.KeepTrain()), table(alt_series, ranges; panel))
                for i in 1:hmax
                    shift = strat isa Direct ? i - 1 : 0
                    X = full.models[strat isa Direct ? i : 1].fitresult.X
                    X2 = alt.models[strat isa Direct ? i : 1].fitresult.X
                    for k in eachindex(series)
                        o = sum(nX[1:(k - 1)] .- shift)
                        rows = (o + 1):(o + cut[k] + 1 - mh)
                        ok &= all(isequal(X[name][rows], X2[name][rows])
                                  for name in keys(X))
                    end
                end
            end
            ok || push!(bad, c)
        end
        return bad
    end

    fs_all = FeatureSet(feats...)
    single = [(t=t, y=y, promo=promo)]
    lens, starts, offs = (34, 31, 27), (0, 2, 5), (0, 2, -3)
    ragged = map(1:3) do k
        r = StableRNG(40 + k)
        tk = collect((Date(2022, 1, 3) + Day(starts[k])):Day(1):(Date(2022, 1, 3) +
                      Day(starts[k] + lens[k] - 1)))
        (t=tk, y=5k .+ cumsum(randn(r, lens[k])), promo=Float64.(rand(r, 0:1, lens[k])))
    end
    cuts_single = [[c] for c in (mh_all + 4):(n - 3)]
    cuts_ragged = [[c + o for o in offs] for c in (mh_all + 6):26]

    kinds = [(s, p) for s in (Recursive(), Direct(3)) for p in (false, true)]
    @testset "end to end, $(nameof(typeof(strat))), panel=$panel" for (strat, panel) in
            kinds
        series, cuts = panel ? (ragged, cuts_ragged) : (single, cuts_single)
        @test length(cuts) > 10
        @test skewed_cuts(fs_all, strat, series, cuts; panel) == Int[]
        @test rewritten_cuts(fs_all, strat, series, cuts; panel) == Int[]
    end

    @testset "the end-to-end checks catch a feature that peeks" begin
        fs = FeatureSet(Lag(1), leaky, Exogenous(:promo))
        fs_min = FeatureSet(Lag(1), leaky_min, Exogenous(:promo))
        shapes = ((single, cuts_single, false), (ragged, cuts_ragged, true))
        for strat in (Recursive(), Direct(3)), (series, cuts, panel) in shapes
            all_cuts = collect(eachindex(cuts))
            @test skewed_cuts(fs, strat, series, cuts; panel) == all_cuts
            @test rewritten_cuts(fs, strat, series, cuts; panel) == all_cuts
            @test rewritten_cuts(fs_min, strat, series, cuts; panel) == all_cuts
        end
    end

    @testset "backtest forecasts ignore the held-out and later actuals" begin
        fs = FeatureSet(Lag(1), Lag(5), RollingMean(3), Diff(1), Calendar(:dayofweek),
                        Fourier(7, 1), Exogenous(:promo), customs[1])
        data(yv) = (ds=t, y=yv, promo=promo)
        bt(m, strat, yv) = backtest(Forecaster(m; features=fs, strategy=strat,
                                               freq=Day(1)), data(yv);
                                    horizon=3, initial=20, step=5, metrics=(mae,))
        learners = (() -> EvoTreeRegressor(nrounds=5),
                    () -> DecisionTreeRegressor(max_depth=3, rng=StableRNG(1)))
        combos = [(make, strat) for make in learners for strat in (Recursive(), Direct(3))]
        @testset "$(nameof(typeof(make()))), $strat" for (make, strat) in combos
            base = bt(make(), strat, y)
            @test unique(base.folds.fold) == [1, 2, 3]
            for k in 1:3
                y2 = copy(y)
                after = (20 + 5 * (k - 1) + 1):n
                y2[after] .= 1e6 .* (-1.0) .^ (1:length(after))
                alt = bt(make(), strat, y2)
                kept = base.folds.fold .<= k
                @test base.folds.y_hat[kept] == alt.folds.y_hat[kept]
                # The rewrite does reach the later folds that train on it.
                k < 3 && @test base.folds.y_hat[.!kept] != alt.folds.y_hat[.!kept]
            end
        end
    end

    @testset "panel backtest forecasts ignore the held-out and later actuals" begin
        fs = FeatureSet(Lag(1), Lag(5), RollingMean(3), Diff(1), Calendar(:dayofweek),
                        Fourier(7, 1), Exogenous(:promo), customs[1])
        tbl = table(ragged, [1:length(s.y) for s in ragged]; panel=true)
        bt(strat, yv) = backtest(
            Forecaster(DecisionTreeRegressor(max_depth=3, rng=StableRNG(1));
                       features=fs, strategy=strat, freq=Day(1), id=:id),
            merge(tbl, (y=yv,)); horizon=3, initial=18, step=5, metrics=(mae,))
        @testset "$strat" for strat in (Recursive(), Direct(3))
            base = bt(strat, tbl.y)
            @test unique(base.folds.fold) == [1, 2, 3]
            for k in 1:3
                origin = t[1] + Day(18 + 5 * (k - 1) - 1)
                later = findall(>(origin), tbl.ds)
                y2 = copy(tbl.y)
                y2[later] .= 1e6 .* (-1.0) .^ (1:length(later))
                alt = bt(strat, y2)
                kept = base.folds.fold .<= k
                @test base.folds.y_hat[kept] == alt.folds.y_hat[kept]
                k < 3 && @test base.folds.y_hat[.!kept] != alt.folds.y_hat[.!kept]
            end
        end
    end
end
