@testset "backtest" begin
    n = 100
    t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
    y = Float64.(1:n)
    df = (ds=t, y=y)
    fs = FeatureSet(Lag(1))
    # Echoing y_lag_1 is the naive forecast.
    fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=fs,
                    strategy=Recursive(), freq=Day(1))

    @testset "windows holds out the last folds of the series" begin
        # 3 windows of 5, 10 apart, end at row 100: origins 75, 85, 95.
        rw = backtest(fc, df; horizon=5, windows=3, step=10, metrics=(mae, rmse))
        ri = backtest(fc, df; horizon=5, initial=75, step=10, metrics=(mae, rmse))
        @test rw.forecasts == ri.forecasts && isequal(rw.scores, ri.scores)
        @test unique(rw.forecasts.origin) == t[[75, 85, 95]]
        @test last(rw.forecasts.ds) == t[end]
        @test unique(backtest(fc, df; horizon=5, windows=1).forecasts.fold) == [1]
        @test_throws("backtest needs exactly one of initial (folds growing from the " *
                     "start of the data) and windows", backtest(fc, df; horizon=5))
        @test_throws("got initial=80 and windows=2. Pass e.g. initial=730 or windows=3.",
                     backtest(fc, df; horizon=5, initial=80, windows=2))
        @test_throws("backtest windows must be ≥ 1, got 0. Pass the number of folds",
                     backtest(fc, df; horizon=5, windows=0))
        @test_throws("backtest windows=30 with step=5 and horizon=5 would train its " *
                     "first fold on -50 of the 100 rows, but the feature set needs " *
                     "more than 1.", backtest(fc, df; horizon=5, windows=30))
        @test_throws("tune needs exactly one of initial",
                     tune(fc, df; grid=(features=[fs],), horizon=5))
    end

    @testset "fold structure and exact values" begin
        res = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(mae, rmse))
        # origins: 80, 90 (100 would need rows 101:105)
        @test res isa BacktestResult
        @test keys(res.forecasts) == (:fold, :origin, :step, :ds, :y, :y_hat)
        @test res.forecasts.fold == [fill(1, 5); fill(2, 5)]
        @test length(res.forecasts.step) == 2 * 5
        @test res.forecasts.step == [1:5; 1:5]
        @test res.forecasts.origin == [fill(t[80], 5); fill(t[90], 5)]
        @test res.forecasts.ds == [t[81:85]; t[91:95]]
        @test res.forecasts.y == [y[81:85]; y[91:95]]
        @test res.forecasts.y_hat == [fill(80.0, 5); fill(90.0, 5)]
        # per-fold metrics: |y - origin| averaged: mean(1,2,3,4,5) = 3
        m = res.scores
        @test length(m.fold) == 4   # 2 folds × 2 metrics
        @test all(m.value[m.metric .== :mae] .≈ 3.0)
        @test !any(ismissing, m.origin)
        s = summarize(res)
        @test only(s.mean[s.metric .== :mae]) ≈ 3.0
        @test only(s.mean[s.metric .== :rmse]) ≈ sqrt(mean([1, 4, 9, 16, 25]))
        @test Tables.istable(res.forecasts) && Tables.istable(res.scores)
        s = sprint(show, MIME"text/plain"(), res)
        @test occursin("2 folds", s) && occursin("mae", s)
    end

    @testset "exogenous columns are sliced from held-out data" begin
        dfe = (ds=t, y=y, promo=Float64.(1:n) .* 10)
        fce = Forecaster(TestModels.EchoColumn(:promo);
                         features=FeatureSet(Lag(1), Exogenous(:promo)),
                         strategy=Recursive(), freq=Day(1))
        res = backtest(fce, dfe; horizon=3, initial=90, step=100, metrics=(mae,))
        @test res.forecasts.y_hat == dfe.promo[91:93]
    end

    @testset "rmsse is scaled by each fold's training window" begin
        yw = [i + 3 * sin(i) + (i % 7) for i in 1:n]
        dw = (ds=t, y=yw)
        for m in (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fcr = Forecaster(m; features=fs, strategy=Recursive(), freq=Day(1))
            res = backtest(fcr, dw; horizon=5, initial=60, step=10,
                           metrics=(rmse, rmsse))
            mt = res.scores
            folds = sort(unique(res.forecasts.fold))
            for k in folds
                sel = res.forecasts.fold .== k
                o = findfirst(==(res.forecasts.origin[findfirst(sel)]), t)
                tr = yw[1:o]
                sc = mean(abs2(tr[i] - tr[i - 1]) for i in 2:o)
                got = only(mt.value[(mt.fold .== k) .& (mt.metric .== :rmsse)])
                @test got ≈
                      rmse(res.forecasts.y[sel], res.forecasts.y_hat[sel]) / sqrt(sc)
            end
            vals = mt.value[mt.metric .== :rmsse]
            s = summarize(res)
            @test only(s.mean[s.metric .== :rmsse]) ≈ mean(vals)
        end
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
        for bad in (true, false)
            @test_throws "backtest horizon must be a whole number, not the Bool $bad. " *
                         "Pass the number of steps each fold forecasts, e.g. " *
                         "horizon=28." backtest(fc, df; horizon=bad, initial=50)
            @test_throws "backtest initial must be a whole number, not the Bool $bad. " *
                         "Pass the size of the first training window, e.g. " *
                         "initial=730." backtest(fc, df; horizon=5, initial=bad)
            @test_throws "backtest windows must be a whole number, not the Bool $bad. " *
                         "Pass the number of folds to hold out at the end of each " *
                         "series, e.g. windows=3." backtest(fc, df; horizon=5,
                                                             windows=bad)
            @test_throws "backtest step must be a whole number, not the Bool $bad. " *
                         "Pass how many steps each fold moves the origin, e.g. " *
                         "step=5, which equals horizon (the default)." backtest(
                fc, df; horizon=5, initial=50, step=bad)
        end
        @test_throws ArgumentError backtest(fc, df; horizon=30, initial=90)
        @test_throws "no complete backtest folds: data has 100 rows, but the first fold " *
                     "needs initial + horizon = 120. Provide more data" backtest(
            fc, df; horizon=30, initial=90)
        fc_deep = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(30)),
                             freq=Day(1))
        @test_throws ArgumentError backtest(fc_deep, df; horizon=5, initial=30)
        @test_throws "backtest initial=30 must exceed the feature set's minimum history " *
                     "(30 rows) so the first training window has at least one usable " *
                     "row. Pass initial=31 or more, or reduce lags/windows." backtest(
            fc_deep, df; horizon=5, initial=30)
        fc_d = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Direct(3),
                          freq=Day(1))
        @test_throws ArgumentError backtest(fc_d, df; horizon=5, initial=80)
        @test_throws "backtest horizon=5 exceeds the Direct strategy's max_horizon=3. " *
                     "Use Direct(5) or reduce horizon." backtest(fc_d, df; horizon=5,
                                                                initial=80)
        res = backtest(fc_d, df; horizon=3, initial=90, metrics=(mae,))
        @test maximum(res.scores.fold) == 3   # origins 90, 93, 96
    end

    @testset "metrics must be a tuple or vector of functions" begin
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fcm = Forecaster(m; features=fs, freq=Day(1))
            for bad in (mae, :mae, "mae", (:mae,), ("mae", "rmse"), Dict(:mae => mae),
                        [mae, :rmse])
                @test_throws ArgumentError backtest(fcm, df; horizon=5, initial=80,
                                                    metrics=bad)
                @test_throws "got $(repr(bad)). Pass e.g. metrics=(mae, rmse)" backtest(
                    fcm, df; horizon=5, initial=80, metrics=bad)
            end
            bt(ms) = backtest(fcm, df; horizon=5, initial=80, metrics=ms).scores
            for (it, ref) in ((Set([mae]), (mae,)), ((f for f in (mae,)), (mae,)),
                              ((f for f in (mae, rmse)), (mae, rmse)))
                @test isequal(bt(it), bt(ref))
            end
            # A one-shot iterator must still score every fold.
            @test isequal(bt(Iterators.Stateful((mae, rmse))), bt((mae, rmse)))
        end
        r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=[mae, rmse])
        @test r.scores.metric == [:mae, :rmse, :mae, :rmse]
        r2 = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(a=mae,))
        @test r2.scores.value == r.scores.value[r.scores.metric .== :mae]
        r3 = backtest(fc, df; horizon=5, initial=80, step=10,
                      metrics=((a, b) -> mae(a, b),))
        @test r3.scores.value == r2.scores.value
    end

    @testset "metrics must have distinct names" begin
        mkm(k) = (a, b) -> k * mae(a, b)
        msg = "backtest metrics must have distinct names, but 2 metrics are named " *
              ":mae, so summarize would merge their per-fold rows into one wrong " *
              "mean. Remove the repeated metric, or name each one with a NamedTuple " *
              "key (closures from one factory share a name), e.g. " *
              "metrics=(a=mae, b=Mase(12))."
        @test_throws ArgumentError backtest(fc, df; horizon=5, initial=80,
                                            metrics=(mae, mae))
        @test_throws msg backtest(fc, df; horizon=5, initial=80, metrics=(mae, mae))
        @test_throws msg backtest(fc, df; horizon=5, initial=80, metrics=[mae, rmse, mae])
        name = Symbol(mkm(1))
        @test_throws ArgumentError backtest(fc, df; horizon=5, initial=80,
                                            metrics=(mkm(1), mkm(2)))
        @test_throws "2 metrics are named $(repr(name))" backtest(
            fc, df; horizon=5, initial=80, metrics=(mkm(1), mkm(2)))
        @test_throws "3 metrics are named :mae" backtest(
            fc, df; horizon=5, initial=80, metrics=(mae, rmse, mae, mae))
        @test_throws "2 metrics are named :mae" backtest(
            fc, df; horizon=5, windows=2, metrics=(mae, mae))
        r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(mae, rmse))
        @test r.scores.metric == [:mae, :rmse, :mae, :rmse]
        r2 = backtest(fc, df; horizon=5, initial=80, step=10,
                      metrics=((a, b) -> mae(a, b), (a, b) -> 2 * mae(a, b)))
        s2 = summarize(r2)
        @test s2.mean[1] * 2 ≈ s2.mean[2]
        @test length(unique(r2.scores.metric)) == 2
    end

    @testset "edge cases: horizon 1 and bad data" begin
        r1 = backtest(fc, df; horizon=1, initial=95, metrics=(mae,))
        @test r1.forecasts.step == ones(Int, 5)
        @test r1.forecasts.y_hat == y[95:99] && r1.forecasts.y == y[96:100]
        @test all(==(1.0), r1.scores.value)
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
            @test r.forecasts.y_hat == want
            for (data, msg) in bad
                @test_throws ArgumentError backtest(fcm, data; horizon=1, initial=5)
                @test_throws msg backtest(fcm, data; horizon=1, initial=5)
            end
        end
    end

    @testset "metric protocol: names, scaled metrics, custom metrics" begin
        yw = [i + 3 * sin(i) + (i % 7) for i in 1:n]
        dw = (ds=t, y=yw)
        models = (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
        fcm(m; kw...) = Forecaster(m; features=fs, strategy=Recursive(), freq=Day(1), kw...)
        col(r, name, k) = only(r.scores.value[(r.scores.fold .== k) .&
                                               (r.scores.metric .== name)])

        @testset "a NamedTuple key names the metric" begin
            r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(a=mae,))
            @test r.scores.metric == [:a, :a]
            @test r.scores.value == backtest(fc, df; horizon=5, initial=80, step=10,
                                              metrics=(mae,)).scores.value
            r2 = backtest(fc, df; horizon=5, initial=80, step=10,
                          metrics=(first=mae, second=mae))
            @test r2.scores.metric == [:first, :second, :first, :second]
            @test r2.scores.value[1:2:end] == r2.scores.value[2:2:end]
            # Two closures from one factory run once keys name them.
            mkm(k) = (a, b) -> k * mae(a, b)
            r3 = backtest(fc, df; horizon=5, initial=80, step=10,
                          metrics=(small=mkm(1), large=mkm(2)))
            @test unique(r3.scores.metric) == [:small, :large]
            @test 2 .* r3.scores.value[r3.scores.metric .== :small] ≈
                  r3.scores.value[r3.scores.metric .== :large]
            @test_throws "2 metrics are named" backtest(
                fc, df; horizon=5, initial=80, metrics=(mkm(1), mkm(2)))
        end

        @testset "Mase(m) and Rmsse(m) are scored on each fold's window" begin
            for m in models
                fcr = fcm(m)
                res = backtest(fcr, dw; horizon=5, initial=60, step=10,
                               metrics=(mae, Mase(12), Rmsse(7), Mase()))
                @test unique(res.scores.metric) == [:mae, :mase_12, :rmsse_7, :mase]
                for k in sort(unique(res.forecasts.fold))
                    sel = res.forecasts.fold .== k
                    o = findfirst(==(res.forecasts.origin[findfirst(sel)]), t)
                    yk, ŷk = res.forecasts.y[sel], res.forecasts.y_hat[sel]
                    @test col(res, :mase_12, k) == mase(yk, ŷk; y_train=yw[1:o], m=12)
                    @test col(res, :rmsse_7, k) == rmsse(yk, ŷk; y_train=yw[1:o], m=7)
                    @test col(res, :mase, k) == mase(yk, ŷk; y_train=yw[1:o])
                end
                s = summarize(res)
                for name in (:mase_12, :rmsse_7, :mase)
                    vals = res.scores.value[res.scores.metric .== name]
                    @test only(s.mean[s.metric .== name]) == mean(vals)
                end
                # The object and the function give the same bits.
                ra = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(mase,))
                rb = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(Mase(),))
                @test isequal(ra.scores, rb.scores)
                rc = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(rmsse,))
                rd = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(Rmsse(),))
                @test isequal(rc.scores, rd.scores)
            end
        end

        @testset "custom metrics of each kind" begin
            for m in models
                fcr = fcm(m)
                ns = TestModels.NaiveScaled(2)
                f = (a, b) -> mae(a, b)
                res = backtest(fcr, dw; horizon=5, initial=60, step=10,
                               metrics=(f, ns, Mase(12)))
                @test res.scores.metric[1:3] == [Symbol(f), :naive_scaled_2, :mase_12]
                origins = unique(res.scores.origin)
                os = [findfirst(==(o), t) for o in origins]
                @test ns.seen == [yw[1:o] for o in os]
                for (k, o) in enumerate(os)
                    sel = res.forecasts.fold .== k
                    @test col(res, :naive_scaled_2, k) == mae(
                        res.forecasts.y[sel], res.forecasts.y_hat[sel]) /
                        mae(yw[2:o], yw[1:(o - 1)])
                end
                # A key beats metric_name.
                r2 = backtest(fcr, dw; horizon=5, initial=60, step=10,
                              metrics=(TestModels.NaiveScaled(3),))
                @test unique(r2.scores.metric) == [:naive_scaled_3]
                r3 = backtest(fcr, dw; horizon=5, initial=60, step=10,
                              metrics=(own=TestModels.NaiveScaled(3),))
                @test unique(r3.scores.metric) == [:own]
                @test r3.scores.value == r2.scores.value
            end
        end

        @testset "a row-dropping transform leaves the scaled window raw" begin
            for tf in (Difference(1), LogRatio(3))
                ns = TestModels.NaiveScaled()
                res = backtest(fcm(TestModels.LinAR(1.0, 0.0); transform=tf), dw;
                               horizon=5, initial=60, step=10, metrics=(ns,))
                os = [findfirst(==(o), t) for o in unique(res.forecasts.origin)]
                @test ns.seen == [yw[1:o] for o in os]
            end
        end

        @testset "horizon 1 and windows" begin
            for m in models
                r = backtest(fcm(m), dw; horizon=1, windows=3,
                             metrics=(Mase(12), Rmsse(7)))
                @test unique(r.scores.metric) == [:mase_12, :rmsse_7]
                @test all(isfinite, r.scores.value)
                @test length(r.scores.value) == 2 * 3
            end
        end

        @testset "protocol errors are ArgumentErrors" begin
            for m in models
                fcr = fcm(m)
                for (bad, msg) in (
                        (Mase,) => "got the type Mase; pass Mase() or Mase(12)",
                        (a=Rmsse,) => "got the type Rmsse; pass Rmsse() or Rmsse(12)",
                        (TestModels.badtrait,) => "metric_inputs returned true, " *
                            "which is not Point(), Scaled() or Samples().",
                        (a=:b => mae,) => "not with a Pair.",
                        (TestModels.BadName(),) => "which is not a Symbol.")
                    @test_throws ArgumentError backtest(
                        fcr, dw; horizon=5, initial=60, metrics=bad)
                    @test_throws msg backtest(fcr, dw; horizon=5, initial=60, metrics=bad)
                end
            end
            @test_throws "got the type Mase" backtest(
                fc, df; horizon=5, initial=60, metrics=[Mase])
        end

        @testset "a single series has no series prefix on a metric error" begin
            err = try backtest(fc, df; horizon=5, initial=80, metrics=(Mase(80),))
            catch e; e end
            @test err isa ArgumentError
            @test err.msg == "mase needs length(y_train) > m; got length(y_train)=80 " *
                             "with m=80. Pass a longer y_train or a smaller m."
        end
    end

    @testset "BacktestResult and summarize" begin
        yw = [i + 3 * sin(i) + (i % 7) for i in 1:n]
        dw = (ds=t, y=yw)
        scores_value(r) = r.scores.value

        @testset "forecasts, scores without overall rows, and the summary" begin
            r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(mae, rmse))
            @test r isa BacktestResult{<:NamedTuple,<:NamedTuple}
            @test keys(r.forecasts) == (:fold, :origin, :step, :ds, :y, :y_hat)
            @test keys(r.scores) == (:fold, :origin, :metric, :value)
            @test r.scores.fold == [1, 1, 2, 2]
            @test r.scores.metric == [:mae, :rmse, :mae, :rmse]
            @test !hasproperty(r, :folds) && !hasproperty(r, :metrics)
            s = summarize(r)
            @test keys(s) == (:metric, :mean, :std, :nfolds)
            @test s.metric isa Vector{Symbol} && s.mean isa Vector{Float64}
            @test s.std isa Vector{Float64} && s.nfolds isa Vector{Int}
            @test s.metric == [:mae, :rmse]
            @test s.mean ≈ [3.0, sqrt(11)]
            @test s.nfolds == [2, 2] && allequal(s.nfolds)
            @test s.std ≈ [0.0, 0.0] atol=1e-12
            @test Tables.istable(r.forecasts) && Tables.istable(r.scores)
            @test Tables.istable(s)
            # Two metrics over two folds, no overall rows.
            @test 0 ∉ r.scores.fold
            @test length(r.scores.fold) == length(unique(r.forecasts.fold)) * 2
            sh = sprint(show, MIME"text/plain"(), r)
            @test startswith(sh, "BacktestResult: 2 folds, horizon 5\n  mae      = 3.0")
            @test endswith(sh, "(see .forecasts and .scores; summarize(r) for " *
                               "mean and std)")
            @test !occursin(".folds", sh) && !occursin(".metrics", sh)
        end

        @testset "mean and std follow the per-fold scores" begin
            r = backtest(fc, dw; horizon=5, initial=60, step=10, metrics=(rmse, mae))
            s = summarize(r)
            @test s.metric == [:rmse, :mae]
            @test s.nfolds == [length(unique(r.forecasts.fold)) for _ in 1:2]
            for (i, name) in enumerate(s.metric)
                v = r.scores.value[r.scores.metric .== name]
                @test length(unique(v)) > 1
                @test s.mean[i] ≈ sum(v) / length(v)
                dev = sum(abs2, v .- sum(v) / length(v))
                @test s.std[i] ≈ sqrt(dev / (length(v) - 1))
                @test s.std[i] > 0
            end
            # The order of the metrics argument is the row order.
            @test summarize(backtest(fc, dw; horizon=5, initial=60, step=10,
                                     metrics=(mae, rmse))).metric == [:mae, :rmse]
            k = backtest(fc, dw; horizon=5, initial=60, step=10,
                         metrics=(zz=mae, aa=Mase(12), mm=rmse))
            @test summarize(k).metric == [:zz, :aa, :mm]
            @test summarize(backtest(fc, dw; horizon=5, initial=60,
                                     metrics=(Mase(12), rmse, mae))).metric ==
                  [:mase_12, :rmse, :mae]
        end

        @testset "summarize keeps the bits recorded on the base commit" begin
            # Bit patterns recorded on the base commit.
            noise(i) = sin(i^2 * 0.37) + 0.5 * cos(i * 1.7)
            hexes(x) = reinterpret(UInt64, x)
            d1 = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(119))
            sy = [50 + 0.1i + 5 * sin(2π * i / 7) + noise(i) for i in 1:120]
            fcs = Forecaster(TestModels.LinAR(0.9, 4.0);
                             features=FeatureSet(Lag(1), Lag(7), RollingMean(7)),
                             strategy=Recursive(), freq=Day(1))
            ms = (mae, rmse, smape, Mase(7), rmsse)
            s = summarize(backtest(fcs, (ds=d1, y=sy); horizon=5, windows=4, step=5,
                                   metrics=ms))
            @test s.metric == [:mae, :rmse, :smape, :mase_7, :rmsse]
            @test hexes(s.mean) == UInt64[0x4017a147f346cad2, 0x401ba4596f2dba7d,
                0x3fb9fa90d00485d6, 0x40172023752a04a3, 0x4000edc12d9174f9]
            @test hexes(s.std) == UInt64[0x400b65f9a0be5cc1, 0x400bf61909f046b8,
                0x3faeb5ada19091c9, 0x400a84dd1976c499, 0x3ff107994eafb87f]
            @test s.nfolds == fill(4, 5)
            pid = [fill("a", 120); fill("b", 100); fill("c", 90)]
            pds = [d1; d1[11:110]; d1[1:90]]
            py = [sy; [20 + 0.2i + 3 * sin(2π * i / 7) + noise(2i) for i in 1:100];
                  [80 + 0.05i + 8 * sin(2π * i / 7) + noise(3i) for i in 1:90]]
            fcp = Forecaster(TestModels.MeanModel();
                             features=FeatureSet(Lag(1), Lag(7), RollingMean(7)),
                             strategy=Direct(5), freq=Day(1), id=:id, time=:ds,
                             target=:y)
            sp = summarize(backtest(fcp, (id=pid, ds=pds, y=py); horizon=5, initial=60,
                                    step=7, metrics=ms))
            # Bits as Pkg.test runs it, with bounds checks forced.
            @test hexes(sp.mean) == UInt64[0x402d56b4ccec28c0, 0x40314cf5a451b607,
                0x3fd18daed3bbd805, 0x4029b2b78243bea0, 0x401580c6af127496]
            @test hexes(sp.std) == UInt64[0x401221d64b7f4094, 0x4014abe3f45d3da8,
                0x3fb4148caecce8f8, 0x40136f34cf05702a, 0x3ff2caa176a27a60]
            @test sp.nfolds == fill(8, 5)
        end

        @testset "one fold: std is NaN" begin
            r = backtest(fc, dw; horizon=5, windows=1, metrics=(mae, rmse))
            s = summarize(r)
            @test s.nfolds == [1, 1]
            @test all(isnan, s.std) && s.std isa Vector{Float64}
            @test s.mean == r.scores.value[1:2]
            @test occursin("1 fold, horizon 5", sprint(show, MIME"text/plain"(), r))
        end

        @testset "horizon 1 counts every fold" begin
            r = backtest(fc, df; horizon=1, initial=95, metrics=(mae, rmse))
            s = summarize(r)
            @test s.nfolds == [5, 5] && s.mean == [1.0, 1.0] && s.std == [0.0, 0.0]
            @test length(r.scores.fold) == 10
        end

        @testset "metrics=() gives empty tables" begin
            r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=())
            @test r.scores == (fold=Int[], origin=Union{Missing,Date}[],
                               metric=Symbol[], value=Float64[])
            @test typeof(r.scores) == typeof((fold=Int[], origin=Union{Missing,Date}[],
                                              metric=Symbol[], value=Float64[]))
            s = summarize(r)
            @test s == (metric=Symbol[], mean=Float64[], std=Float64[], nfolds=Int[])
            @test typeof(s) == @NamedTuple{metric::Vector{Symbol}, mean::Vector{Float64},
                                           std::Vector{Float64}, nfolds::Vector{Int}}
            sh = sprint(show, MIME"text/plain"(), r)
            @test startswith(sh, "BacktestResult: 2 folds, horizon 5\n  (see .forecasts")
        end

        @testset "NaN and Inf scores propagate" begin
            nanfirst(y, ŷ) = ŷ[1] == 80.0 ? NaN : 1.0
            r = backtest(fc, df; horizon=5, initial=80, step=10,
                         metrics=(bad=nanfirst, ok=mae))
            s = summarize(r)
            @test s.metric == [:bad, :ok] && s.nfolds == [2, 2]
            @test isnan(s.mean[1]) && isnan(s.std[1])
            @test s.mean[2] ≈ 3.0
            # A constant window has a zero scale, so Inf.
            flat = (ds=t, y=[fill(5.0, 80); 1.0 .* (81:n)])
            fi = Forecaster(TestModels.LinAR(1.0, 1.0); features=fs,
                            strategy=Recursive(), freq=Day(1))
            ri = @test_logs (:warn, r"returning Inf") match_mode=:any backtest(
                fi, flat; horizon=5, initial=80, step=10, metrics=(Mase(),))
            @test isinf(ri.scores.value[1]) && isfinite(ri.scores.value[2])
            si = summarize(ri)
            @test only(si.mean) == Inf && only(si.nfolds) == 2
        end

        @testset "summarize agrees with the scores for every model and strategy" begin
            models = (TestModels.LinAR(1.0, 0.0), TestModels.MeanModel(),
                      EvoTreeRegressor(nrounds=5),
                      DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            for m in models, strat in (Recursive(), Direct(5))
                fcr = Forecaster(m; features=fs, strategy=strat, freq=Day(1))
                r = backtest(fcr, dw; horizon=5, initial=60, step=10,
                             metrics=(mae, Mase(7)))
                s = summarize(r)
                nf = length(unique(r.forecasts.fold))
                @test s.nfolds == [nf, nf] && length(r.scores.fold) == 2nf
                for (i, name) in enumerate(s.metric)
                    @test s.mean[i] ≈ sum(r.scores.value[r.scores.metric .== name]) / nf
                end
            end
        end

        @testset "types are concrete" begin
            r = backtest(fc, df; horizon=5, initial=80, step=10)
            @test typeof(r) <: BacktestResult{<:NamedTuple,<:NamedTuple}
            @test isconcretetype(fieldtype(typeof(r), :scores))
            @test @inferred(summarize(r)) isa NamedTuple
            @test @inferred(scores_value(r)) isa Vector{Float64}
            @test eltype(r.scores.origin) == Union{Missing,Date}
        end

        @testset "a wrong argument is a MethodError" begin
            @test_throws MethodError summarize(nothing)
            @test_throws MethodError summarize((fold=[1], value=[1.0]))
        end
    end
end
