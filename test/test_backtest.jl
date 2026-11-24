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
        @test rw.folds == ri.folds && isequal(rw.metrics, ri.metrics)
        @test unique(rw.folds.origin) == t[[75, 85, 95]] && last(rw.folds.ds) == t[end]
        @test unique(backtest(fc, df; horizon=5, windows=1).folds.fold) == [1]
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
        @test keys(res.folds) == (:fold, :origin, :step, :ds, :y, :y_hat)
        @test res.folds.fold == [fill(1, 5); fill(2, 5)]
        @test length(res.folds.step) == 2 * 5
        @test res.folds.step == [1:5; 1:5]
        @test res.folds.origin == [fill(t[80], 5); fill(t[90], 5)]
        @test res.folds.ds == [t[81:85]; t[91:95]]
        @test res.folds.y == [y[81:85]; y[91:95]]
        @test res.folds.y_hat == [fill(80.0, 5); fill(90.0, 5)]
        # per-fold metrics: |y - origin| averaged: mean(1,2,3,4,5) = 3
        m = res.metrics
        perfold = m.fold .> 0
        @test count(perfold) == 4   # 2 folds × 2 metrics
        @test all(m.value[perfold .& (m.metric .== :mae)] .≈ 3.0)
        overall = m.fold .== 0
        @test count(overall) == 2
        @test all(ismissing, m.origin[overall])
        @test only(m.value[overall .& (m.metric .== :mae)]) ≈ 3.0
        @test only(m.value[overall .& (m.metric .== :rmse)]) ≈ sqrt(mean([1, 4, 9, 16, 25]))
        @test Tables.istable(res.folds) && Tables.istable(res.metrics)
        s = sprint(show, MIME"text/plain"(), res)
        @test occursin("2 folds", s) && occursin("mae", s)
    end

    @testset "exogenous columns are sliced from held-out data" begin
        dfe = (ds=t, y=y, promo=Float64.(1:n) .* 10)
        fce = Forecaster(TestModels.EchoColumn(:promo);
                         features=FeatureSet(Lag(1), Exogenous(:promo)),
                         strategy=Recursive(), freq=Day(1))
        res = backtest(fce, dfe; horizon=3, initial=90, step=100, metrics=(mae,))
        @test res.folds.y_hat == dfe.promo[91:93]
    end

    @testset "rmsse is scaled by each fold's training window" begin
        yw = [i + 3 * sin(i) + (i % 7) for i in 1:n]
        dw = (ds=t, y=yw)
        for m in (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fcr = Forecaster(m; features=fs, strategy=Recursive(), freq=Day(1))
            res = backtest(fcr, dw; horizon=5, initial=60, step=10,
                           metrics=(rmse, rmsse))
            mt = res.metrics
            folds = sort(unique(res.folds.fold))
            for k in folds
                sel = res.folds.fold .== k
                o = findfirst(==(res.folds.origin[findfirst(sel)]), t)
                tr = yw[1:o]
                sc = mean(abs2(tr[i] - tr[i - 1]) for i in 2:o)
                got = only(mt.value[(mt.fold .== k) .& (mt.metric .== :rmsse)])
                @test got ≈ rmse(res.folds.y[sel], res.folds.y_hat[sel]) / sqrt(sc)
            end
            vals = mt.value[(mt.fold .> 0) .& (mt.metric .== :rmsse)]
            @test only(mt.value[(mt.fold .== 0) .& (mt.metric .== :rmsse)]) ≈ mean(vals)
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
        @test maximum(res.metrics.fold) == 3   # origins 90, 93, 96
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
            bt(ms) = backtest(fcm, df; horizon=5, initial=80, metrics=ms).metrics
            for (it, ref) in ((Set([mae]), (mae,)), ((f for f in (mae,)), (mae,)),
                              ((f for f in (mae, rmse)), (mae, rmse)))
                @test isequal(bt(it), bt(ref))
            end
            # A one-shot iterator must still score every fold.
            @test isequal(bt(Iterators.Stateful((mae, rmse))), bt((mae, rmse)))
        end
        r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=[mae, rmse])
        @test r.metrics.metric == [:mae, :rmse, :mae, :rmse, :mae, :rmse]
        r2 = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(a=mae,))
        @test r2.metrics.value == r.metrics.value[r.metrics.metric .== :mae]
        r3 = backtest(fc, df; horizon=5, initial=80, step=10,
                      metrics=((a, b) -> mae(a, b),))
        @test r3.metrics.value == r2.metrics.value
    end

    @testset "metrics must have distinct names" begin
        mkm(k) = (a, b) -> k * mae(a, b)
        msg = "backtest metrics must have distinct names, but 2 metrics are named " *
              ":mae, so their per-fold rows would be merged into one wrong overall " *
              "score. Remove the repeated metric, or name each one with a NamedTuple " *
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
        @test r.metrics.metric == [:mae, :rmse, :mae, :rmse, :mae, :rmse]
        r2 = backtest(fc, df; horizon=5, initial=80, step=10,
                      metrics=((a, b) -> mae(a, b), (a, b) -> 2 * mae(a, b)))
        @test r2.metrics.value[end - 1] * 2 ≈ r2.metrics.value[end]
        @test length(unique(r2.metrics.metric)) == 2
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

    @testset "metric protocol: names, scaled metrics, custom metrics" begin
        yw = [i + 3 * sin(i) + (i % 7) for i in 1:n]
        dw = (ds=t, y=yw)
        models = (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
        fcm(m; kw...) = Forecaster(m; features=fs, strategy=Recursive(), freq=Day(1), kw...)
        col(r, name, k) = only(r.metrics.value[(r.metrics.fold .== k) .&
                                               (r.metrics.metric .== name)])

        @testset "a NamedTuple key names the metric" begin
            r = backtest(fc, df; horizon=5, initial=80, step=10, metrics=(a=mae,))
            @test r.metrics.metric == [:a, :a, :a]
            @test r.metrics.value == backtest(fc, df; horizon=5, initial=80, step=10,
                                              metrics=(mae,)).metrics.value
            r2 = backtest(fc, df; horizon=5, initial=80, step=10,
                          metrics=(first=mae, second=mae))
            @test r2.metrics.metric == [:first, :second, :first, :second, :first, :second]
            @test r2.metrics.value[1:2:end] == r2.metrics.value[2:2:end]
            # Two closures from one factory run once keys name them.
            mkm(k) = (a, b) -> k * mae(a, b)
            r3 = backtest(fc, df; horizon=5, initial=80, step=10,
                          metrics=(small=mkm(1), large=mkm(2)))
            @test unique(r3.metrics.metric) == [:small, :large]
            @test 2 .* r3.metrics.value[r3.metrics.metric .== :small] ≈
                  r3.metrics.value[r3.metrics.metric .== :large]
            @test_throws "2 metrics are named" backtest(
                fc, df; horizon=5, initial=80, metrics=(mkm(1), mkm(2)))
        end

        @testset "Mase(m) and Rmsse(m) are scored on each fold's window" begin
            for m in models
                fcr = fcm(m)
                res = backtest(fcr, dw; horizon=5, initial=60, step=10,
                               metrics=(mae, Mase(12), Rmsse(7), Mase()))
                @test unique(res.metrics.metric) == [:mae, :mase_12, :rmsse_7, :mase]
                for k in sort(unique(res.folds.fold))
                    sel = res.folds.fold .== k
                    o = findfirst(==(res.folds.origin[findfirst(sel)]), t)
                    yk, ŷk = res.folds.y[sel], res.folds.y_hat[sel]
                    @test col(res, :mase_12, k) == mase(yk, ŷk; y_train=yw[1:o], m=12)
                    @test col(res, :rmsse_7, k) == rmsse(yk, ŷk; y_train=yw[1:o], m=7)
                    @test col(res, :mase, k) == mase(yk, ŷk; y_train=yw[1:o])
                end
                for name in (:mase_12, :rmsse_7, :mase)
                    vals = res.metrics.value[(res.metrics.fold .> 0) .&
                                             (res.metrics.metric .== name)]
                    @test col(res, name, 0) == mean(vals)
                end
                # The object and the function give the same bits.
                ra = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(mase,))
                rb = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(Mase(),))
                @test isequal(ra.metrics, rb.metrics)
                rc = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(rmsse,))
                rd = backtest(fcr, dw; horizon=5, initial=60, step=10, metrics=(Rmsse(),))
                @test isequal(rc.metrics, rd.metrics)
            end
        end

        @testset "custom metrics of each kind" begin
            for m in models
                fcr = fcm(m)
                ns = TestModels.NaiveScaled(2)
                f = (a, b) -> mae(a, b)
                res = backtest(fcr, dw; horizon=5, initial=60, step=10,
                               metrics=(f, ns, Mase(12)))
                @test res.metrics.metric[1:3] == [Symbol(f), :naive_scaled_2, :mase_12]
                origins = unique(res.metrics.origin[res.metrics.fold .> 0])
                os = [findfirst(==(o), t) for o in origins]
                @test ns.seen == [yw[1:o] for o in os]
                for (k, o) in enumerate(os)
                    sel = res.folds.fold .== k
                    @test col(res, :naive_scaled_2, k) == mae(
                        res.folds.y[sel], res.folds.y_hat[sel]) /
                        mae(yw[2:o], yw[1:(o - 1)])
                end
                # A key beats metric_name.
                r2 = backtest(fcr, dw; horizon=5, initial=60, step=10,
                              metrics=(TestModels.NaiveScaled(3),))
                @test unique(r2.metrics.metric) == [:naive_scaled_3]
                r3 = backtest(fcr, dw; horizon=5, initial=60, step=10,
                              metrics=(own=TestModels.NaiveScaled(3),))
                @test unique(r3.metrics.metric) == [:own]
                @test r3.metrics.value == r2.metrics.value
            end
        end

        @testset "a row-dropping transform leaves the scaled window raw" begin
            for tf in (Difference(1), LogRatio(3))
                ns = TestModels.NaiveScaled()
                res = backtest(fcm(TestModels.LinAR(1.0, 0.0); transform=tf), dw;
                               horizon=5, initial=60, step=10, metrics=(ns,))
                os = [findfirst(==(o), t) for o in unique(res.folds.origin)]
                @test ns.seen == [yw[1:o] for o in os]
            end
        end

        @testset "horizon 1 and windows" begin
            for m in models
                r = backtest(fcm(m), dw; horizon=1, windows=3,
                             metrics=(Mase(12), Rmsse(7)))
                @test unique(r.metrics.metric) == [:mase_12, :rmsse_7]
                @test all(isfinite, r.metrics.value)
                @test length(r.metrics.value) == 2 * 4
            end
        end

        @testset "protocol errors are ArgumentErrors" begin
            for m in models
                fcr = fcm(m)
                for (bad, msg) in (
                        (Mase,) => "got the type Mase; pass Mase() or Mase(12)",
                        (a=Rmsse,) => "got the type Rmsse; pass Rmsse() or Rmsse(12)",
                        (TestModels.badtrait,) => "metric_inputs returned true, which " *
                                                  "is not Point() or Scaled().",
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
end
