import MachineLearningForecast as M

@testset "dropped_rows and minhistory(fc)" begin
    t0 = Date(2022, 1, 1)
    series(n; level=100.0) = level .+ 2.0 .* (1:n) .+ 3.0 .* sin.(2π .* (1:n) ./ 7) .+
                             randn(StableRNG(n), n)
    # Each pair is `id => rows`; all series start on one day.
    function panel(lens::Pair...)
        ids = reduce(vcat, [fill(id, n) for (id, n) in lens])
        ds = reduce(vcat, [collect(t0:Day(1):(t0 + Day(n - 1))) for (_, n) in lens])
        y = reduce(vcat, [series(n; level=10.0 * k) for (k, (_, n)) in enumerate(lens)])
        return (unique_id=ids, ds=ds, y=y)
    end
    single(n) = (ds=collect(t0:Day(1):(t0 + Day(n - 1))), y=series(n))
    chain = TransformChain(Difference(12), LastScale(3))
    spec(model; transform=chain, strategy=Recursive(), features=FeatureSet(Lag(1)),
         id=:unique_id) = Forecaster(model; features, strategy, transform, freq=Day(1),
                                     id)
    fc_single(model; kw...) = spec(model; id=nothing, kw...)
    skipmsg(n, id="short") =
        "skipping 1 series with fewer than 14 rows, which is the minimum the feature " *
        "set and transform need (1 history rows plus 12 dropped by the transform): " *
        "unique_id=\"$id\" ($n rows)"
    models() = (EvoTreeRegressor(nrounds=8, seed=1),
                DecisionTreeRegressor(max_depth=3, rng=StableRNG(1)))

    @testset "dropped_rows of every built-in" begin
        for t in (NoTransform(), MeanScale(), LastScale(3), Deseasonalize(12),
                  TestModels.VaryingState())
            @test M.dropped_rows(t) == 0
        end
        @test M.dropped_rows(Difference(1)) == 1
        @test M.dropped_rows(Difference(12)) == 12
        @test M.dropped_rows(LogRatio(5)) == 5
        @test M.dropped_rows(chain) == 12
        @test M.dropped_rows(TransformChain(Difference(1), Difference(12))) == 13
        @test M.dropped_rows(TransformChain(Difference(1), LogRatio(5), LastScale(2))) == 6
        @test M.dropped_rows(TransformChain(Difference(7))) == 7
        nested = TransformChain(TransformChain(Difference(2), Difference(3)), LogRatio(4))
        @test M.dropped_rows(nested) == 9
        @test M.dropped_rows(TransformChain(nested, nested)) == 18
        @test M.dropped_rows(TestModels.FussyDrop(7)) == 7
        @test M.dropped_rows(TestModels.QuietDrop(7)) == 0
    end

    @testset "the declared count is the number of rows fit_transform drops" begin
        y = series(60)
        for t in (NoTransform(), MeanScale(), LastScale(3), Deseasonalize(7),
                  Difference(1), Difference(12), LogRatio(5), chain,
                  TransformChain(Difference(1), Difference(12)),
                  TransformChain(LogRatio(3), Difference(2)),
                  TransformChain(LogRatio(4),
                                 TransformChain(Difference(2), Difference(3))))
            z, _ = M.fit_transform(t, y)
            @test length(z) == length(y) - M.dropped_rows(t)
        end
    end

    @testset "minhistory(fc) adds the transform's rows to the feature set's" begin
        fs = FeatureSet(Lag.(1:3)...)
        @test M.minhistory(spec(TestModels.LinAR(1.0, 0.0); features=fs)) == 3 + 12
        @test M.minhistory(spec(TestModels.LinAR(1.0, 0.0); features=fs,
                                transform=NoTransform())) == 3
        @test M.minhistory(spec(TestModels.LinAR(1.0, 0.0); features=fs,
                                transform=TestModels.QuietDrop(5))) == 3
        @test M.minhistory(spec(TestModels.LinAR(1.0, 0.0); features=fs,
                                transform=TestModels.FussyDrop(5))) == 3 + 5
        @test M.minhistory(spec(TestModels.LinAR(1.0, 0.0); features=FeatureSet(
            Calendar(:dayofweek)), transform=Difference(4))) == 4
    end

    @testset "a series too short for the transform is skipped, not fatal" begin
        for model in models(), strategy in (Recursive(), Direct(3)), n in (10, 12, 13)
            fc = spec(model; strategy)
            p = panel("long" => 60, "short" => n)
            f = @test_logs (:warn, skipmsg(n)) fit(fc, p)
            solo = fit(fc, panel("long" => 60))
            @test nseries(f) == 1 && [s.id for s in f.series] == ["long"]
            @test forecast(f, 3) == forecast(solo, 3)
            @test f.series[1].transform_state == solo.series[1].transform_state
        end
        # The short series comes first; Difference(12) alone.
        fc = spec(TestModels.LinAR(1.0, 0.0); transform=Difference(12))
        f = @test_logs (:warn, "skipping 1 series with fewer than 14 rows, which is " *
                        "the minimum the feature set and transform need (1 history " *
                        "rows plus 12 dropped by the transform): unique_id=\"short\" " *
                        "(10 rows)") fit(fc, panel("short" => 10, "long" => 60))
        @test [s.id for s in f.series] == ["long"]
    end

    @testset "the boundary: minhistory(fc) + 1 rows fit, minhistory(fc) rows do not" begin
        for strategy in (Recursive(), Direct(1)), ids in (("long", "edge"), (1, 2))
            fc = spec(TestModels.LinAR(1.0, 0.0); strategy)
            @test M.minhistory(fc) == 13
            f = fit(fc, panel(ids[1] => 60, ids[2] => 14))
            @test nseries(f) == 2
            @test length(forecast(f, 1).y_hat) == 2
            f = @test_logs (:warn, r"fewer than 14 rows.*\(13 rows\)") fit(
                fc, panel(ids[1] => 60, ids[2] => 13))
            @test nseries(f) == 1
            @test fit(fc_single(TestModels.LinAR(1.0, 0.0); strategy), single(14)) isa
                  FittedForecaster
            @test_throws ArgumentError fit(
                fc_single(TestModels.LinAR(1.0, 0.0); strategy), single(13))
        end
        # Integer ids are named as they are.
        fc = spec(TestModels.LinAR(1.0, 0.0))
        @test_logs (:warn, "skipping 1 series with fewer than 14 rows, which is the " *
                    "minimum the feature set and transform need (1 history rows plus " *
                    "12 dropped by the transform): unique_id=2 (3 rows)") fit(
                        fc, panel(1 => 60, 2 => 3))
    end

    @testset "skip warning counts the rows the user gave" begin
        fc = spec(TestModels.LinAR(1.0, 0.0))
        p = panel("long" => 60, "a" => 5, "b" => 13, "c" => 1)
        f = @test_logs (:warn, "skipping 3 series with fewer than 14 rows, which is " *
                        "the minimum the feature set and transform need (1 history " *
                        "rows plus 12 dropped by the transform): unique_id=\"a\" " *
                        "(5 rows), unique_id=\"b\" (13 rows), unique_id=\"c\" " *
                        "(1 rows)") fit(fc, p)
        @test nseries(f) == 1
        # Without a transform the text is the old one.
        fc0 = spec(TestModels.LinAR(1.0, 0.0); transform=NoTransform())
        @test_logs (:warn, "skipping 1 series with fewer than 2 rows, which is the " *
                    "minimum the feature set needs: unique_id=\"b\" (1 rows)") fit(
                        fc0, panel("long" => 60, "b" => 1))
    end

    @testset "no series long enough" begin
        fc = spec(TestModels.LinAR(1.0, 0.0))
        @test_throws("no series has enough rows to train on: the feature set needs 1 " *
                     "history rows and the transform drops 12, so a series needs at " *
                     "least 14 rows, and the longest series has 13. Provide longer " *
                     "series or reduce lags/windows or the transform's lag/window.",
                     fit(fc, panel("a" => 13, "b" => 4)))
        fc0 = spec(TestModels.LinAR(1.0, 0.0); transform=NoTransform())
        @test_throws("no series has enough rows to train on: the feature set needs 1 " *
                     "history rows before the first usable row, and the longest series " *
                     "has 1. Provide longer series or reduce lags/windows.",
                     fit(fc0, panel("a" => 1, "b" => 1)))
    end

    @testset "a single series shorter than the transform and features need" begin
        fc = fc_single(TestModels.LinAR(1.0, 0.0))
        for n in (12, 13)
            @test_throws("not enough data: the feature set needs 1 history rows and " *
                         "the transform drops 12, so a series needs at least 14 rows, " *
                         "but the data has only $n rows. Provide at least 14 rows or " *
                         "reduce lags/windows or the transform's lag/window.",
                         fit(fc, single(n)))
        end
        @test fit(fc, single(14)) isa FittedForecaster
        # One row would leave LastScale an empty mean.
        @test_throws "but the data has only 1 rows" fit(fc, single(1))
        # An empty series is still reported as empty.
        @test_throws "time column :ds is empty. Provide at least one row." fit(
            fc, (ds=Date[], y=Float64[]))
        # Without a transform the text is the old one.
        @test_throws("not enough data: the feature set needs 1 history rows before " *
                     "the first usable training row, but the data has only 1 rows. " *
                     "Provide at least 2 rows or reduce lags/windows.",
                     fit(fc_single(TestModels.LinAR(1.0, 0.0); transform=NoTransform()),
                         single(1)))
    end

    @testset "Direct names the rows the transform drops" begin
        fc = fc_single(TestModels.LinAR(1.0, 0.0); strategy=Direct(4))
        @test_throws("not enough data for Direct(4): after dropping the transform's " *
                     "12 rows and the feature set's 1 history rows, only 2 training " *
                     "rows remain, but the target must be shifted up to 3 steps. " *
                     "Provide at least 17 rows, reduce max_horizon, or reduce " *
                     "lags/windows or the transform's lag/window.", fit(fc, single(15)))
        @test_throws "only 3 training rows remain" fit(fc, single(16))
        # The panel text names the lever too.
        fcp = spec(TestModels.LinAR(1.0, 0.0); strategy=Direct(5))
        @test_throws("not enough data for Direct(5): no series has enough rows to " *
                     "train the step-4 model. Shorten max_horizon, provide longer " *
                     "series, or reduce lags/windows or the transform's lag/window.",
                     fit(fcp, panel("a" => 16, "b" => 15)))
        @test fit(fc, single(17)) isa FittedForecaster
        fc0 = fc_single(TestModels.LinAR(1.0, 0.0); strategy=Direct(4),
                        transform=NoTransform())
        @test_throws("after dropping the feature set's 1 history rows, only 2 training " *
                     "rows remain", fit(fc0, single(3)))
    end

    @testset "a declared transform is never fitted on a series that is skipped" begin
        fc = spec(TestModels.LinAR(1.0, 0.0); transform=TestModels.FussyDrop(3))
        # FussyDrop throws on short input, so the check is first.
        f = @test_logs (:warn, "skipping 2 series with fewer than 5 rows, which is " *
                        "the minimum the feature set and transform need (1 history " *
                        "rows plus 3 dropped by the transform): unique_id=\"a\" " *
                        "(3 rows), unique_id=\"b\" (4 rows)") fit(
                            fc, panel("long" => 30, "a" => 3, "b" => 4))
        @test nseries(f) == 1
        @test nseries(fit(fc, panel("long" => 30, "c" => 5))) == 2
        @test_throws("the transform drops 3, so a series needs at least 5 rows, but " *
                     "the data has only 3 rows", fit(
                         fc_single(TestModels.LinAR(1.0, 0.0);
                                   transform=TestModels.FussyDrop(3)), single(3)))
        @test fit(fc_single(TestModels.LinAR(1.0, 0.0);
                            transform=TestModels.FussyDrop(3)), single(5)) isa
              FittedForecaster
        # All series are short: this error, not FussyDrop's.
        @test_throws "no series has enough rows to train on" fit(
            fc, panel("a" => 2, "b" => 3))
    end

    @testset "an undeclared transform keeps the check after the transform" begin
        fc = spec(TestModels.LinAR(1.0, 0.0); transform=TestModels.QuietDrop(3))
        for strategy in (Recursive(), Direct(3))
            fcs = spec(TestModels.LinAR(1.0, 0.0); transform=TestModels.QuietDrop(3),
                       strategy)
            # Four rows leave one: not more than the feature set needs.
            f = @test_logs (:warn, "skipping 1 series with fewer than 2 rows, which " *
                            "is the minimum the feature set needs: unique_id=\"tiny\" " *
                            "(1 rows)") fit(fcs, panel("long" => 30, "tiny" => 4))
            @test [s.id for s in f.series] == ["long"]
            @test forecast(f, 3) == forecast(fit(fcs, panel("long" => 30)), 3)
            @test nseries(fit(fcs, panel("long" => 30, "ok" => 7))) == 2
            # A series shorter than the drop is skipped, not an error.
            f = @test_logs (:warn, "skipping 1 series with fewer than 2 rows, which " *
                            "is the minimum the feature set needs: unique_id=\"tiny\" " *
                            "(0 rows)") fit(fcs, panel("long" => 30, "tiny" => 2))
            @test nseries(f) == 1
        end
        @test_throws("no series has enough rows to train on: the feature set needs 1 " *
                     "history rows before the first usable row, and the longest " *
                     "series has 1. Provide longer series or reduce lags/windows.",
                     fit(fc, panel("a" => 4, "b" => 3)))
        # A single series fails after the transform, as before.
        @test_throws("the feature set needs 1 history rows before the first usable " *
                     "training row, but the data has only 1 rows",
                     fit(fc_single(TestModels.LinAR(1.0, 0.0);
                                   transform=TestModels.QuietDrop(3)), single(4)))
        @test fit(fc_single(TestModels.LinAR(1.0, 0.0);
                            transform=TestModels.QuietDrop(3)), single(5)) isa
              FittedForecaster
    end

    @testset "errors and warnings follow the order of the series" begin
        fc = spec(TestModels.LinAR(1.0, 0.0))
        bad(p, i, v) = (y = Vector{Union{Missing,Float64}}(p.y); y[i] = v;
                        merge(p, (y=y,)))
        nanmsg(row) = "target column :y contains the non-finite value NaN at row " *
                      "$row. Impute, drop, or clip non-finite targets before fitting " *
                      "— they propagate silently through lag features into every " *
                      "forecast."
        # A short series is validated like a long one.
        @test_throws("series unique_id=\"short\": " * nanmsg(62),
                     fit(fc, bad(panel("long" => 60, "short" => 5), 62, NaN)))
        # The first bad series in the table wins, short or not.
        p = bad(bad(panel("a" => 60, "b" => 5), 10, missing), 62, NaN)
        @test_throws("series unique_id=\"a\": target column :y contains missing " *
                     "values (first at row 10). Impute or drop missing targets " *
                     "before fitting.", fit(fc, p))
        p = bad(bad(panel("a" => 5, "b" => 60), 3, NaN), 10, missing)
        @test_throws "series unique_id=\"a\": " * nanmsg(3) fit(fc, p)
        # An earlier transform error beats a later NaN.
        fcl = spec(TestModels.LinAR(1.0, 0.0); transform=LogRatio(2),
                   features=FeatureSet(Lag(1)))
        p = bad(bad(panel("a" => 20, "b" => 20), 5, -1.0), 25, NaN)
        @test_throws "series unique_id=\"a\": LogRatio(2) takes logarithms" fit(fcl, p)
        # A single series reports its target before its length.
        @test_throws(nanmsg(1), fit(fc_single(TestModels.LinAR(1.0, 0.0)),
                                    (ds=[t0], y=[NaN])))
        @test_throws(nanmsg(1), fit(fc_single(TestModels.LinAR(1.0, 0.0);
                                              transform=NoTransform()),
                                    (ds=[t0], y=[NaN])))
        # Skips list in table order, pre- or post-transform.
        fcu = spec(TestModels.LinAR(1.0, 0.0); transform=TestModels.UnderDeclared(3, 1))
        @test M.minhistory(fcu) == 2
        @test_logs((:warn, "skipping 3 series with fewer than 3 rows, which is the " *
                    "minimum the feature set and transform need (1 history rows plus " *
                    "1 dropped by the transform): unique_id=\"a\" (2 rows), " *
                    "unique_id=\"b\" (1 rows), unique_id=\"c\" (1 rows)"),
                   fit(fcu, panel("long" => 30, "a" => 2, "b" => 4, "c" => 1)))
    end

    @testset "backtest validates initial and windows against the transform" begin
        fc = fc_single(TestModels.LinAR(1.0, 0.0))
        d = single(30)
        @test_throws("backtest initial=13 must exceed the feature set's minimum " *
                     "history plus the 12 rows the transform drops (13 rows) so the " *
                     "first training window has at least one usable row. Pass " *
                     "initial=14 or more, or reduce lags/windows or the transform's " *
                     "lag/window.", backtest(fc, d; horizon=3, initial=13))
        @test_throws("backtest initial=5 must exceed the feature set's minimum " *
                     "history plus the 12 rows", backtest(fc, d; horizon=3, initial=5))
        @test_throws("backtest windows=7 with step=3 and horizon=3 would train its " *
                     "first fold on 9 of the 30 rows, but the feature set and " *
                     "transform need more than 13. Reduce windows, step or horizon or " *
                     "the transform's lag/window, or provide more data.",
                     backtest(fc, d; horizon=3, windows=7, step=3))
        r = backtest(fc, d; horizon=3, initial=14)
        @test r.folds.origin[1] == d.ds[14]
        manual = forecast(fit(fc, map(c -> c[1:14], d)), 3).y_hat
        @test r.folds.y_hat[r.folds.fold .== 1] == manual
        @test backtest(fc, d; horizon=1, windows=3, step=1) isa BacktestResult
        # first == minhistory(fc) throws, one more row trains.
        @test_throws "would train its first fold on 13 of the 30 rows" backtest(
            fc, d; horizon=3, windows=15, step=1)
        @test backtest(fc, d; horizon=3, windows=14, step=1).folds.origin[1] == d.ds[14]
        r5 = backtest(fc, d; horizon=3, windows=5, step=3)
        @test r5.folds.origin[1] == d.ds[15]
        # Without a transform the old text stays.
        @test_throws("must exceed the feature set's minimum history (1 rows) so the " *
                     "first training window", backtest(
                         fc_single(TestModels.LinAR(1.0, 0.0); transform=NoTransform()),
                         d; horizon=3, initial=1))
    end

    @testset "panel backtest validates against the transform and skips short series" begin
        fc = spec(TestModels.LinAR(1.0, 0.0))
        p = panel("a" => 30, "b" => 28)
        @test_throws("backtest initial=13 must exceed the feature set's minimum " *
                     "history plus the 12 rows the transform drops (13 timestamps) so " *
                     "the first training window has at least one usable row. For a " *
                     "panel, initial counts distinct timestamps, not rows. Pass " *
                     "initial=14 or more, or reduce lags/windows or the transform's " *
                     "lag/window.", backtest(fc, p; horizon=3, initial=13))
        @test_throws("backtest windows=7 with step=3 and horizon=3 would train its " *
                     "first fold on at most 9 rows of the longest series (30 rows), " *
                     "but the feature set and transform need more than 13. Reduce " *
                     "windows, step or horizon or the transform's lag/window, or " *
                     "provide longer series.",
                     backtest(fc, p; horizon=3, windows=7, step=3))
        @test_throws "would train its first fold on at most 13 rows" backtest(
            fc, panel("a" => 30, "b" => 30); horizon=3, windows=15, step=1)
        r = backtest(fc, panel("a" => 30, "b" => 30); horizon=3, windows=14, step=1)
        @test r.folds.origin[1] == t0 + Day(13)
        r = backtest(fc, panel("a" => 30, "b" => 30); horizon=3, initial=14, step=13)
        @test r.folds.origin[1] == t0 + Day(13)
        # The short series sits out; the long one fits as alone.
        for model in models()
            fcm = spec(model)
            both = panel("long" => 60, "short" => 10)
            r = @test_logs (:warn, skipmsg(10)) match_mode = :any backtest(
                fcm, both; horizon=3, initial=20, step=10)
            solo = backtest(fcm, panel("long" => 60); horizon=3, initial=20, step=10)
            @test r.folds.y_hat == solo.folds.y_hat
            @test all(==("long"), r.folds.unique_id)
            @test r.metrics.value == solo.metrics.value
        end
    end
end
