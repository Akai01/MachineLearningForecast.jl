@testset "panel" begin
    # Distinct levels make any cross-series bleed obvious.
    function makepanel(; lens=(40, 30, 35), levels=(100.0, 10.0, 50.0),
                       starts=fill(Date(2022, 1, 1), 3))
        ids = String[]; ds = Date[]; y = Float64[]
        for (k, (n, lv, t0)) in enumerate(zip(lens, levels, starts))
            t = collect(t0:Day(1):t0 + Day(n - 1))
            append!(ids, fill("s$k", n)); append!(ds, t)
            append!(y, lv .+ Float64.(1:n))
        end
        return (unique_id=ids, ds=ds, y=y)
    end
    panel = makepanel()
    mk(model, feats; strat=Recursive(), id=:unique_id) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1),
                   target=:y, time=:ds, id=id)

    @testset "construction validates the id column" begin
        f = FeatureSet(Lag(1))
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0); features=f,
                                              freq=Day(1), target=:y, time=:ds, id=:y)
        @test_throws "id and target must be different columns, both were :y. Pass the " *
                     "name of your series-id column, e.g. id=:unique_id." Forecaster(
            TestModels.LinAR(1.0, 0.0); features=f, freq=Day(1), target=:y, time=:ds,
            id=:y)
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0); features=f,
                                              freq=Day(1), target=:y, time=:ds, id=:ds)
        @test_throws "id and time must be different columns, both were :ds. Pass the " *
                     "name of your series-id column, e.g. id=:unique_id." Forecaster(
            TestModels.LinAR(1.0, 0.0); features=f, freq=Day(1), target=:y, time=:ds,
            id=:ds)
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0); features=f,
                                              freq=Day(1), target=:y, time=:ds, id=:y_hat)
        @test_throws "id=:y_hat collides with a column name reserved by forecast()/" *
                     "backtest() results" Forecaster(TestModels.LinAR(1.0, 0.0);
            features=f, freq=Day(1), target=:y, time=:ds, id=:y_hat)
        idmsg = "id=:fold collides with a column name reserved by forecast()/" *
                "backtest() results (:origin, :step, :y, :y_hat, :sample, :fold). " *
                "Rename the id column in your data."
        @test_throws idmsg mk(TestModels.LinAR(1.0, 0.0), f; id=:fold)
        # refused at construction, so no fold is ever fitted
        @test_throws idmsg backtest(
            mk(TestModels.LinAR(1.0, 0.0), f; id=:fold),
            (fold=panel.unique_id, ds=panel.ds, y=panel.y); horizon=5, initial=20)
        @test_throws idmsg fit(mk(TestModels.LinAR(1.0, 0.0), f; id=:fold),
                               (fold=panel.unique_id, ds=panel.ds, y=panel.y))
        tmsg = "time=:fold collides with a column name reserved by forecast()/" *
               "backtest() results (:origin, :step, :y, :y_hat, :sample, :fold). " *
               "Rename the time column in your data."
        @test_throws tmsg backtest(
            Forecaster(TestModels.LinAR(1.0, 0.0); features=f, freq=Day(1), target=:y,
                       time=:fold, id=:unique_id),
            (unique_id=panel.unique_id, fold=panel.ds, y=panel.y); horizon=5, initial=20)
        err = try Forecaster(TestModels.LinAR(1.0, 0.0);
                             features=FeatureSet(Lag(1), Exogenous(:sid)),
                             freq=Day(1), target=:y, time=:ds, id=:sid) catch e; e end
        @test err isa ArgumentError && occursin("series id", err.msg)
        @test !MachineLearningForecast.ispanel(Forecaster(TestModels.LinAR(1.0, 0.0);
                                                          features=f, freq=Day(1)))
        @test MachineLearningForecast.ispanel(mk(TestModels.LinAR(1.0, 0.0), f))
    end

    @testset "fit covers every series and keeps their state apart" begin
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel)
        @test nseries(f) == 3
        @test length(f.models) == 1
        @test [s.id for s in f.series] == ["s1", "s2", "s3"]
        @test [s.n_train for s in f.series] == [40, 30, 35]
        @test [s.t_last for s in f.series] ==
              [Date(2022, 2, 9), Date(2022, 1, 30), Date(2022, 2, 4)]
        err = try f.y_history catch e; e end
        @test err isa ArgumentError && occursin("3 series", err.msg)
    end

    @testset "features never reach across a series boundary" begin
        # MeanModel exposes the pooled targets; bleed adds rows.
        f = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(1))), panel)
        expected = Float64[]
        for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))
            append!(expected, lv .+ Float64.(2:n))       # row 1 of each series dropped
        end
        @test only(unique(forecast(f, 1).y_hat)) ≈ mean(expected)

        # With Lag(7) each series loses 7 rows, not 7 overall.
        f7 = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(7))), panel)
        exp7 = Float64[]
        for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))
            append!(exp7, lv .+ Float64.(8:n))
        end
        @test only(unique(forecast(f7, 1).y_hat)) ≈ mean(exp7)

        # Each target feature drops minhistory rows per series.
        for g in (RollingMean(3; lag=2), RollingStd(2), RollingMin(2; lag=4),
                  RollingMax(6), Diff(2), CustomFeature(:hm, mean, 7))
            mh = MachineLearningForecast.minhistory(g)
            fg = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(1), g)), panel)
            kept = reduce(vcat, [lv .+ Float64.((mh + 1):n)
                                 for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))])
            @test only(unique(forecast(fg, 1).y_hat)) ≈ mean(kept)
        end
    end

    @testset "each series forecasts from its own history and timestamp" begin
        # LinAR(1, 0) echoes y_lag_1: each series' last value.
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel)
        out = forecast(f, 3)
        @test keys(out) == (:unique_id, :ds, :y_hat)
        @test length(out.y_hat) == 3 * 3
        for (k, st) in enumerate(f.series)
            rows = (3k - 2):(3k)
            @test all(out.unique_id[rows] .== st.id)
            @test out.y_hat[rows] ≈ fill(st.y_history[end], 3)
            @test out.ds[rows] == [st.t_last + Day(i) for i in 1:3]
        end
    end

    @testset "grouping tolerates interleaved and unsorted rows" begin
        n = length(panel.y)
        perm = shuffle(StableRNG(3), 1:n)
        shuffled = (unique_id=panel.unique_id[perm], ds=panel.ds[perm], y=panel.y[perm])
        a = forecast(fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel), 2)
        b = forecast(fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), shuffled), 2)
        # Shuffling reorders rows but not any series' forecast.
        keyed(o) = Dict((o.unique_id[i], o.ds[i]) => o.y_hat[i] for i in eachindex(o.y_hat))
        @test Set(keys(keyed(a))) == Set(keys(keyed(b)))
        @test all(keyed(a)[k] ≈ keyed(b)[k] for k in keys(keyed(a)))
    end

    @testset "per-series validation errors name the series" begin
        bad = makepanel()
        drop = findfirst(i -> bad.unique_id[i] == "s2" && bad.ds[i] == Date(2022, 1, 15),
                         eachindex(bad.y))
        @test drop !== nothing                      # the fixture really has that row
        keep = setdiff(eachindex(bad.y), drop)
        gapped = (unique_id=bad.unique_id[keep], ds=bad.ds[keep], y=bad.y[keep])
        spec = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        err = try fit(spec, gapped) catch e; e end
        @test err isa ArgumentError
        @test occursin("series unique_id=\"s2\"", err.msg) && occursin("gap", err.msg)

        miss = (unique_id=Vector{Union{Missing,String}}(bad.unique_id),
                ds=bad.ds, y=bad.y)
        miss.unique_id[3] = missing
        err2 = try fit(spec, miss) catch e; e end
        @test err2 isa ArgumentError && occursin("missing value", err2.msg)
        @test occursin("id column :unique_id has a missing value at row 3. Every row " *
                       "must belong to a series; drop or fill the rows with a missing " *
                       "id before fitting.", err2.msg)
    end

    @testset "an empty panel, and Direct steps no series can reach" begin
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            empty = (unique_id=String[], ds=Date[], y=Float64[])
            @test_throws ArgumentError fit(mk(m, FeatureSet(Lag(1))), empty)
            @test_throws "the panel is empty: no rows found in the id column " *
                         ":unique_id. Pass a table with at least one row." fit(
                mk(m, FeatureSet(Lag(1))), empty)
            # 3 rows per series leave 2 training rows, so no step-3 model
            fc = mk(m, FeatureSet(Lag(1)); strat=Direct(5))
            tiny = makepanel(lens=(3, 3, 3))
            @test_throws ArgumentError fit(fc, tiny)
            @test_throws "not enough data for Direct(5): no series has enough rows to " *
                         "train the step-3 model. Shorten max_horizon" fit(fc, tiny)
        end
    end

    @testset "per-series data errors name the series and the user's row" begin
        # s2 holds rows 61:110, so its 5th row is the user's row 65.
        big = merge(makepanel(lens=(60, 50, 55)), (promo=Float64.(1:165),))
        cases = ((:y, missing,
                  "target column :y contains missing values (first at row 65)"),
                 (:y, NaN, "target column :y contains the non-finite value NaN at row 65"),
                 (:promo, missing,
                  "feature column :promo has a missing value at row 65 (time 2022-01-05)"))
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            for (col, val, phrase) in cases
                fc = mk(m, FeatureSet(Lag(1), Exogenous(:promo)))
                v = Vector{Union{Missing,Float64}}(big[col])
                v[65] = val
                data = merge(big, NamedTuple{(col,)}((v,)))
                msg = "series unique_id=\"s2\": " * phrase
                @test_throws ArgumentError fit(fc, data)
                @test_throws msg fit(fc, data)
                @test_throws msg backtest(fc, data; horizon=5, initial=40, step=10)
            end
            # a missing timestamp sorts last within its series
            tm = merge(big, (ds=Vector{Union{Missing,Date}}(big.ds),))
            tm.ds[5] = missing
            msg = "series unique_id=\"s1\": time column :ds contains missing values " *
                  "(first at row 5)"
            @test_throws msg fit(mk(m, FeatureSet(Lag(1))), tm)
            @test_throws msg backtest(mk(m, FeatureSet(Lag(1))), tm; horizon=5, initial=40)
            # interleaved weekly: b's 5th week is the user's row 10
            mondays = collect(Date(2024, 1, 1):Week(1):Date(2024, 3, 4))
            wk = (unique_id=repeat(["a", "b"], 10), ds=repeat(mondays, inner=2),
                  y=Float64.(1:20))
            wk.ds[10] += Day(1)
            fcw = Forecaster(m; features=FeatureSet(Lag(1)), freq=Week(1), id=:unique_id)
            @test_throws "series unique_id=\"b\": time column :ds has the timestamp " *
                         "2024-01-30 at row 10" fit(fcw, wk)
        end
    end

    @testset "series too short to train are skipped, not fatal" begin
        short = makepanel(lens=(40, 3, 35))
        f = @test_logs (:warn, r"skipping 1 series") match_mode=:any fit(
            mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(7))), short)
        @test nseries(f) == 2 && [s.id for s in f.series] == ["s1", "s3"]
        err = try fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(30))),
                      makepanel(lens=(5, 4, 3))) catch e; e end
        @test err isa ArgumentError && occursin("no series has enough rows", err.msg)
    end

    @testset "infinite exogenous and non-finite custom values name the series" begin
        promo = Float64.(eachindex(panel.y))
        promo[45] = Inf               # s2, row 5: 2022-01-05
        ex = (unique_id=panel.unique_id, ds=panel.ds, y=panel.y, promo=promo)
        tail = "Infinite values turn into Inf or NaN in the model's arithmetic; " *
               "replace or clip them (NaN is allowed and passes through to the " *
               "model as a missing marker)."
        nanc = CustomFeature(:c, h -> h[end] < 20 ? NaN : h[end], 1)
        for m in (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1))),
            strat in (Recursive(), Direct(2))
            fc = mk(m, FeatureSet(Lag(1), Exogenous(:promo)); strat)
            @test_throws "series unique_id=\"s2\": exogenous column :promo has the " *
                         "infinite value Inf at time 2022-01-05 in the training data. " *
                         "$tail" fit(fc, ex)
            @test_throws "series unique_id=\"s2\": CustomFeature(:c) returned NaN at " *
                         "time 2022-01-02, where f sees 1 history value, but f must " *
                         "return a finite number." fit(
                mk(m, FeatureSet(Lag(1), nanc); strat), panel)
        end
        fc = mk(TestModels.EchoColumn(:promo), FeatureSet(Lag(1), Exogenous(:promo)))
        f = fit(fc, (unique_id=panel.unique_id, ds=panel.ds, y=panel.y,
                     promo=Float64.(eachindex(panel.y))))
        grids = [[s.t_last + Day(i) for i in 1:2] for s in f.series]
        fpromo = [11.0, 12.0, 21.0, 22.0, 31.0, -Inf]
        nd = (unique_id=repeat(["s1", "s2", "s3"], inner=2), ds=vcat(grids...),
              promo=fpromo)
        @test_throws "series unique_id=\"s3\": exogenous column :promo has the " *
                     "infinite value -Inf at time $(grids[3][2]) in `future` (or the " *
                     "held-out rows)" forecast(f, 2; future=nd)
        fpromo[6] = NaN
        @test isnan(forecast(f, 2; future=nd).y_hat[6])
        # The series is named once, not once per layer.
        blow = CustomFeature(:blow, h -> h[end] > 1000 ? NaN : h[end], 1)
        fb = fit(mk(TestModels.LinAR(20.0, 0.0), FeatureSet(Lag(1), blow)), panel)
        @test_throws "series unique_id=\"s1\": CustomFeature(:blow) returned NaN at " *
                     "time $(fb.series[1].t_last + Day(2)), where f sees 41 history " *
                     "values, but f must return a finite number" forecast(fb, 3)
        @test all(isfinite, forecast(fb, 1).y_hat)
        # Only ArgumentErrors get the series prefix.
        dom = CustomFeature(:dom, h -> h[end] > 1000 ? throw(DomainError(h[end], "big")) :
                                  h[end], 1)
        fd = fit(mk(TestModels.LinAR(20.0, 0.0), FeatureSet(Lag(1), dom)), panel)
        err = try forecast(fd, 3) catch e; e end
        @test err isa DomainError && err.msg == "big"
        late = copy(promo); late[45] = 1.0; late[104] = Inf   # s3, row 34
        @test_throws "series unique_id=\"s3\": exogenous column :promo has the " *
                     "infinite value Inf at time 2022-02-03 in `future` (or the " *
                     "held-out rows)" backtest(
            mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1), Exogenous(:promo))),
            (unique_id=panel.unique_id, ds=panel.ds, y=panel.y, promo=late);
            horizon=2, initial=20)
        pre = "series unique_id=\"s3\": "
        msg = try forecast(f, 2; future=merge(nd, (promo=[11.0, 12.0, 21.0, 22.0, 31.0,
                                                          -Inf],))) catch e; e.msg end
        @test startswith(msg, pre) && count(pre, msg) == 1
        msg = try
            backtest(mk(TestModels.LinAR(1.0, 0.0),
                        FeatureSet(Lag(1), Exogenous(:promo))),
                     (unique_id=panel.unique_id, ds=panel.ds, y=panel.y, promo=late);
                     horizon=2, initial=20)
        catch e; e.msg end
        @test startswith(msg, pre) && count(pre, msg) == 1
        bt = (unique_id=panel.unique_id, ds=panel.ds, y=panel.y, promo=promo)
        @test_throws "series unique_id=\"s2\": exogenous column :promo has the " *
                     "infinite value Inf at time 2022-01-05" backtest(
            mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1), Exogenous(:promo))), bt;
            horizon=2, initial=20)
    end

    @testset "exogenous covariates join on (id, time)" begin
        ex = (unique_id=panel.unique_id, ds=panel.ds, y=panel.y,
              promo=Float64.(eachindex(panel.y)))
        fc = mk(TestModels.EchoColumn(:promo), FeatureSet(Lag(1), Exogenous(:promo)))
        f = fit(fc, ex)
        grids = [[s.t_last + Day(i) for i in 1:2] for s in f.series]
        nd = (unique_id=repeat(["s1", "s2", "s3"], inner=2),
              ds=vcat(grids...), promo=[11.0, 12.0, 21.0, 22.0, 31.0, 32.0])
        @test forecast(f, 2; future=nd).y_hat == [11.0, 12.0, 21.0, 22.0, 31.0, 32.0]
        p2 = shuffle(StableRNG(9), 1:6)
        shuf = (unique_id=nd.unique_id[p2], ds=nd.ds[p2], promo=nd.promo[p2])
        @test forecast(f, 2; future=shuf).y_hat == [11.0, 12.0, 21.0, 22.0, 31.0, 32.0]

        @test_throws ArgumentError forecast(f, 2)
        @test_throws "features contain Exogenous(:promo) but forecast() was called " *
                     "without `future`. Pass a table with columns (:unique_id, :ds, " *
                     ":promo) covering all 3 series over their 2 forecast steps." (
            forecast(f, 2))
        missing_row = (unique_id=nd.unique_id[1:5], ds=nd.ds[1:5], promo=nd.promo[1:5])
        err = try forecast(f, 2; future=missing_row) catch e; e end
        @test err isa ArgumentError && occursin("no row for", err.msg)
        @test_throws "`future` has no row for unique_id=\"s3\" at $(nd.ds[6]), which is " *
                     "step 2 of that series' forecast. Provide exogenous values for " *
                     "every series at every forecast timestamp." forecast(
            f, 2; future=missing_row)
        @test_throws MethodError forecast(f, 2; new_data=nd)
        @test_throws "Every `future` row needs a timestamp; drop or fill that row." (
            forecast(f, 2; future=merge(
                nd, (ds=Union{Missing,Date}[missing; nd.ds[2:end]],))))
        plain = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel)
        @test_logs (:warn, "`future` was passed but the feature set has no Exogenous " *
                           "features; it will be ignored.") forecast(plain, 2; future=nd)
        noid = (ds=nd.ds, promo=nd.promo)
        err2 = try forecast(f, 2; future=noid) catch e; e end
        @test err2 isa ArgumentError && occursin("unique_id", err2.msg)
        @test_throws "`future` is missing column :unique_id; a panel forecast needs " *
                     "the id column :unique_id, the time column :ds and the exogenous " *
                     "column :promo." forecast(f, 2; future=noid)
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            @test_throws "id column :unique_id not found in the data. Available " *
                         "columns: ds, y, promo. Pass id= with the name of your id " *
                         "column, or rename it to :unique_id." fit(
                mk(m, FeatureSet(Lag(1))), (ds=ex.ds, y=ex.y, promo=ex.promo))
        end
    end

    @testset "time columns typed Union{Missing,Date} or Any work with Exogenous" begin
        ex = merge(makepanel(lens=(60, 50, 55)), (promo=Float64.(1:165),))
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fc = mk(m, FeatureSet(Lag(1), Exogenous(:promo)))
            f = fit(fc, ex)
            grids = [[s.t_last + Day(i) for i in 1:2] for s in f.series]
            nd = (unique_id=repeat(["s1", "s2", "s3"], inner=2), ds=vcat(grids...),
                  promo=[11.0, 12.0, 21.0, 22.0, 31.0, 32.0])
            want = forecast(f, 2; future=nd)
            bt = backtest(fc, ex; horizon=5, initial=40, step=10)
            for T in (Union{Missing,Date}, Any)
                @test forecast(f, 2; future=merge(nd, (ds=Vector{T}(nd.ds),))) == want
                r = backtest(fc, merge(ex, (ds=Vector{T}(ex.ds),)); horizon=5,
                             initial=40, step=10)
                @test isequal(r.forecasts, bt.forecasts) && isequal(r.scores, bt.scores)
            end
            holed = merge(nd, (ds=Vector{Union{Missing,Date}}(nd.ds),))
            holed.ds[4] = missing
            @test_throws ArgumentError forecast(f, 2; future=holed)
            @test_throws "`future`'s time column :ds has a missing value at row 4" forecast(
                f, 2; future=holed)
            @test_throws "has element type DateTime at row 1, but the training time " *
                         "column is Date" forecast(f, 2; future=merge(
                             nd, (ds=DateTime.(nd.ds),)))
            dup = (unique_id=[nd.unique_id; "s2"], ds=[nd.ds; nd.ds[3]],
                   promo=[nd.promo; 0.0])
            @test_throws ArgumentError forecast(f, 2; future=dup)
            @test_throws "`future` has duplicate rows for unique_id=\"s2\" at " *
                         "$(nd.ds[3]). Deduplicate `future`" forecast(f, 2; future=dup)
            gap = merge(nd, (promo=Union{Missing,Float64}[nd.promo...],))
            gap.promo[4] = missing
            @test_throws ArgumentError forecast(f, 2; future=gap)
            @test_throws "`future` has a missing value in exogenous column :promo for " *
                         "unique_id=\"s2\" at $(nd.ds[4]). Provide complete exogenous " *
                         "values for every series at every forecast step." forecast(
                f, 2; future=gap)
        end
    end

    @testset "Direct on a panel" begin
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=Direct(4)), panel)
        @test length(f.models) == 4
        out = forecast(f, 4)
        @test length(out.y_hat) == 3 * 4
        @test_throws ArgumentError forecast(f, 5)
        @test_throws "strategy=Direct(4) was fit with max_horizon=4 but forecast(h=5) " *
                     "was requested. Refit with Direct(5) or use Recursive()." (
            forecast(f, 5))

        # MeanModel shows machine i's targets, shifted per series.
        fm = fit(mk(TestModels.MeanModel(), FeatureSet(Lag(1)); strat=Direct(3)), panel)
        got = forecast(fm, 3)
        for i in 1:3
            pooled = Float64[]
            for (n, lv) in zip((40, 30, 35), (100.0, 10.0, 50.0))
                append!(pooled, lv .+ Float64.((1 + i):n))   # y[i:nX] within series
            end
            @test got.y_hat[i] ≈ mean(pooled)
        end
    end

    @testset "panel Direct anchors time/exogenous columns to the target row" begin
        # Levels keep y, and so promo, distinct across series.
        starts = [Date(2022, 1, 1), Date(2022, 1, 11), Date(2022, 1, 21)]
        lv = (1000.0, 2000.0, 3000.0)
        rag = makepanel(lens=(60, 50, 40), levels=lv, starts=starts)
        dd = merge(rag, (promo=3 .* rag.y,))
        fc = mk(TestModels.PairLookup(:promo), FeatureSet(Lag(1), Exogenous(:promo));
                strat=Direct(4))
        # Row r of series k has y = lv[k] + r, so promo = 3 * y there.
        rows = ([10, 20, 30, 40], [8, 18, 28, 38], [6, 16, 26, 36])
        future = (unique_id=repeat(["s1", "s2", "s3"], inner=4),
                  ds=repeat(Date(2022, 3, 2) .+ Day.(0:3), 3),
                  promo=[3 * (lv[k] + r) for k in 1:3 for r in rows[k]])
        got = forecast(fit(fc, dd), 4; future=future).y_hat
        # Correct anchoring maps promo to y of its own row.
        @test got ≈ [lv[k] + r for k in 1:3 for r in rows[k]]
        # Taking promo from the feature row shifts step i by i - 1.
        @test got != [lv[k] + r + i - 1 for k in 1:3 for (i, r) in enumerate(rows[k])]

        # A TimeFeature must be recoverable at every step too.
        dw = (unique_id=rag.unique_id, ds=rag.ds, y=Float64.(dayofweek.(rag.ds)))
        fc_cal = mk(TestModels.PairLookup(:dayofweek),
                    FeatureSet(Lag(1), Calendar(:dayofweek)); strat=Direct(4))
        fut_ds = Date(2022, 3, 2) .+ Day.(0:3)
        @test forecast(fit(fc_cal, dw), 4).y_hat ≈ repeat(Float64.(dayofweek.(fut_ds)), 3)
    end

    @testset "panel Direct honours anchor, not the supertype" begin
        P = AnchorProbes
        # Slope 1 keeps y - ylast equal across series.
        a = Float64.(1000 .+ (1:60))
        ra = (unique_id=fill("a", 60), ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 3, 1)),
              y=a)
        rb = (unique_id=fill("b", 40),
              ds=collect(Date(2022, 1, 11):Day(1):Date(2022, 2, 19)), y=a[1:40] .+ 500)
        rag = map(vcat, ra, rb)
        trained(feat) = [m.fitresult for m in fit(
            mk(TestModels.KeepTrain(), FeatureSet(feat); strat=Direct(4)), rag).models]
        origin(base, n, i) = (base .+ (1:(n - i)), base .+ ((1 + i):n))
        target(base, n, i) = (base .+ (i:(n - 1)), base .+ ((1 + i):n))
        for feat in (P.LagOrigin(), P.YOrigin())
            for (i, r) in enumerate(trained(feat))
                (xa, ya), (xb, yb) = origin(1000, 60, i), origin(1500, 40, i)
                @test r.X.ylast == [xa; xb] && r.y == [ya; yb]
                @test r.y .- r.X.ylast == fill(float(i), length(r.y))
            end
        end
        for (i, r) in enumerate(trained(P.LagAtTarget()))
            (xa, ya), (xb, yb) = target(1000, 60, i), target(1500, 40, i)
            @test r.X.ylast == [xa; xb] && r.y == [ya; yb]
            @test r.y .- r.X.ylast == fill(1.0, length(r.y))
        end
        dw = (unique_id=rag.unique_id, ds=rag.ds, y=Float64.(dayofweek.(rag.ds)))
        fc_dow = mk(TestModels.PairLookup(:dayofweek), FeatureSet(P.DowTarget());
                    strat=Direct(4))
        fut_ds = Date(2022, 3, 2) .+ Day.(0:3)
        got = forecast(fit(fc_dow, dw), 4).y_hat
        @test got == Float64.(dayofweek.([fut_ds; Date(2022, 2, 20) .+ Day.(0:3)]))

        fs = FeatureSet(P.NoAnchor())
        err = "NoAnchor has no anchor. Direct fitting must know whether its columns"
        fix = "or define MachineLearningForecast.anchor(f::NoAnchor) = " *
              "MachineLearningForecast.AtOrigin() or MachineLearningForecast.AtTarget()."
        @test_throws ArgumentError fit(mk(TestModels.KeepTrain(), fs; strat=Direct(3)), rag)
        @test_throws err fit(mk(TestModels.KeepTrain(), fs; strat=Direct(3)), rag)
        @test_throws fix fit(mk(TestModels.KeepTrain(), fs; strat=Direct(3)), rag)
        rec = fit(mk(TestModels.EchoColumn(:ylast), fs), rag)
        @test forecast(rec, 2).y_hat == [1060.0, 1060.0, 1540.0, 1540.0]
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            @test all(isfinite, forecast(fit(mk(m, fs), rag), 2).y_hat)
            @test_throws err fit(mk(m, fs; strat=Direct(3)), rag)
        end
    end

    @testset "panel backtest rejects metrics with the same name" begin
        fcd = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        mkm(k) = (a, b) -> k * mae(a, b)
        @test_throws ArgumentError backtest(fcd, panel; horizon=5, initial=20,
                                            metrics=(mkm(1), mkm(2)))
        @test_throws "2 metrics are named $(repr(Symbol(mkm(1))))" backtest(
            fcd, panel; horizon=5, initial=20, metrics=(mkm(1), mkm(2)))
        @test_throws "backtest metrics must have distinct names, but 2 metrics are " *
                     "named :mae, so" backtest(fcd, panel; horizon=5, initial=20,
                                               metrics=(mae, mae))
        @test_throws "2 metrics are named :mae" backtest(
            fcd, panel; horizon=5, windows=2, step=5, metrics=(mae, mae))
        r = backtest(fcd, panel; horizon=5, initial=20, metrics=(mae, rmse))
        @test Set(r.scores.metric) == Set([:mae, :rmse])
    end

    @testset "backtest cuts folds on the global timestamp grid" begin
        big = makepanel(lens=(60, 50, 55))
        fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        r = backtest(fc, big; horizon=5, initial=40, step=10, metrics=(mae, rmse))
        @test keys(r.forecasts) == (:fold, :origin, :step, :unique_id, :ds, :y, :y_hat)
        @test issorted(r.forecasts.fold) && unique(r.forecasts.fold) == 1:2
        @test length(unique(r.forecasts.origin)) == length(40:10:(60 - 5))
        @test extrema(r.forecasts.step) == (1, 5)
        @test Set(unique(r.forecasts.unique_id)) ⊆ Set(["s1", "s2", "s3"])
        actual = Dict((big.unique_id[i], big.ds[i]) => big.y[i] for i in eachindex(big.y))
        @test all(actual[(r.forecasts.unique_id[j], r.forecasts.ds[j])] == r.forecasts.y[j]
                  for j in eachindex(r.forecasts.y))
        # LinAR(1, 0) is naive: each step repeats the origin's actual.
        @test all(yh == actual[(id, o)] for (yh, id, o) in
                  zip(r.forecasts.y_hat, r.forecasts.unique_id, r.forecasts.origin))
        # y climbs 1 a day, so step s misses by s in every fold.
        @test r.scores.fold == [1, 1, 2, 2]
        @test r.scores.value ≈ repeat([3.0, sqrt(11.0)], 2)
        @test occursin("horizon 5", sprint(show, MIME"text/plain"(), r))
        @test occursin("folds", sprint(show, r))
        @test_throws ArgumentError backtest(mk(TestModels.LinAR(1.0, 0.0),
                                               FeatureSet(Lag(30))), big;
                                            horizon=5, initial=20, step=10)
        @test_throws "backtest initial=20 must exceed the feature set's minimum history " *
                     "(30 timestamps) so the first training window has at least one " *
                     "usable row. For a panel, initial counts distinct timestamps, not " *
                     "rows. Pass initial=31 or more, or reduce lags/windows." backtest(
            mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(30))), big; horizon=5,
            initial=20, step=10)
        for bad in (mae, (:mae,), ("mae",))
            @test_throws ArgumentError backtest(fc, big; horizon=5, initial=40,
                                                metrics=bad)
            @test_throws "metrics=(mae, rmse)" backtest(fc, big; horizon=5,
                                                        initial=40, metrics=bad)
        end
    end

    @testset "scaled metrics use each series' own training values" begin
        # a climbs 1 a day and b 10, so each naive MAE is its slope.
        function trends(nb)
            ids = String[]; ds = Date[]; y = Float64[]
            for i in 1:50, (id, slope, n) in (("a", 1.0, 50), ("b", 10.0, nb))
                i ≤ n || continue
                push!(ids, id); push!(ds, Date(2022, 1, 1) + Day(i - 1))
                push!(y, (id == "a" ? 0.0 : 1000.0) + slope * i)
            end
            return (unique_id=ids, ds=ds, y=y)
        end
        fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        scores(pd) = backtest(fc, pd; horizon=5, initial=40, step=5, metrics=(mae, mase))
        r = scores(trends(50))
        # Naive misses step s by s*slope: MAE 3 and 30, MASE 3 each.
        @test r.scores.value[r.scores.metric .== :mase] ≈ [3.0, 3.0]
        @test r.scores.value[r.scores.metric .== :mae] ≈ [16.5, 16.5]
        pd = trends(50)
        shuffled = map(c -> c[shuffle(StableRNG(2), 1:length(pd.y))], pd)
        @test scores(shuffled).scores.value ≈ r.scores.value
        # b ends 2 days into the last fold: weigh series by their points.
        rr = scores(trends(47))
        @test rr.scores.value[rr.scores.metric .== :mase] ≈
              [3.0, (5 * 3.0 + 2 * 1.5) / 7]
    end

    @testset "rmsse is per series and point-weighted on a panel" begin
        # b's scale differs; b ends 2 days into the last fold.
        function wiggly(lb)
            ids = String[]; ds = Date[]; y = Float64[]
            for (id, lv, n, f) in (("a", 100.0, 55, 1.0), ("b", 5.0, 52, lb))
                for i in 1:n
                    push!(ids, id); push!(ds, Date(2022, 1, 1) + Day(i - 1))
                    push!(y, f * (lv + i + 3 * sin(i) + (i % 7)))
                end
            end
            return (unique_id=ids, ds=ds, y=y)
        end
        function manual(pd, r)
            vals = Float64[]
            for k in sort(unique(r.forecasts.fold))
                tot = 0.0; cnt = 0
                for id in unique(pd.unique_id)
                    sel = (r.forecasts.fold .== k) .& (r.forecasts.unique_id .== id)
                    any(sel) || continue
                    o = r.forecasts.origin[sel][1]
                    tr = pd.y[(pd.unique_id .== id) .& (pd.ds .<= o)]
                    sc = mean(abs2(tr[i] - tr[i - 1]) for i in 2:length(tr))
                    e = rmse(r.forecasts.y[sel], r.forecasts.y_hat[sel])
                    tot += sum(sel) * e / sqrt(sc)
                    cnt += sum(sel)
                end
                push!(vals, tot / cnt)
            end
            return vals
        end
        score(model, pd) = backtest(mk(model, FeatureSet(Lag(1))), pd; horizon=5,
                                    initial=40, step=5, metrics=(rmsse,))
        for m in (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            pd = wiggly(1.0)
            r = score(m, pd)
            vals = r.scores.value
            @test length(vals) == 3
            @test vals ≈ manual(pd, r)
            @test summarize(r).mean ≈ [mean(vals)]
        end
        # Naive is equivariant, so a rescaled series is a no-op.
        naive = TestModels.LinAR(1.0, 0.0)
        @test score(naive, wiggly(1000.0)).scores.value ≈
              score(naive, wiggly(1.0)).scores.value
    end

    @testset "windows counts each series' folds back from its own end" begin
        pd = makepanel(lens=(40, 30, 35),
                       starts=[Date(2022, 1, 1), Date(2022, 1, 10), Date(2022, 1, 3)])
        fc = mk(TestModels.MeanModel(), FeatureSet(Lag(1)))
        r = backtest(fc, pd; horizon=5, windows=1, metrics=(mae, mase))
        @test unique(r.forecasts.fold) == [1]
        for (id, n) in (("s1", 40), ("s2", 30), ("s3", 35))
            ds = pd.ds[pd.unique_id .== id]
            sel = r.forecasts.unique_id .== id
            @test r.forecasts.ds[sel] == ds[(n - 4):n]
            @test all(==(ds[n - 5]), r.forecasts.origin[sel])
        end
        # The series' cutoffs differ, so the fold has no single origin.
        @test ismissing(r.scores.origin[1])
        @test occursin("1 fold, horizon 5", sprint(show, MIME"text/plain"(), r))
        # windows=1 is the M competitions' holdout of each series' end.
        keep = reduce(vcat, [findall(==(id), pd.unique_id)[1:(end - 5)]
                             for id in ("s1", "s2", "s3")])
        manual = forecast(fit(fc, map(c -> c[keep], pd)), 5)
        byrow = Dict(zip(zip(manual.unique_id, manual.ds), manual.y_hat))
        rows = zip(r.forecasts.unique_id, r.forecasts.ds)
        @test r.forecasts.y_hat == [byrow[k] for k in rows]
        r2 = backtest(fc, pd; horizon=5, windows=2, step=3)
        s2 = r2.forecasts.unique_id .== "s2"
        @test unique(r2.forecasts.origin[s2]) == pd.ds[pd.unique_id .== "s2"][[22, 25]]
        pdx = merge(pd, (promo=ones(length(pd.y)),))
        fcx = mk(TestModels.MeanModel(), FeatureSet(Lag(1), Exogenous(:promo)))
        @test backtest(fcx, pdx; horizon=5, windows=2, step=3).forecasts.y_hat ==
              r2.forecasts.y_hat
        @test_throws("would train its first fold on at most -10 rows of the longest " *
                     "series (40 rows)", backtest(fc, pd; horizon=5, windows=10, step=5))
        res = tune(fc, pd; grid=(features=[FeatureSet(Lag(1)), FeatureSet(Lag(2))],),
                   horizon=5, windows=1, metric=mae)
        @test res.table.mean_score[1] ≈
              only(summarize(backtest(fc, pd; horizon=5, windows=1, metrics=(mae,))).mean)
    end

    @testset "BacktestResult and summarize on a ragged panel" begin
        pd = makepanel(lens=(40, 30, 35),
                       starts=[Date(2022, 1, 1), Date(2022, 1, 10), Date(2022, 1, 3)])
        fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        for kw in ((windows=3, step=3), (initial=15, step=5))
            r = backtest(fc, pd; horizon=5, kw..., metrics=(mae, rmse, Mase(2)))
            @test keys(r.forecasts) == (:fold, :origin, :step, :unique_id, :ds, :y, :y_hat)
            @test keys(r.scores) == (:fold, :origin, :metric, :value)
            @test !hasproperty(r, :folds) && !hasproperty(r, :metrics)
            nf = length(unique(r.forecasts.fold))
            @test 0 ∉ r.scores.fold && length(r.scores.fold) == 3nf
            s = summarize(r)
            @test s.metric == [:mae, :rmse, :mase_2] && s.nfolds == fill(nf, 3)
            for (i, name) in enumerate(s.metric)
                v = r.scores.value[r.scores.metric .== name]
                @test s.mean[i] ≈ sum(v) / nf
                @test s.std[i] ≈ sqrt(sum(abs2, v .- sum(v) / nf) / (nf - 1))
            end
            @test Tables.istable(r.forecasts) && Tables.istable(summarize(r))
        end
        # Different cutoffs leave a fold's origin missing.
        rw = backtest(fc, pd; horizon=5, windows=2, step=3)
        @test eltype(rw.scores.origin) == Union{Missing,Date}
        @test all(ismissing, rw.scores.origin)
        ri = backtest(fc, pd; horizon=5, initial=15, step=5)
        @test eltype(ri.scores.origin) == Union{Missing,Date}
        @test !any(ismissing, ri.scores.origin)
        @test ri.scores.origin[1] == minimum(pd.ds) + Day(14)
    end

    @testset "a panel backtest scores ids of any element type" begin
        for ids in (Int, Any, Symbol)
            raw = makepanel()
            conv(x) = ids === Int ? parse.(Int, replace.(x, "s" => "")) :
                      ids === Symbol ? Symbol.(x) : Any[x...]
            pd = merge(raw, (unique_id=conv(raw.unique_id),))
            r = backtest(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), pd;
                         horizon=5, initial=20, step=5, metrics=(mae,))
            ref = backtest(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), raw;
                           horizon=5, initial=20, step=5, metrics=(mae,))
            @test r.forecasts.y == ref.forecasts.y
            @test r.forecasts.y_hat == ref.forecasts.y_hat
            @test r.scores.value == ref.scores.value
            @test eltype(r.forecasts.unique_id) == eltype(pd.unique_id)
            @test summarize(r).mean == summarize(ref).mean
        end
    end

    @testset "the step column is each row's position in its series' forecast" begin
        steps = MachineLearningForecast._steps_by_id
        @test steps(["a", "b", "a", "c", "b", "a"]) == [1, 1, 2, 1, 2, 3]
        @test steps(String[]) == Int[]
        ids = rand(StableRNG(5), 1:7, 300)
        @test steps(ids) == [count(==(ids[j]), ids[1:(j - 1)]) + 1 for j in eachindex(ids)]
        # s2 ends inside the last fold, so its rows stop early.
        big = makepanel(lens=(60, 52, 55))
        r = backtest(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), big;
                     horizon=5, initial=40, step=10)
        @test all(Dates.value(d - o) == s
                  for (d, o, s) in zip(r.forecasts.ds, r.forecasts.origin,
                                       r.forecasts.step))
        @test r.forecasts.step[(r.forecasts.origin .== Date(2022, 2, 19)) .&
                           (r.forecasts.unique_id .== "s2")] == [1, 2]
    end

    @testset "Fourier is locked to the calendar across ragged series" begin
        # A weekly pattern by date, sampled over ragged spans.
        wk(d) = 10 + 5 * sin(2π * Dates.value(d) / 7)
        spans = [(Date(2022, 1, 1), 70), (Date(2022, 1, 4), 60), (Date(2022, 1, 13), 45)]
        ids = reduce(vcat, [fill("s$k", n) for (k, (_, n)) in enumerate(spans)])
        ds = reduce(vcat, [collect(d0:Day(1):d0 + Day(n - 1)) for (d0, n) in spans])
        week = (unique_id=ids, ds=ds, y=wk.(ds))
        tree = DecisionTreeRegressor(min_samples_leaf=1, rng=StableRNG(1))
        for model in (tree, EvoTreeRegressor(nrounds=200, eta=0.3, seed=1)),
            s in (Recursive(), Direct(7)), tf in (NoTransform(), Difference(1))
            fc = Forecaster(model; features=FeatureSet(Fourier(7, 1)), strategy=s,
                            freq=Day(1), id=:unique_id, transform=tf)
            out = forecast(fit(fc, week), 7)
            @test maximum(abs.(out.y_hat .- wk.(out.ds))) < (model === tree ? 1e-9 : 1e-4)
        end
        # Month-end series keep whole-month offsets.
        me = [Date(2021, 1, 31), Date(2021, 3, 31), Date(2021, 8, 31)]
        mdates = [[lastdayofmonth(d + Month(i)) for i in 0:(27 - 3k)]
                  for (k, d) in enumerate(me)]
        monthly = (unique_id=reduce(vcat, [fill("m$k", length(t))
                                          for (k, t) in enumerate(mdates)]),
                   ds=reduce(vcat, mdates), y=ones(sum(length, mdates)))
        fm = fit(Forecaster(TestModels.EchoColumn(:fourier_12_0_cos_1);
                            features=FeatureSet(Fourier(12, 1)), freq=Month(1),
                            id=:unique_id), monthly)
        out = forecast(fm, 2)
        @test out.y_hat ≈ [cos(2π * (month(d) - 1) / 12) for d in out.ds]
    end

    @testset "Fourier phase is a property of the date (Recursive, Direct(3))" begin
        o = Date(2022, 1, 1)
        day(d) = Float64(Dates.value(d - o))
        phase(d) = sin(2π * mod(Dates.value(d - o), 7) / 7)
        function ragged(spans)
            ds = [collect(d0:Day(1):d0 + Day(n - 1)) for (d0, n) in spans]
            ids = reduce(vcat, [fill("s$k", n) for (k, (_, n)) in enumerate(spans)])
            return (unique_id=ids, ds=reduce(vcat, ds), y=day.(reduce(vcat, ds)))
        end
        spans = [(o, 50), (Date(2022, 1, 4), 50), (Date(2022, 1, 11), 40)]
        # The origin skips short series.
        short = [(Date(2021, 12, 20), 3); spans]
        for (sp, feats, mh) in ((spans, [Fourier(7, 2)], 0),
                                (short, [Lag(5), Fourier(7, 2)], 5)),
            strat in (Recursive(), Direct(3))
            data = ragged(sp)
            mk2(model) = Forecaster(model; features=FeatureSet(feats...), strategy=strat,
                                    freq=Day(1), id=:unique_id)
            fitit(model) = length(sp) == 4 ?
                (@test_logs (:warn, r"skipping 1 series with fewer than 6 rows") fit(
                    mk2(model), data)) : fit(mk2(model), data)
            out = forecast(fitit(TestModels.EchoColumn(:fourier_7_0_sin_1)), 3)
            @test out.y_hat ≈ phase.(out.ds)
            for d in unique(out.ds)
                vals = out.y_hat[out.ds .== d]
                @test all(==(first(vals)), vals)
            end
            # Direct anchors time columns to the target row.
            fitted = fitit(TestModels.KeepTrain())
            for m in fitted.models
                X, y = m.fitresult.X, m.fitresult.y
                @test X.fourier_7_0_sin_1 ≈ sin.(2π .* mod.(y, 7) ./ 7)
                @test X.fourier_7_0_cos_2 ≈ cos.(2π .* mod.(2 .* y, 7) ./ 7)
            end
            strat isa Recursive || continue
            # Rows are ordered by series.
            X, y = only(fitted.models).fitresult.X, only(fitted.models).fitresult.y
            r2 = (50 - mh + 1):(2 * (50 - mh))
            d2 = Date(2022, 1, 4) .+ Day.(mh:49)
            @test y[r2] == day.(d2)
            @test X.fourier_7_0_sin_1[r2] ≈ phase.(d2)
        end
    end

    @testset "Fourier under a row-consuming transform, on a panel" begin
        n = 50
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        ds2 = ds .+ Day(3)
        data = (unique_id=[fill("a", n); fill("b", n)], ds=[ds; ds2],
                y=[20 .+ Float64.(1:n); 30 .+ 2 .* Float64.(1:n)])
        sin7(steps) = sin.(2π .* mod.(steps, 7) ./ 7)
        cos7(steps) = cos.(2π .* mod.(steps, 7) ./ 7)
        spec(m, strat) = Forecaster(m; features=FeatureSet(Fourier(7, 1)), strategy=strat,
                                    freq=Day(1), id=:unique_id, transform=LogRatio(3))
        # Row steps count from the transformed table.
        keep = fit(spec(TestModels.KeepTrain(), Recursive()), data)
        @test [s.step0 for s in keep.series] == [0, 3]
        X = only(keep.models).fitresult.X
        steps = [0:46; 3:49]
        @test X.fourier_7_0_sin_1[1] == 0.0 && X.fourier_7_0_cos_1[1] == 1.0
        @test X.fourier_7_0_sin_1 ≈ sin7(steps) && X.fourier_7_0_cos_1 ≈ cos7(steps)
        for (h, m) in enumerate(fit(spec(TestModels.KeepTrain(), Direct(3)), data).models)
            hs = [(h - 1):46; (3 + h - 1):49]
            @test m.fitresult.X.fourier_7_0_sin_1 ≈ sin7(hs)
        end
        # Each series continues after its own transformed rows.
        for strat in (Recursive(), Direct(3))
            fitted = fit(spec(TestModels.KeepPredict(Ref(Any[])), strat), data)
            forecast(fitted, 3)
            calls = reduce(vcat, [m.model.calls[] for m in fitted.models])
            @test length(calls) == 3
            for (s, c) in enumerate(calls)
                @test c.fourier_7_0_sin_1 ≈ sin7([46 + s, 49 + s])
                @test c.fourier_7_0_cos_1 ≈ cos7([46 + s, 49 + s])
            end
        end
    end

    @testset "Fourier on Month(1) freq, ragged panel" begin
        o = Date(2020, 3, 1)
        spans = [(o, 30), (Date(2020, 8, 1), 24), (Date(2021, 1, 1), 20)]
        ms = [collect(d0:Month(1):d0 + Month(n - 1)) for (d0, n) in spans]
        ids = reduce(vcat, [fill("m$k", n) for (k, (_, n)) in enumerate(spans)])
        data = (unique_id=ids, ds=reduce(vcat, ms), y=ones(sum(last, spans)))
        phase(d) = sin(2π * mod(12 * (year(d) - 2020) + month(d) - 3, 12) / 12)
        for strat in (Recursive(), Direct(3))
            fc = Forecaster(TestModels.EchoColumn(:fourier_12_0_sin_1);
                            features=FeatureSet(Fourier(12, 1)), strategy=strat,
                            freq=Month(1), id=:unique_id)
            out = forecast(fit(fc, data), 3)
            @test length(out.ds) == 9
            @test out.y_hat ≈ phase.(out.ds)
            for d in unique(out.ds)
                vals = out.y_hat[out.ds .== d]
                @test all(==(first(vals)), vals)
            end
        end
    end

    @testset "ragged start dates: fit, forecast, Direct and backtest" begin
        starts = [Date(2022, 1, 1), Date(2022, 1, 11), Date(2022, 1, 21)]
        rag = makepanel(lens=(60, 50, 40), starts=starts)     # all end on 2022-03-01
        actual = Dict((rag.unique_id[i], rag.ds[i]) => rag.y[i] for i in eachindex(rag.y))
        fourier = FeatureSet(Lag(1), Fourier(7, 1))
        for s in (Recursive(), Direct(5))
            f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=s), rag)
            @test [st.t_start for st in f.series] == starts
            @test [st.n_train for st in f.series] == [60, 50, 40]
            out = forecast(f, 3)
            @test out.ds == repeat(Date(2022, 3, 1) .+ Day.(1:3), 3)
            @test out.y_hat == repeat([160.0, 60.0, 90.0], inner=3)
            # Fourier counts from the earliest start, so dates share values.
            fe = fit(mk(TestModels.EchoColumn(:fourier_7_0_sin_1), fourier; strat=s), rag)
            got = forecast(fe, 3).y_hat
            @test got ≈ repeat([sin(2π * (59 + i) / 7) for i in 1:3], 3)
            # s3 starts after the first origin and joins at the second.
            r = backtest(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=s), rag;
                         horizon=5, initial=15, step=10)
            @test unique(r.forecasts.origin) == Date(2022, 1, 15) .+ Day.(0:10:40)
            first_fold = r.forecasts.origin .== Date(2022, 1, 15)
            @test unique(r.forecasts.unique_id[first_fold]) == ["s1", "s2"]
            @test count(==("s3"), r.forecasts.unique_id) == 4 * 5
            @test all(yh == actual[(id, o)] for (yh, id, o) in
                      zip(r.forecasts.y_hat, r.forecasts.unique_id, r.forecasts.origin))
            @test all(==(3.0), r.scores.value[r.scores.metric .== :mae])
        end
    end

    @testset "backtest folds that cannot be cut or scored" begin
        big = makepanel(lens=(60, 50, 55))
        # s2 starts after s1 ends: origin 50 has nothing to score
        s1 = collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19))
        s2 = collect(Date(2022, 2, 20):Day(1):Date(2022, 3, 1))
        apart = (unique_id=[fill("s1", 50); fill("s2", 10)], ds=[s1; s2],
                 y=Float64.(1:60))
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fc = mk(m, FeatureSet(Lag(1)))
            @test_throws ArgumentError backtest(fc, big; horizon=5, initial=58)
            @test_throws "no complete backtest folds: the panel spans 60 distinct " *
                         "timestamps, but the first fold needs initial + horizon = 63. " *
                         "Provide more history" backtest(fc, big; horizon=5, initial=58)
            @test_throws ArgumentError backtest(fc, apart; horizon=5, initial=50,
                                                step=100)
            @test_throws "backtest fold 1 (origin 2022-02-19) produced no forecast that " *
                         "lines up with an actual: no series trained on data up to the " *
                         "origin has data after it. Choose initial and step so each " *
                         "origin falls inside the data of a series." backtest(
                fc, apart; horizon=5, initial=50, step=100)
        end
    end

    @testset "backtest with Exogenous on ragged and phase-offset panels" begin
        rng = StableRNG(11)
        # s1 ends before the last origins, s3 inside a window.
        daily = makepanel(lens=(50, 60, 56))
        daily = merge(daily, (y=daily.y .+ randn(rng, length(daily.y)),
                              promo=Float64.(eachindex(daily.y))))
        mondays = collect(Date(2024, 1, 1):Week(1):Date(2024, 7, 22))
        weekly = (unique_id=repeat(["a", "b"], inner=30),
                  ds=vcat(mondays, mondays .+ Day(3)),
                  y=repeat([10.0, 20.0], inner=30) .+ randn(rng, 60),
                  promo=Float64.(1:60))
        cases = ((daily, Day(1), 5, 30, 4), (weekly, Week(1), 2, 20, 3))
        for (data, freq, h, init, stp) in cases,
            m in (EvoTreeRegressor(nrounds=10),
                  DecisionTreeRegressor(max_depth=3, rng=StableRNG(1))),
            s in (Recursive(), Direct(h))
            bt(fs) = backtest(Forecaster(m; features=fs, strategy=s, freq=freq,
                                         id=:unique_id),
                              data; horizon=h, initial=init, step=stp)
            plain = bt(FeatureSet(Lag(1)))
            exog = bt(FeatureSet(Lag(1), Exogenous(:promo)))
            @test exog.forecasts.origin == plain.forecasts.origin
            @test exog.forecasts.step == plain.forecasts.step
            @test exog.forecasts.unique_id == plain.forecasts.unique_id
            @test exog.forecasts.ds == plain.forecasts.ds
        end
        # EchoColumn shows each step reads its own (id, time) row.
        for (data, freq, h, init, stp) in cases
            fc = Forecaster(TestModels.EchoColumn(:promo); freq=freq, id=:unique_id,
                            features=FeatureSet(Lag(1), Exogenous(:promo)))
            r = backtest(fc, data; horizon=h, initial=init, step=stp)
            promo = Dict((data.unique_id[i], data.ds[i]) => data.promo[i]
                         for i in eachindex(data.promo))
            @test r.forecasts.y_hat == [promo[(r.forecasts.unique_id[j], r.forecasts.ds[j])]
                                    for j in eachindex(r.forecasts.ds)]
        end
    end

    @testset "tune drives a panel through backtest" begin
        big = makepanel(lens=(60, 50, 55))
        base = mk(TestModels.MeanModel(), FeatureSet(Lag(1)))
        res = tune(base, big; grid=(features=[FeatureSet(Lag(1)),
                                              FeatureSet(Lag(1), Lag(7))],),
                   horizon=5, initial=40, step=10, metric=mae)
        @test length(res.table.mean_score) == 2
        @test all(!ismissing, res.table.mean_score)
        @test MachineLearningForecast.ispanel(res.best)
        @test nseries(res.best_fitted) == 3
        # a misnamed time column gets backtest's message
        renamed = (unique_id=big.unique_id, when=big.ds, y=big.y)
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            grid = (model=[m],)
            @test_throws ArgumentError tune(mk(m, FeatureSet(Lag(1))), renamed;
                                            grid=grid, horizon=5, initial=40)
            @test_throws "time column :ds not found in the data. Available columns: " *
                         "unique_id, when, y" tune(mk(m, FeatureSet(Lag(1))), renamed;
                                                   grid=grid, horizon=5, initial=40)
        end
    end

    @testset "real learners fit, forecast and backtest a panel" begin
        big = makepanel(lens=(60, 50, 55))
        noisy = merge(big, (y=big.y .+ randn(StableRNG(6), length(big.y)),))
        fs = FeatureSet(Lag(1), Lag(7), RollingMean(3), Calendar(:dayofweek))
        for m in (EvoTreeRegressor(nrounds=20, seed=123),
                  DecisionTreeRegressor(max_depth=4, rng=StableRNG(2))),
            s in (Recursive(), Direct(4))
            fc = mk(m, fs; strat=s)
            f = fit(fc, noisy)
            @test nseries(f) == 3 && length(f.models) == (s isa Direct ? 4 : 1)
            out = forecast(f, 4)
            @test out.unique_id == repeat(["s1", "s2", "s3"], inner=4)
            @test out.ds == reduce(vcat, [st.t_last .+ Day.(1:4) for st in f.series])
            @test all(isfinite, out.y_hat)
            @test forecast(fit(fc, noisy), 4) == out
            # s1 ends near 160, s3 near 105, s2 near 60.
            @test all(out.y_hat[1:4] .> out.y_hat[9:12] .> out.y_hat[5:8])
            r = backtest(fc, noisy; horizon=4, initial=40, step=10)
            ref = backtest(mk(TestModels.LinAR(1.0, 0.0), fs; strat=s), noisy;
                           horizon=4, initial=40, step=10)
            @test r.forecasts.origin == ref.forecasts.origin
            @test r.forecasts.ds == ref.forecasts.ds
            @test r.forecasts.unique_id == ref.forecasts.unique_id
            @test r.forecasts.y == ref.forecasts.y
            @test all(isfinite, r.forecasts.y_hat) && all(isfinite, r.scores.value)
        end
    end

    @testset "fit, backtest and tune leave the model prototype unchanged" begin
        big = makepanel(lens=(60, 50, 55))
        noisy = merge(big, (y=big.y .+ randn(StableRNG(5), length(big.y)),))
        fs = FeatureSet(Lag(1), Lag(2), Lag(7), RollingMean(3))
        # MLJ's == ignores RNG state, so compare every field.
        same(a, b) = all(k -> isequal(getfield(a, k), getfield(b, k)),
                         fieldnames(typeof(a)))
        for m in (DecisionTreeRegressor(n_subfeatures=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=10, rowsample=0.5, colsample=0.5)),
            s in (Recursive(), Direct(3))
            before = deepcopy(m)
            fc = mk(m, fs; strat=s)
            first_fit = forecast(fit(fc, noisy), 3).y_hat
            @test same(m, before)
            @test forecast(fit(fc, noisy), 3).y_hat == first_fit
            backtest(fc, noisy; horizon=3, initial=40, step=10)
            @test same(m, before)
            r = tune(fc, noisy; grid=(features=[fs, fs, fs],), horizon=3, initial=40,
                     step=10)
            @test same(m, before)
            @test allequal(r.table.mean_score)
        end
    end

    @testset "show displays panel specs and fitted panels" begin
        for m in (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1))),
            s in (Recursive(), Direct(2))
            fc = mk(m, FeatureSet(Lag(1)); strat=s)
            @test occursin("id=:unique_id", sprint(show, fc))
            f = fit(fc, panel)
            for txt in (sprint(show, f), sprint(show, MIME"text/plain"(), f))
                @test occursin("trained on 3 series", txt)
                @test occursin(string(s), txt)
                @test occursin(string(nameof(typeof(m))), txt)
            end
        end
        single = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                            freq=Day(1))
        @test !occursin("id=", sprint(show, single))
    end

    @testset "stacking the series' rows takes memory linear in the rows" begin
        Xs = NamedTuple[(a=rand(StableRNG(k), 10), b=rand(StableRNG(k + 1), 10))
                        for k in 1:2000]
        stack = MachineLearningForecast._vcat_frames
        @test stack(Xs) == (a=reduce(vcat, [X.a for X in Xs]),
                            b=reduce(vcat, [X.b for X in Xs]))
        stack(Xs)
        # The output is 2 columns of 20000 rows; 4x is headroom.
        @test (@allocated stack(Xs)) < 4 * 2 * 20_000 * sizeof(Float64)
        big = makepanel(lens=fill(30, 2000), levels=fill(1.0, 2000),
                        starts=fill(Date(2022, 1, 1), 2000))
        fc = mk(TestModels.MeanModel(), FeatureSet(Lag(1)))
        f = fit(fc, big)
        @test nseries(f) == 2000
        @test only(f.models).fitresult == 1.0 + sum(2:30) / 29
        # A quadratic stack of 2000 series allocates about 1 GB.
        @test (@allocated fit(fc, big)) < 100 * 2^20
    end

    @testset "panel Direct gathers each step's rows in one pass" begin
        k = 2000
        big = makepanel(lens=fill(30, k), levels=fill(1.0, k),
                        starts=fill(Date(2022, 1, 1), k))
        feats = FeatureSet(Lag.(1:7)..., RollingMean(7), Calendar(:dayofweek))
        dir = mk(TestModels.MeanModel(), feats; strat=Direct(5))
        rec = mk(TestModels.MeanModel(), feats)
        # Step i trains on targets at t = 7 + i, ..., 30.
        @test [t.fitresult for t in fit(dir, big).models] == [1 + (37 + i) / 2 for i in 1:5]
        fit(rec, big)
        extra = (@allocated fit(dir, big)) - (@allocated fit(rec, big))
        steps = k * sum(24 - i for i in 1:5) * 10 * sizeof(Float64)
        # Slicing per series first allocated the steps twice.
        @test extra < 1.6 * steps
    end

    @testset "single-series behaviour is unchanged" begin
        df = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19)), y=Float64.(1:50))
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        strategy=Recursive(), freq=Day(1))
        f = fit(fc, df)
        @test nseries(f) == 1
        @test f.y_history == Float64.(1:50)
        @test f.t_last == Date(2022, 2, 19) && f.n_train == 50
        @test keys(forecast(f, 3)) == (:ds, :y_hat)
    end
    @testset "a single series is a panel of one" begin
        n = 70
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        promo = Float64.(mod.(1:n, 5) .== 0)
        yv = 50 .+ 2 .* sin.(1:n) .+ 0.1 .* (1:n)
        stbl = (ds=ds, y=yv, promo=promo)
        ptbl = (unique_id=fill("a", n), ds=ds, y=yv, promo=promo)
        fds = ds[end] .+ Day.(1:7)
        fpro = Float64.(mod.(n + 1:n + 7, 5) .== 0)
        sfut = (ds=fds, promo=fpro)
        pfut = (unique_id=fill("a", 7), ds=fds, promo=fpro)
        feats = FeatureSet(Lag(1), RollingMean(3), Diff(1), Calendar(:dayofweek),
                           Fourier(7, 1), Exogenous(:promo),
                           CustomFeature(:hlen, length, 0))
        mkp(model, strat, tr) = Forecaster(model; features=feats, strategy=strat,
                                           freq=Day(1), id=:unique_id, transform=tr)
        mks(model, strat, tr) = Forecaster(model; features=feats, strategy=strat,
                                           freq=Day(1), transform=tr)
        transforms = (NoTransform(), Difference(2),
                      TransformChain(LastScale(4), LogRatio(3)))
        for strat in (Recursive(), Direct(3)), tr in transforms
            h = strat isa Direct ? 3 : 7
            fs = fit(mks(TestModels.KeepTrain(), strat, tr), stbl)
            fp = fit(mkp(TestModels.KeepTrain(), strat, tr), ptbl)
            @test length(fs.models) == length(fp.models)
            @test all(isequal(a.fitresult, b.fitresult)
                      for (a, b) in zip(fs.models, fp.models))
            sa, sb = only(fs.series), only(fp.series)
            @test sa.id === nothing && sb.id == "a"
            for fld in (:y_history, :t_last, :t_start, :n_train, :z_history,
                        :transform_state, :step0)
                @test isequal(getfield(sa, fld), getfield(sb, fld))
            end
            @test sa.step0 == 0
            for model in (TestModels.LinAR(0.9, 1.0), EvoTreeRegressor(nrounds=5),
                          DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
                a = forecast(fit(mks(model, strat, tr), stbl), h; future=sfut)
                b = forecast(fit(mkp(model, strat, tr), ptbl), h; future=pfut)
                @test isequal(a.y_hat, b.y_hat) && isequal(a.ds, b.ds)
            end
        end
    end

    @testset "panel fit errors keep their order and prefix" begin
        fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        diff = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                          transform=Difference(1), freq=Day(1), id=:unique_id)
        bad = makepanel(lens=(10, 10))
        bad.y[end - 1] = 1e308; bad.y[end] = -1e308
        err = try fit(diff, bad) catch e; e end
        @test err isa ArgumentError
        @test startswith(err.msg, "series unique_id=\"s2\": transform=Difference(1) " *
                                  "turned the target into a non-finite value")
        err = try fit(Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                                 transform=Difference(1), freq=Day(1)),
                      (ds=bad.ds[11:20], y=bad.y[11:20])) catch e; e end
        @test err isa ArgumentError
        @test startswith(err.msg, "transform=Difference(1) turned the target into a " *
                                  "non-finite value")
        unsorted = (unique_id=["a", "a", "a"],
                    ds=[Date(2022, 1, 3), Date(2022, 1, 1), Date(2022, 1, 2)])
        @test_throws "target column :y not found" fit(fc, unsorted)
        @test_throws "id column :unique_id not found" fit(
            fc, (ds=unsorted.ds, y=[1.0, 2, 3]))
        for tr in (Difference(5), LogRatio(7))
            n = tr isa Difference ? 5 : 7
            one = makepanel(lens=(n,))
            fct = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                             transform=tr, freq=Day(1), id=:unique_id)
            @test_throws "no series has enough rows to train on" fit(fct, one)
        end
    end

    @testset "fit leaves the table and the model prototype untouched" begin
        model = TestModels.LinAR(0.5, 1.0)
        keep = deepcopy(model)
        for tbl in (panel, makepanel(lens=(30,)))
            copy_ = deepcopy(tbl)
            fit(mk(model, FeatureSet(Lag(1), RollingMean(3)); strat=Direct(2)), tbl)
            @test tbl == copy_ && model == keep
        end
    end

    @testset "skip warning comes before the models' clean! warnings" begin
        fc = mk(TestModels.Clamped(0), FeatureSet(Lag(7)); strat=Direct(2))
        @test_logs((:warn, r"skipping 1 series with fewer than 8 rows"),
                   (:warn, "k must be ≥ 1; resetting k to 1."),
                   (:warn, "k must be ≥ 1; resetting k to 1."),
                   fit(fc, makepanel(lens=(40, 3, 35))))
    end

    @testset "Direct without enough rows throws before any model trains" begin
        tiny = makepanel(lens=(3, 3, 3))
        fc = mk(TestModels.Clamped(0), FeatureSet(Lag(1)); strat=Direct(5))
        @test_logs min_level=Logging.Warn @test_throws(
            "not enough data for Direct(5): no series has enough rows to train the " *
            "step-3 model. Shorten max_horizon, provide longer series, or reduce " *
            "lags/windows.", fit(fc, tiny))
        fcs = Forecaster(TestModels.Clamped(0); features=FeatureSet(Lag(2)),
                         strategy=Direct(4), freq=Day(1))
        short = makepanel(lens=(4,))
        short = (ds=short.ds, y=short.y)
        @test_logs min_level=Logging.Warn @test_throws(
            "not enough data for Direct(4): after dropping the feature set's 2 " *
            "history rows, only 2 training rows remain", fit(fcs, short))
    end

    @testset "Direct warns once when a panel series is too short for a step model" begin
        kt = TestModels.KeepTrain
        fcd(model=kt(); lag=1, h=5) = mk(model, FeatureSet(Lag(lag)); strat=Direct(h))
        msg = "Direct(5): 1 series with fewer than 5 training rows is left out of the " *
              "step models beyond their row count: unique_id=\"s2\" (3 training rows)"

        @testset "one short series: the message, and the numbers do not change" begin
            f = @test_logs (:warn, msg) fit(fcd(), makepanel(lens=(40, 4, 35)))
            lens, lv = (39, 3, 34), (100.0, 10.0, 50.0)
            for i in 1:5
                keep = [k for k in 1:3 if lens[k] >= i]
                @test length(keep) == (i <= 3 ? 3 : 2)
                x = reduce(vcat, [lv[k] .+ Float64.(1:(lens[k] - i + 1)) for k in keep])
                y = reduce(vcat, [lv[k] .+ Float64.((1 + i):(lens[k] + 1)) for k in keep])
                r = f.models[i].fitresult
                @test only(values(r.X)) == x
                @test r.y == y
            end
        end

        @testset "a series that is skipped is announced first" begin
            p = makepanel(lens=(40, 4, 35, 1), levels=(100.0, 10.0, 50.0, 5.0),
                            starts=fill(Date(2022, 1, 1), 4))
            @test_logs((:warn, r"skipping 1 series with fewer than 2 rows"),
                       (:warn, msg), fit(fcd(), p))
            @test_logs((:warn, r"skipping 1 series with fewer than 2 rows"),
                       (:warn, msg), (:warn, "k must be ≥ 1; resetting k to 1."),
                       (:warn, "k must be ≥ 1; resetting k to 1."),
                       (:warn, "k must be ≥ 1; resetting k to 1."),
                       (:warn, "k must be ≥ 1; resetting k to 1."),
                       (:warn, "k must be ≥ 1; resetting k to 1."),
                       fit(fcd(TestModels.Clamped(0)), p))
        end

        @testset "seven short series: five are named, then an ellipsis" begin
            p = makepanel(lens=(40, fill(4, 7)...), levels=Tuple(1.0:8.0),
                          starts=fill(Date(2022, 1, 1), 8))
            names = join(("unique_id=\"s$k\" (3 training rows)" for k in 2:6), ", ")
            @test_logs (:warn,
                "Direct(5): 7 series with fewer than 5 training rows are left out of " *
                "the step models beyond their row count: $names, …") fit(fcd(), p)
            p6 = makepanel(lens=(40, fill(4, 5)...), levels=Tuple(1.0:6.0),
                           starts=fill(Date(2022, 1, 1), 6))
            @test_logs (:warn,
                "Direct(5): 5 series with fewer than 5 training rows are left out of " *
                "the step models beyond their row count: $names") fit(fcd(), p6)
        end

        @testset "Int ids, a longer lag, one row short of the last step" begin
            p = makepanel(lens=(40, 7, 35))
            p = (unique_id=[parse(Int, s[2:end]) for s in p.unique_id], ds=p.ds, y=p.y)
            @test_logs (:warn,
                "Direct(5): 1 series with fewer than 5 training rows is left out of the " *
                "step models beyond their row count: unique_id=2 (4 training rows)"
            ) fit(fcd(; lag=3), p)
        end

        @testset "no warning when nothing is left out" begin
            short = makepanel(lens=(40, 4, 35))
            @test_logs min_level=Logging.Warn fit(mk(kt(), FeatureSet(Lag(1))), short)
            # Exactly H training rows still reach the last step.
            even = makepanel(lens=(40, 6, 35))
            f = @test_logs min_level=Logging.Warn fit(fcd(; h=5), even)
            @test length(f.models) == 5
            @test_logs min_level=Logging.Warn fit(fcd(; h=3), short)
            # A single series has no other series to be left out of.
            one = makepanel(lens=(40,))
            one = (ds=one.ds, y=one.y)
            fcs = Forecaster(kt(); features=FeatureSet(Lag(1)), strategy=Direct(5),
                             freq=Day(1))
            @test_logs min_level=Logging.Warn fit(fcs, one)
            # All short: the error comes first, with no warning.
            @test_logs min_level=Logging.Warn @test_throws(
                "not enough data for Direct(5)", fit(fcd(), makepanel(lens=(3, 3, 3))))
        end

        @testset "once per fit, not once per step model, with several models" begin
            for m in (EvoTreeRegressor(nrounds=3),
                      DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
                f = @test_logs (:warn, msg) fit(fcd(m), makepanel(lens=(40, 4, 35)))
                @test length(f.models) == 5
            end
        end

        @testset "once per fold in a panel backtest" begin
            p = makepanel(lens=(60, 30, 55), starts=[Date(2022, 1, 1),
                          Date(2022, 1, 31), Date(2022, 1, 1)])
            fc = fcd(TestModels.LinAR(1.0, 0.0); h=20)
            wrn = (:warn, "Direct(20): 1 series with fewer than 20 training rows is " *
                          "left out of the step models beyond their row count: " *
                          "unique_id=\"s2\" (9 training rows)")
            wrn2 = (:warn, "Direct(20): 1 series with fewer than 20 training rows is " *
                           "left out of the step models beyond their row count: " *
                           "unique_id=\"s2\" (19 training rows)")
            r = @test_logs wrn wrn2 backtest(fc, p; horizon=5, initial=40, step=10)
            @test unique(r.forecasts.fold) == [1, 2]
        end
    end

    @testset "the facade throws on every panel, even of one series" begin
        one = makepanel(lens=(40,), levels=(100.0,), starts=[Date(2022, 1, 1)])
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), one)
        @test nseries(f) == 1
        for name in (:y_history, :t_last, :t_start, :n_train, :z_history,
                     :transform_state)
            @test_throws("single-series state, but this forecaster was fit on 1 series",
                         getproperty(f, name))
        end
        @test only(f.series).n_train == 40
    end

    @testset "single Direct fit slices each step once" begin
        n = 40_000
        big = (ds=collect(Date(2000, 1, 1):Day(1):Date(2000, 1, 1) + Day(n - 1)),
               y=Float64.(1:n))
        sfs = FeatureSet(Lag.(1:3)...)
        dir = Forecaster(TestModels.MeanModel(); features=sfs, strategy=Direct(5),
                         freq=Day(1))
        rec = Forecaster(TestModels.MeanModel(); features=sfs, freq=Day(1))
        fit(dir, big); fit(rec, big)
        extra = (@allocated fit(dir, big)) - (@allocated fit(rec, big))
        steps = sum(n - 3 - i + 1 for i in 1:5) * 4 * sizeof(Float64)
        @test extra < 1.15 * steps
    end

    @testset "series groups: strict for one series, sorted for a panel" begin
        M = MachineLearningForecast
        n = 20
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        tbl = (ds=ds, y=Float64.(1:n))
        fcs = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                         freq=Day(1))
        g = M._series_groups(fcs, tbl)
        @test length(g) == 1 && g[1][1] === nothing && g[1][2] === tbl && g[1][3] == 1:n
        @test_throws "time column :ds is not sorted" M._series_groups(
            fcs, (ds=reverse(ds), y=tbl.y))
        shuf = shuffle(StableRNG(3), 1:length(panel.y))
        sp = map(c -> c[shuf], panel)
        gp = M._series_groups(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), sp)
        @test [x[1] for x in gp] == unique(sp.unique_id)
        for (id, sub, rows) in gp
            @test issorted(sub.ds) && sp.ds[rows] == sub.ds
            @test sp.unique_id[rows] == sub.unique_id
            @test sub.y == sp.y[rows]
        end
    end
    @testset "ids of mixed types and per-series states" begin
        base = makepanel(lens=(30, 30), levels=(100.0, 10.0))
        withids(a, b) = (unique_id=Any[k == "s1" ? a : b for k in base.unique_id],
                         ds=base.ds, y=base.y)
        fc = mk(TestModels.LinAR(0.9, 1.0), FeatureSet(Lag(1)))

        @testset "a String and an Int id" begin
            out = forecast(fit(fc, withids("a", 2)), 2)
            @test isequal(out.unique_id, Any["a", "a", 2, 2])
            @test eltype(out.unique_id) == Any
        end

        @testset "an Int and a Float64 id keep their types" begin
            out = forecast(fit(fc, withids(1, 2.0)), 2)
            @test isequal(out.unique_id, Real[1, 1, 2.0, 2.0])
            @test eltype(out.unique_id) == Real
            @test typeof.(out.unique_id) == [Int, Int, Float64, Float64]
            out = forecast(fit(fc, withids(1, 2.5)), 2)
            @test isequal(out.unique_id, Real[1, 1, 2.5, 2.5])
        end

        @testset "a transform whose state type varies per series" begin
            tr = TestModels.VaryingState()
            fct = Forecaster(TestModels.LinAR(0.9, 1.0); features=FeatureSet(Lag(1)),
                             freq=Day(1), id=:unique_id, transform=tr)
            f = fit(fct, base)
            @test [s.transform_state for s in f.series] == Any[nothing, 5.0]
            @test [s.z_history[1] for s in f.series] == [101.0, 6.0]
            out = forecast(f, 2)
            solo = fit(Forecaster(TestModels.LinAR(0.9, 1.0);
                                  features=FeatureSet(Lag(1)), freq=Day(1),
                                  transform=tr),
                       (ds=base.ds[31:60], y=base.y[31:60]))
            @test out.y_hat[3:4] == forecast(solo, 2).y_hat
        end
    end

    @testset "forecast checks keep their messages and order, single and panel" begin
        n = 40
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        stbl = (ds=ds, y=50 .+ 0.1 .* (1:n), promo=Float64.(1:n))
        pe = merge(makepanel(), (promo=Float64.(1:length(panel.y)),))
        feats = FeatureSet(Lag(1), Exogenous(:promo))
        fs = fit(Forecaster(TestModels.EchoColumn(:promo); features=feats,
                            strategy=Direct(3), freq=Day(1)), stbl)
        fp = fit(mk(TestModels.EchoColumn(:promo), feats; strat=Direct(3)), pe)
        g(h) = ds[end] .+ Day.(1:h)
        nofut = "features contain Exogenous(:promo) but forecast() was called without " *
                "`future`. Pass a table with columns "
        @test_throws nofut * "(:ds, :promo) covering $(g(3)[1]) … $(g(3)[3])." forecast(
            fs, 3)
        @test_throws nofut * "(:unique_id, :ds, :promo) covering all 3 series over " *
                     "their 3 forecast steps." forecast(fp, 3)
        # The exogenous error comes before the horizon error.
        @test_throws nofut * "(:ds, :promo) covering $(g(5)[1]) … $(g(5)[5])." forecast(
            fs, 5)
        @test_throws nofut * "(:unique_id, :ds, :promo) covering all 3 series over " *
                     "their 5 forecast steps." forecast(fp, 5)
        horizon = "strategy=Direct(3) was fit with max_horizon=3 but forecast(h=5) was " *
                  "requested. Refit with Direct(5) or use Recursive()."
        sfut = (ds=g(5), promo=Float64.(1:5))
        @test_throws horizon forecast(fs, 5; future=sfut)
        short = (ds=g(3), promo=Float64.(1:3))
        @test_throws "`future` is missing 2 of the 5 timestamps" forecast(
            fs, 5; future=short)

        # Direct warns about a `future` it cannot use.
        ignored = "`future` was passed but the feature set has no Exogenous features; " *
                  "it will be ignored."
        plains = fit(Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                                strategy=Direct(3), freq=Day(1)), stbl)
        plainp = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=Direct(3)),
                     panel)
        nd = (unique_id=["s1"], ds=[Date(2022, 3, 1)], promo=[1.0])
        @test_logs (:warn, ignored) forecast(plains, 2; future=sfut)
        @test_logs (:warn, ignored) forecast(plainp, 2; future=nd)
        @test_logs (:warn, ignored) @test_throws(horizon, forecast(plains, 5; future=sfut))
        @test_logs (:warn, ignored) @test_throws(horizon, forecast(plainp, 5; future=nd))
        @test_throws horizon forecast(plains, 5)
        @test_throws horizon forecast(plainp, 5)

        # Single checks columns, rows, timestamps, then values.
        rec = fit(Forecaster(TestModels.EchoColumn(:promo); features=feats,
                             freq=Day(1)), stbl)
        gs = g(3)
        nopromo = (ds=gs[1:1],)
        @test_throws "`future` is missing column :promo; it must contain the time " *
                     "column :ds and the exogenous column :promo." forecast(
            rec, 3; future=nopromo)
        dupabsent = (ds=[gs[1], gs[1]], promo=[1.0, 2.0])
        @test_throws "`future` has duplicate timestamps in column :ds; first duplicate " *
                     "at $(gs[1])." forecast(rec, 3; future=dupabsent)
        holed = (ds=gs[1:2], promo=[missing, 2.0])
        @test_throws "`future` is missing 1 of the 3 timestamps required for the " *
                     "forecast grid $(gs[1]) … $(gs[3]); first missing: $(gs[3]). " *
                     "Provide exogenous values for every future step." forecast(
            rec, 3; future=holed)
        @test_throws "`future` has a missing value in exogenous column :promo at time " *
                     "$(gs[1]). Provide complete exogenous values for every future " *
                     "step." forecast(rec, 3; future=(ds=gs, promo=[missing, 2.0, 3.0]))

        # Panel checks series by series, step by step.
        recp = fit(mk(TestModels.EchoColumn(:promo), feats), pe)
        s1 = recp.series[1].t_last .+ Day.(1:2)
        part = (unique_id=["s1", "s1"], ds=s1, promo=[missing, 1.0])
        @test_throws "`future` has a missing value in exogenous column :promo for " *
                     "unique_id=\"s1\" at $(s1[1]). Provide complete exogenous values " *
                     "for every series at every forecast step." forecast(
            recp, 2; future=part)
        part2 = (unique_id=["s1", "s1"], ds=s1, promo=[1.0, 2.0])
        @test_throws "`future` has no row for unique_id=\"s2\" at " forecast(
            recp, 2; future=part2)
    end

    @testset "a single series forecasts as a panel of one, with a shuffled future" begin
        n = 50
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        yv = 50 .+ 2 .* sin.(1:n)
        stbl = (ds=ds, y=yv, promo=Float64.(1:n))
        ptbl = (unique_id=fill("a", n), ds=ds, y=yv, promo=Float64.(1:n))
        feats = FeatureSet(Lag(1), Exogenous(:promo))
        ahead = ds[end] .+ Day.(1:12)
        p = shuffle(StableRNG(5), 1:12)
        vals = 100.0 .+ (1:12)
        sfut = (ds=ahead[p], promo=vals[p], other=fill(NaN, 12))
        pfut = (unique_id=fill("a", 12), ds=ahead[p], promo=vals[p], other=fill(NaN, 12))
        for strat in (Recursive(), Direct(4)), tr in (NoTransform(), LastScale(3))
            h = strat isa Direct ? 4 : 9
            fs = fit(Forecaster(TestModels.EchoColumn(:promo); features=feats,
                                strategy=strat, freq=Day(1), transform=tr), stbl)
            fp = fit(Forecaster(TestModels.EchoColumn(:promo); features=feats,
                                strategy=strat, freq=Day(1), id=:unique_id,
                                transform=tr), ptbl)
            a = forecast(fs, h; future=sfut)
            b = forecast(fp, h; future=pfut)
            @test isequal(a.y_hat, b.y_hat) && isequal(a.ds, b.ds)
            @test b.unique_id == fill("a", h)
            tr isa NoTransform && @test a.y_hat == vals[1:h]
        end
    end

    @testset "forecast output columns, order and types" begin
        n = 30
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        out = forecast(fit(Forecaster(TestModels.LinAR(1.0, 0.0);
                                      features=FeatureSet(Lag(1)), freq=Day(1)),
                           (ds=ds, y=Float64.(1:n))), 3)
        @test keys(out) == (:ds, :y_hat)
        @test out.ds isa Vector{Date} && out.y_hat isa Vector{Float64}
        @test out.ds == ds[end] .+ Day.(1:3)
        pf = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1))), panel)
        out = forecast(pf, 2)
        @test keys(out) == (:unique_id, :ds, :y_hat)
        @test out.unique_id isa Vector{String} && out.ds isa Vector{Date}
        @test out.y_hat isa Vector{Float64}
        @test out.unique_id == repeat(["s1", "s2", "s3"], inner=2)
        @test out.ds == vcat([s.t_last .+ Day.(1:2) for s in pf.series]...)
        renamed = (store=[fill(7, 20); fill(8, 20)], date=[ds[1:20]; ds[1:20]],
                   value=Float64.(1:40))
        fr = fit(Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                            freq=Day(1), target=:value, time=:date, id=:store), renamed)
        out = forecast(fr, 2)
        @test keys(out) == (:store, :date, :y_hat) && out.store isa Vector{Int}
        @test out.store == [7, 7, 8, 8]
        sr = fit(Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                            freq=Day(1), target=:value, time=:date),
                 (date=ds, value=Float64.(1:n)))
        @test keys(forecast(sr, 2)) == (:date, :y_hat)
    end

    @testset "forecast is repeatable and leaves the fitted state alone" begin
        n = 40
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        stbl = (ds=ds, y=50 .+ 2 .* sin.(1:n) .+ 0.1 .* (1:n))
        snap(f) = [(copy(s.y_history), copy(s.z_history), s.t_last, s.t_start,
                    s.n_train, s.transform_state, s.step0) for s in f.series]
        for strat in (Recursive(), Direct(4)), tr in (NoTransform(), LastScale(3))
            for (data, id) in ((stbl, nothing), (panel, :unique_id))
                fc = Forecaster(TestModels.LinAR(0.9, 1.0); features=FeatureSet(Lag(1),
                                RollingMean(3)), strategy=strat, freq=Day(1), id,
                                transform=tr)
                f = fit(fc, data)
                before = snap(f)
                a = forecast(f, 4)
                @test isequal(snap(f), before)
                b = forecast(f, 4)
                @test isequal(a, b)
                @test isequal(snap(f), before)
            end
        end
    end

    @testset "the forecast loop: one body for a single series and a panel" begin
        M = MachineLearningForecast
        n = 40
        mkt(model, feats; strat=Recursive(), tr=NoTransform()) =
            Forecaster(model; features=feats, strategy=strat, freq=Day(1),
                       target=:y, time=:ds, id=:unique_id, transform=tr)
        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
        stbl = (ds=ds, y=50 .+ 2 .* sin.(1:n) .+ 0.1 .* (1:n))
        mks(model, feats; strat=Recursive(), tr=NoTransform()) =
            Forecaster(model; features=feats, strategy=strat, freq=Day(1), transform=tr)
        grids_of(f, hs) = [M.future_grid(st.t_start, f.spec.freq, st.n_train, hs[k])
                           for (k, st) in enumerate(f.series)]

        @testset "_check_horizon: one name, two receivers" begin
            lin = TestModels.LinAR(1.0, 0.0)
            feats = FeatureSet(Lag(1))
            fcd = mks(lin, feats; strat=Direct(3))
            fcp = mkt(lin, feats; strat=Direct(3))
            fitted = "strategy=Direct(3) was fit with max_horizon=3 but forecast(h=5) " *
                     "was requested. Refit with Direct(5) or use Recursive()."
            spec = "backtest horizon=5 exceeds the Direct strategy's max_horizon=3. Use " *
                   "Direct(5) or reduce horizon."
            for (fc, data) in ((fcd, stbl), (fcp, panel))
                f = fit(fc, data)
                @test_throws ArgumentError M._check_horizon(f, 5)
                @test_throws fitted M._check_horizon(f, 5)
                @test_throws spec M._check_horizon(fc, 5)
                @test M._check_horizon(f, 3) === nothing
                @test M._check_horizon(fc, 3) === nothing
            end
            fcr = mks(lin, feats)
            @test M._check_horizon(fcr, 10^6) === nothing
            @test M._check_horizon(fit(fcr, stbl), 10^6) === nothing
            @test M._check_horizon(mkt(lin, feats), 10^6) === nothing
        end

        @testset "_forecast_states is what forecast computes" begin
            feats = FeatureSet(Lag(1), RollingMean(3))
            for strat in (Recursive(), Direct(5)), tr in (NoTransform(), LastScale(3))
                f = fit(mks(TestModels.LinAR(0.9, 1.0), feats; strat, tr), stbl)
                st = only(f.series)
                grid = M.future_grid(st.t_start, f.spec.freq, st.n_train, 5)
                out = M._forecast_states(f, [st], [5], [grid], nothing)
                @test only(out) == forecast(f, 5).y_hat
                @test out isa Vector{Vector{Float64}}

                fp = fit(mkt(TestModels.LinAR(0.9, 1.0), feats; strat, tr), panel)
                full = forecast(fp, 5).y_hat
                hs = [5, 2, 3]
                out = M._forecast_states(fp, fp.series, hs, grids_of(fp, hs), nothing)
                @test out == [full[1:5], full[6:7], full[11:13]]
                if strat isa Direct
                    @test_throws "strategy=Direct(5) was fit with max_horizon=5 but " *
                                 "forecast(h=6)" M._forecast_states(
                        fp, fp.series, [6, 1, 1], grids_of(fp, [6, 1, 1]), nothing)
                else
                    out = M._forecast_states(fp, fp.series, [6, 1, 1],
                                             grids_of(fp, [6, 1, 1]), nothing)
                    @test length(out[1]) == 6
                end
            end
        end

        @testset "Direct forecasts step s with the model trained for step s" begin
            # MeanModel's step-s model predicts the mean of y[s+1:n].
            fs = fit(mks(TestModels.MeanModel(), FeatureSet(Lag(1)); strat=Direct(4)),
                     stbl)
            @test forecast(fs, 4).y_hat == [mean(stbl.y[(s + 1):end]) for s in 1:4]
            fp = fit(mkt(TestModels.MeanModel(), FeatureSet(Lag(1)); strat=Direct(4)),
                     panel)
            # One global model per step: the mean pools every series.
            ys = [panel.y[panel.unique_id .== id] for id in ("s1", "s2", "s3")]
            pooled = [mean(reduce(vcat, [y[(s + 1):end] for y in ys])) for s in 1:4]
            @test forecast(fp, 4).y_hat ≈ repeat(pooled, 3)
        end

        @testset "_forecast_states is inferred" begin
            feats = FeatureSet(Lag(1), RollingMean(3), Exogenous(:promo))
            pe = merge(panel, (promo=Float64.(eachindex(panel.y)),))
            se = merge(stbl, (promo=Float64.(1:n),))
            chain = TransformChain(LastScale(4), LogRatio(3))
            for strat in (Recursive(), Direct(5)), tr in (NoTransform(), chain)
                for (fc, data) in ((mks(TestModels.LinAR(0.9, 1.0), feats; strat, tr), se),
                                   (mkt(TestModels.LinAR(0.9, 1.0), feats; strat, tr), pe))
                    f = fit(fc, data)
                    hs = fill(5, nseries(f))
                    grids = grids_of(f, hs)
                    fut = if fc.id === nothing
                        (ds=grids[1], promo=Float64.(1:5))
                    else
                        (unique_id=repeat([s.id for s in f.series], inner=5),
                         ds=vcat(grids...), promo=Float64.(1:5nseries(f)))
                    end
                    rows = M._exogenous_rows(fc, f.series, grids, fut, hs)
                    out = @inferred M._forecast_states(f, f.series, hs, grids, rows)
                    @test out isa Vector{Vector{Float64}}
                    @test M._output_table(fc, f.series, grids, out).y_hat ==
                          forecast(f, 5; future=fut).y_hat
                end
            end
            plain = FeatureSet(Lag(1))
            for strat in (Recursive(), Direct(5)), tr in (NoTransform(), chain)
                lin = TestModels.LinAR(0.9, 1.0)
                for (fc, data) in ((mks(lin, plain; strat, tr), stbl),
                                   (mkt(lin, plain; strat, tr), panel))
                    f = fit(fc, data)
                    hs = fill(5, nseries(f))
                    out = @inferred M._forecast_states(f, f.series, hs, grids_of(f, hs),
                                                       nothing)
                    @test out isa Vector{Vector{Float64}}
                end
            end
        end

        @testset "a model must return one prediction per forecast row" begin
            feats = FeatureSet(Lag(1))
            msg(got, rows) = "model WrongCount returned $got predictions for $rows " *
                "forecast row(s) at step 1; a model must return one prediction per row."
            for strat in (Recursive(), Direct(3)), extra in (1, -1)
                m = TestModels.WrongCount(extra)
                fs = fit(mks(m, feats; strat), stbl)
                fp = fit(mkt(m, feats; strat), panel)
                @test_throws msg(1 + extra, 1) forecast(fs, 3)
                @test_throws msg(3 + extra, 3) forecast(fp, 3)
                @test_throws ArgumentError forecast(fp, 3)
            end
            # The right count passes.
            fp = fit(mkt(TestModels.WrongCount(0), feats), panel)
            @test forecast(fp, 3).y_hat == zeros(9)
        end

        @testset "a model may return a column matrix and reshape its input" begin
            feats = FeatureSet(Lag(1), RollingMean(3))
            lin = TestModels.LinAR(0.9, 1.0)
            for strat in (Recursive(), Direct(5)), asmatrix in (true, false)
                rs = TestModels.ReshapeLinAR(0.9, 1.0, 0, asmatrix)
                @test forecast(fit(mks(rs, feats; strat), stbl), 5).y_hat ==
                      forecast(fit(mks(lin, feats; strat), stbl), 5).y_hat
                fr = fit(mkt(rs, feats; strat), panel)
                fl = fit(mkt(lin, feats; strat), panel)
                @test forecast(fr, 5).y_hat == forecast(fl, 5).y_hat
                # A ragged active set shrinks columns the model reshaped.
                hs = [5, 2, 3]
                @test M._forecast_states(fr, fr.series, hs, grids_of(fr, hs), nothing) ==
                      M._forecast_states(fl, fl.series, hs, grids_of(fl, hs), nothing)
            end
            for extra in (1, -1)
                m = TestModels.ReshapeLinAR(0.9, 1.0, extra, true)
                @test_throws "model ReshapeLinAR returned $(1 + extra) predictions for " *
                             "1 forecast row(s) at step 1; a model must return one " *
                             "prediction per row." forecast(
                    fit(mks(m, feats), stbl), 3)
                @test_throws "model ReshapeLinAR returned $(3 + extra) predictions for " *
                             "3 forecast row(s) at step 1" forecast(
                    fit(mkt(m, feats), panel), 3)
            end
        end

        @testset "real learners: one loop for a single series and a panel" begin
            feats = FeatureSet(Lag(1), RollingMean(3))
            small = makepanel(lens=(30, 25, 28))
            one = makepanel(lens=(30,))
            sone = (ds=one.ds, y=one.y)
            for m in (EvoTreeRegressor(nrounds=5),
                      DecisionTreeRegressor(max_depth=2, rng=StableRNG(1))),
                (strat, hs) in ((Recursive(), [5, 2, 3]), (Direct(3), [3, 2, 1]))
                hmax = maximum(hs)
                a = forecast(fit(mks(m, feats; strat), sone), hmax)
                b = forecast(fit(mkt(m, feats; strat), one), hmax)
                @test a.y_hat == b.y_hat && a.ds == b.ds
                fp = fit(mkt(m, feats; strat), small)
                full = forecast(fp, hmax).y_hat
                out = M._forecast_states(fp, fp.series, hs, grids_of(fp, hs), nothing)
                @test out == [full[(k - 1) * hmax .+ (1:hs[k])] for k in 1:3]
            end
        end

        @testset "the step loop's state is concrete and its rows cost what they must" begin
            feats = FeatureSet(Lag(1), RollingMean(3))
            fp = fit(mkt(TestModels.LinAR(0.9, 1.0), feats), panel)
            hs = [5, 2, 3]
            run = @inferred M._StepRun(fp.spec.features, fp.spec.strategy,
                                       fp.feature_names, fp.series, hs,
                                       grids_of(fp, hs), nothing)
            @test all(isconcretetype, fieldtypes(typeof(run)))

            # Columns are reused until the row count shrinks.
            probe = TestModels.ColumnProbe(Ref(Any[]))
            fq = fit(mkt(probe, feats), panel)
            M._forecast_states(fq, fq.series, hs, grids_of(fq, hs), nothing)
            c = only(fq.models).model.calls[]
            @test length.(c) == [3, 3, 2, 1, 1]
            @test c[1] === c[2] && c[4] === c[5]
            @test c[3] !== c[2] && c[4] !== c[3]

            # A step costs a predict plus a bounded fill per row.
            bytes(f, args...) = (f(args...); @allocated f(args...))
            np = 50
            many = makepanel(lens=fill(60, np), levels=collect(1.0:np),
                             starts=fill(Date(2022, 1, 1), np))
            nf = FeatureSet(Lag(1), Lag(7), RollingMean(7))
            fm = fit(mkt(TestModels.LinAR(0.9, 1.0), nf), many)
            perstep = (bytes(forecast, fm, 40) - bytes(forecast, fm, 20)) / 20
            rows = NamedTuple{Tuple(fm.feature_names)}(
                Tuple([ones(np) for _ in fm.feature_names]))
            pbytes = bytes(M._predict, only(fm.models), rows)
            @test perstep <= pbytes + np * 64 * length(nf) + 256
        end
    end

    @testset "metric protocol on a panel" begin
        # a climbs 1 a day, b 10, as in the test above.
        function trends(nb; na=50)
            ids = String[]; ds = Date[]; y = Float64[]
            for i in 1:50, (id, slope, n) in (("a", 1.0, na), ("b", 10.0, nb))
                i ≤ n || continue
                push!(ids, id); push!(ds, Date(2022, 1, 1) + Day(i - 1))
                push!(y, (id == "a" ? 0.0 : 1000.0) + slope * i)
            end
            return (unique_id=ids, ds=ds, y=y)
        end
        function wiggly(na, nb)
            ids = String[]; ds = Date[]; y = Float64[]
            for (id, lv, n) in (("a", 100.0, na), ("b", 5.0, nb)), i in 1:n
                push!(ids, id); push!(ds, Date(2022, 1, 1) + Day(i - 1))
                push!(y, lv + i + 3 * sin(i) + (i % 7))
            end
            return (unique_id=ids, ds=ds, y=y)
        end
        # Per-series scores of each fold, weighted by scored points.
        function manual(f, pd, r)
            vals = Float64[]
            for k in sort(unique(r.forecasts.fold))
                tot = 0.0; cnt = 0
                for id in unique(pd.unique_id)
                    sel = (r.forecasts.fold .== k) .& (r.forecasts.unique_id .== id)
                    any(sel) || continue
                    o = r.forecasts.origin[sel][1]
                    tr = pd.y[(pd.unique_id .== id) .& (pd.ds .<= o)]
                    tot += sum(sel) * f(r.forecasts.y[sel], r.forecasts.y_hat[sel], tr)
                    cnt += sum(sel)
                end
                push!(vals, tot / cnt)
            end
            return vals
        end
        models = (TestModels.LinAR(1.0, 0.0), EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
        val(r, name) = r.scores.value[r.scores.metric .== name]

        @testset "Mase(m) and Rmsse(m) are per series and point-weighted" begin
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
            r = backtest(fc, trends(47); horizon=5, initial=40, step=5,
                         metrics=(Mase(2), Rmsse(2), Mase()))
            @test unique(r.scores.metric) == [:mase_2, :rmsse_2, :mase]
            # Lag-2 scale is 2 slopes, the naive miss averages 3 slopes.
            @test val(r, :mase_2) ≈ [1.5, (5 * 1.5 + 2 * 0.75) / 7]
            @test val(r, :mase) ≈ [3.0, (5 * 3.0 + 2 * 1.5) / 7]
            for m in models
                pd = wiggly(55, 52)
                rr = backtest(mk(m, FeatureSet(Lag(1))), pd; horizon=5, initial=40,
                              step=5, metrics=(Mase(12), Rmsse(7), mae))
                @test unique(rr.scores.metric) == [:mase_12, :rmsse_7, :mae]
                @test val(rr, :mase_12) ≈
                      manual((y, ŷ, tr) -> mase(y, ŷ; y_train=tr, m=12), pd, rr)
                @test val(rr, :rmsse_7) ≈
                      manual((y, ŷ, tr) -> rmsse(y, ŷ; y_train=tr, m=7), pd, rr)
                ro = backtest(mk(m, FeatureSet(Lag(1))), pd; horizon=5, initial=40,
                              step=5, metrics=(mase, rmsse))
                rn = backtest(mk(m, FeatureSet(Lag(1))), pd; horizon=5, initial=40,
                              step=5, metrics=(Mase(), Rmsse()))
                @test isequal(ro.scores, rn.scores)
            end
        end

        @testset "a fold that scores one series keeps the weighted mean" begin
            # Bits differ by Julia version, so one case must show.
            shows = Bool[]
            for (m, metric, h) in ((TestModels.LinAR(1.0, 0.0), Rmsse(7), 6),
                                   (DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)),
                                    Mase(7), 12))
                # b ends early, so fold 3 scores a alone.
                pd = wiggly(40 + 3h, 40 + 2h)
                r = backtest(mk(m, FeatureSet(Lag(1))), pd; horizon=h, initial=40,
                             step=h, metrics=(metric,))
                last_fold = r.forecasts.fold .== 3
                @test unique(r.forecasts.unique_id[last_fold]) == ["a"]
                o = r.forecasts.origin[last_fold][1]
                tr = pd.y[(pd.unique_id .== "a") .& (pd.ds .<= o)]
                v = metric(r.forecasts.y[last_fold], r.forecasts.y_hat[last_fold];
                           y_train=tr)
                push!(shows, (h * v) / h != v)
                @test val(r, metric isa Mase ? :mase_7 : :rmsse_7)[3] == (h * v) / h
            end
            @test any(shows)
        end

        @testset "series errors name the series" begin
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
            @test_throws "series unique_id=\"a\": mase needs length(y_train) > m; got " *
                         "length(y_train)=40 with m=40. Pass a longer y_train or a " *
                         "smaller m." backtest(fc, trends(50); horizon=5, initial=40,
                                               step=5, metrics=(Mase(40),))
            @test_throws "series unique_id=\"a\": rmsse needs length(y_train) > m" backtest(
                fc, trends(50); horizon=5, initial=40, step=5, metrics=(Rmsse(40),))
        end

        @testset "custom scaled and point metrics side by side" begin
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
            pd = trends(50)
            ns = TestModels.NaiveScaled(2)
            f = (a, b) -> mae(a, b)
            r = backtest(fc, pd; horizon=5, initial=40, step=5, metrics=(f, ns, Mase(2)))
            @test unique(r.scores.metric) == [Symbol(f), :naive_scaled_2, :mase_2]
            ya = pd.y[pd.unique_id .== "a"]; yb = pd.y[pd.unique_id .== "b"]
            @test ns.seen == [ya[1:40], yb[1:40], ya[1:45], yb[1:45]]
            @test val(r, :naive_scaled_2) ≈ [3.0, 3.0]
            @test val(r, Symbol(f)) ≈ [16.5, 16.5]
        end

        @testset "horizon 1 on a ragged panel" begin
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
            r = backtest(fc, trends(47); horizon=1, initial=45, step=1,
                         metrics=(Mase(2), Rmsse(2)))
            @test length(val(r, :mase_2)) == 5 && length(val(r, :rmsse_2)) == 5
            @test val(r, :mase_2) ≈ fill(0.5, 5)
            @test val(r, :rmsse_2) ≈ fill(0.5, 5)
            for m in models
                r2 = backtest(mk(m, FeatureSet(Lag(1))), wiggly(55, 52); horizon=1,
                              initial=48, step=1, metrics=(Mase(12),))
                @test all(isfinite, r2.scores.value)
            end
        end

        @testset "a type is not a metric" begin
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
            @test_throws "got the type Mase; pass Mase() or Mase(12)" backtest(
                fc, trends(50); horizon=5, initial=40, metrics=(Mase,))
        end
    end
end
