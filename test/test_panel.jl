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
    mk(model, feats; strat=Recursive()) =
        Forecaster(model; features=feats, strategy=strat, freq=Day(1),
                   target=:y, time=:ds, id=:unique_id)

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
        @test length(f.machines) == 1
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

    @testset "exogenous covariates join on (id, time)" begin
        ex = (unique_id=panel.unique_id, ds=panel.ds, y=panel.y,
              promo=Float64.(eachindex(panel.y)))
        fc = mk(TestModels.EchoColumn(:promo), FeatureSet(Lag(1), Exogenous(:promo)))
        f = fit(fc, ex)
        grids = [[s.t_last + Day(i) for i in 1:2] for s in f.series]
        nd = (unique_id=repeat(["s1", "s2", "s3"], inner=2),
              ds=vcat(grids...), promo=[11.0, 12.0, 21.0, 22.0, 31.0, 32.0])
        @test forecast(f, 2; new_data=nd).y_hat == [11.0, 12.0, 21.0, 22.0, 31.0, 32.0]
        p2 = shuffle(StableRNG(9), 1:6)
        shuf = (unique_id=nd.unique_id[p2], ds=nd.ds[p2], promo=nd.promo[p2])
        @test forecast(f, 2; new_data=shuf).y_hat == [11.0, 12.0, 21.0, 22.0, 31.0, 32.0]

        @test_throws ArgumentError forecast(f, 2)
        @test_throws "features contain Exogenous(:promo) but forecast() was called " *
                     "without new_data. Pass a table with columns (:unique_id, :ds, " *
                     ":promo) covering all 3 series over their 2 forecast steps." (
            forecast(f, 2))
        missing_row = (unique_id=nd.unique_id[1:5], ds=nd.ds[1:5], promo=nd.promo[1:5])
        err = try forecast(f, 2; new_data=missing_row) catch e; e end
        @test err isa ArgumentError && occursin("no row for", err.msg)
        noid = (ds=nd.ds, promo=nd.promo)
        err2 = try forecast(f, 2; new_data=noid) catch e; e end
        @test err2 isa ArgumentError && occursin("unique_id", err2.msg)
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
            want = forecast(f, 2; new_data=nd)
            bt = backtest(fc, ex; horizon=5, initial=40, step=10)
            for T in (Union{Missing,Date}, Any)
                @test forecast(f, 2; new_data=merge(nd, (ds=Vector{T}(nd.ds),))) == want
                r = backtest(fc, merge(ex, (ds=Vector{T}(ex.ds),)); horizon=5,
                             initial=40, step=10)
                @test isequal(r.folds, bt.folds) && isequal(r.metrics, bt.metrics)
            end
            holed = merge(nd, (ds=Vector{Union{Missing,Date}}(nd.ds),))
            holed.ds[4] = missing
            @test_throws ArgumentError forecast(f, 2; new_data=holed)
            @test_throws "new_data's time column :ds has a missing value at row 4" forecast(
                f, 2; new_data=holed)
            @test_throws "has element type DateTime at row 1, but the training time " *
                         "column is Date" forecast(f, 2; new_data=merge(
                             nd, (ds=DateTime.(nd.ds),)))
            dup = (unique_id=[nd.unique_id; "s2"], ds=[nd.ds; nd.ds[3]],
                   promo=[nd.promo; 0.0])
            @test_throws ArgumentError forecast(f, 2; new_data=dup)
            @test_throws "new_data has duplicate rows for unique_id=\"s2\" at " *
                         "$(nd.ds[3]). Deduplicate new_data" forecast(f, 2; new_data=dup)
            gap = merge(nd, (promo=Union{Missing,Float64}[nd.promo...],))
            gap.promo[4] = missing
            @test_throws ArgumentError forecast(f, 2; new_data=gap)
            @test_throws "new_data has a missing value in exogenous column :promo for " *
                         "unique_id=\"s2\" at $(nd.ds[4]). Provide complete exogenous " *
                         "values for every series at every forecast step." forecast(
                f, 2; new_data=gap)
        end
    end

    @testset "Direct on a panel" begin
        f = fit(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=Direct(4)), panel)
        @test length(f.machines) == 4
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

    @testset "backtest cuts folds on the global timestamp grid" begin
        big = makepanel(lens=(60, 50, 55))
        fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)))
        r = backtest(fc, big; horizon=5, initial=40, step=10, metrics=(mae, rmse))
        @test keys(r.folds) == (:origin, :step, :unique_id, :ds, :y, :y_hat)
        @test length(unique(r.folds.origin)) == length(40:10:(60 - 5))
        @test extrema(r.folds.step) == (1, 5)
        @test Set(unique(r.folds.unique_id)) ⊆ Set(["s1", "s2", "s3"])
        actual = Dict((big.unique_id[i], big.ds[i]) => big.y[i] for i in eachindex(big.y))
        @test all(actual[(r.folds.unique_id[j], r.folds.ds[j])] == r.folds.y[j]
                  for j in eachindex(r.folds.y))
        # LinAR(1, 0) is naive: each step repeats the origin's actual.
        @test all(yh == actual[(id, o)] for (yh, id, o) in
                  zip(r.folds.y_hat, r.folds.unique_id, r.folds.origin))
        # y climbs 1 a day, so step s misses by s in every fold.
        @test r.metrics.fold == [1, 1, 2, 2, 0, 0]
        @test r.metrics.value ≈ repeat([3.0, sqrt(11.0)], 3)
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
            # Fourier's step index is 0-based per series, as documented.
            fe = fit(mk(TestModels.EchoColumn(:fourier_7_0_sin_1), fourier; strat=s), rag)
            got = forecast(fe, 3).y_hat
            @test got ≈ [sin(2π * (n - 1 + i) / 7) for n in (60, 50, 40) for i in 1:3]
            @test !(got[1] ≈ got[4]) && !(got[1] ≈ got[7])
            # s3 starts after the first origin and joins at the second.
            r = backtest(mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=s), rag;
                         horizon=5, initial=15, step=10)
            @test unique(r.folds.origin) == Date(2022, 1, 15) .+ Day.(0:10:40)
            first_fold = r.folds.origin .== Date(2022, 1, 15)
            @test unique(r.folds.unique_id[first_fold]) == ["s1", "s2"]
            @test count(==("s3"), r.folds.unique_id) == 4 * 5
            @test all(yh == actual[(id, o)] for (yh, id, o) in
                      zip(r.folds.y_hat, r.folds.unique_id, r.folds.origin))
            @test all(==(3.0), r.metrics.value[r.metrics.metric .== :mae])
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
            @test exog.folds.origin == plain.folds.origin
            @test exog.folds.step == plain.folds.step
            @test exog.folds.unique_id == plain.folds.unique_id
            @test exog.folds.ds == plain.folds.ds
        end
        # EchoColumn shows each step reads its own (id, time) row.
        for (data, freq, h, init, stp) in cases
            fc = Forecaster(TestModels.EchoColumn(:promo); freq=freq, id=:unique_id,
                            features=FeatureSet(Lag(1), Exogenous(:promo)))
            r = backtest(fc, data; horizon=h, initial=init, step=stp)
            promo = Dict((data.unique_id[i], data.ds[i]) => data.promo[i]
                         for i in eachindex(data.promo))
            @test r.folds.y_hat == [promo[(r.folds.unique_id[j], r.folds.ds[j])]
                                    for j in eachindex(r.folds.ds)]
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
            @test nseries(f) == 3 && length(f.machines) == (s isa Direct ? 4 : 1)
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
            @test r.folds.origin == ref.folds.origin && r.folds.ds == ref.folds.ds
            @test r.folds.unique_id == ref.folds.unique_id && r.folds.y == ref.folds.y
            @test all(isfinite, r.folds.y_hat) && all(isfinite, r.metrics.value)
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
        for s in (Recursive(), Direct(2))
            fc = mk(TestModels.LinAR(1.0, 0.0), FeatureSet(Lag(1)); strat=s)
            @test occursin("id=:unique_id", sprint(show, fc))
            f = fit(fc, panel)
            for txt in (sprint(show, f), sprint(show, MIME"text/plain"(), f))
                @test occursin("trained on 3 series", txt)
                @test occursin(string(s), txt)
            end
        end
        single = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                            freq=Day(1))
        @test !occursin("id=", sprint(show, single))
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
end
