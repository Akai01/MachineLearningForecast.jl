module ThirdPartyExog

using MachineLearningForecast: ExogenousFeature, ColumnAccumulator
import MachineLearningForecast: outputnames, materialize!, featurevalues

"Stores its input as `col`, not in the `cols` field forecast reads."
struct NoCols <: ExogenousFeature
    col::Symbol
end

"Stores its input in the required `cols::Vector{Symbol}` field."
struct Doubled <: ExogenousFeature
    cols::Vector{Symbol}
end

"Exposes its input as a `cols` property, not a field."
struct PropCols <: ExogenousFeature
    col::Symbol
end
Base.getproperty(f::PropCols, s::Symbol) =
    s === :cols ? [getfield(f, :col)] : getfield(f, s)

"Has a `cols` property whose getter fails for its own reason."
struct BrokenCols <: ExogenousFeature
    col::Symbol
end
Base.getproperty(f::BrokenCols, s::Symbol) =
    s === :cols ? error("boom in getproperty") : getfield(f, s)

const Ours = Union{NoCols,Doubled,PropCols,BrokenCols}
_col(f::Union{NoCols,PropCols,BrokenCols}) = f.col
_col(f::Doubled) = only(f.cols)
outputnames(f::Ours) = [Symbol(_col(f), :_x2)]
function materialize!(out::ColumnAccumulator, f::Ours, y, t, data)
    push!(out, only(outputnames(f)) => Vector{Union{Missing,Float64}}(2 .* data[_col(f)]))
    return out
end
featurevalues(f::Ours, y_hist, t_next, exog_row) = (2.0 * exog_row[_col(f)],)

end

@testset "validation and guards" begin
    df = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19)), y=Float64.(1:50))
    fs = FeatureSet(Lag(1))
    mk(; kwargs...) = Forecaster(TestModels.LinAR(1.0, 0.0);
                                 features=get(kwargs, :features, fs),
                                 strategy=Recursive(), freq=Day(1),
                                 target=get(kwargs, :target, :y),
                                 time=get(kwargs, :time, :ds))

    @testset "a feature emitting the target column is refused (total leak)" begin
        err = try mk(features=FeatureSet(Lag(1), Exogenous(:y))) catch e; e end
        @test err isa ArgumentError
        @test occursin("column named :y", err.msg)
        @test occursin("leak", err.msg)
        @test_throws ArgumentError mk(features=FeatureSet(Lag(1),
                                                          CustomFeature(:y, length, 0)))
        @test_throws "the feature set produces a column named :y, which is the target " *
                     "column" mk(features=FeatureSet(Lag(1), CustomFeature(:y, length, 0)))
        err2 = try mk(features=FeatureSet(Lag(1), Exogenous(:ds))) catch e; e end
        @test err2 isa ArgumentError
        @test occursin("Calendar", err2.msg)
    end

    @testset "mutating the features vector cannot bypass the leak guard" begin
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            v = MachineLearningForecast.AbstractFeature[Lag(1)]
            fc = Forecaster(m; features=FeatureSet(v), freq=Day(1))
            push!(v, Exogenous(:y))
            @test MachineLearningForecast.outputnames(fc.features) == [:y_lag_1]
            @test fit(fc, df).feature_names == [:y_lag_1]
        end
    end

    @testset "time column may not shadow reserved result columns" begin
        for bad in (:origin, :step, :y_hat)
            err = try mk(target=:v, time=bad) catch e; e end
            @test err isa ArgumentError
            @test occursin("reserved", err.msg)
        end
        @test_throws ArgumentError mk(target=:v, time=:y)   # collides with folds.y
        @test_throws "time=:y collides with a column name reserved by forecast()/" *
                     "backtest() results (:origin, :step, :y, :y_hat). Rename the time " *
                     "column" mk(target=:v, time=:y)
    end

    @testset "ragged tables are rejected by name and length" begin
        err = try fit(mk(), (ds=df.ds, y=df.y[1:49])) catch e; e end
        @test err isa ArgumentError
        @test occursin("unequal lengths", err.msg)
        @test occursin(":y has 49 rows", err.msg)
    end

    @testset "non-finite targets are rejected, not propagated" begin
        for bad in (NaN, Inf, -Inf)
            y = copy(df.y); y[10] = bad
            err = try fit(mk(), (ds=df.ds, y=y)) catch e; e end
            @test err isa ArgumentError
            @test occursin("non-finite", err.msg)
            @test occursin("row 10", err.msg)
        end
    end

    @testset "non-finite exogenous values pass through to the model" begin
        # Rejecting them like targets was considered and declined.
        promo = Float64.(1:50); promo[20] = NaN; promo[30] = Inf
        dfe = (ds=df.ds, y=df.y, promo=promo)
        fse = FeatureSet(Lag(1), Exogenous(:promo))
        future = (ds=df.ds[end] .+ Day.(1:3), promo=[NaN, 1.0, -Inf])
        echo = Forecaster(TestModels.EchoColumn(:promo); features=fse, freq=Day(1))
        @test isequal(forecast(fit(echo, dfe), 3; new_data=future).y_hat,
                      [NaN, 1.0, -Inf])
        X, _, _ = MachineLearningForecast.build_training_frame(fse, dfe, :y, :ds)
        @test isnan(X.promo[19]) && X.promo[29] == Inf
        dt = Forecaster(DecisionTreeRegressor(max_depth=2, rng=StableRNG(1));
                        features=fse, freq=Day(1))
        @test length(backtest(dt, dfe; horizon=3, initial=40).folds.y_hat) == 9
        clean = (ds=df.ds, y=df.y, promo=Float64.(1:50))
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            f = fit(Forecaster(m; features=fse, freq=Day(1)), clean)
            @test length(forecast(f, 3; new_data=future).y_hat) == 3
        end
    end

    @testset "time column type must support + freq" begin
        err = try fit(mk(), (ds=string.(df.ds), y=df.y)) catch e; e end
        @test err isa ArgumentError
        @test occursin("does not support", err.msg)
        @test occursin("String", err.msg)
    end

    @testset "month-end series are gap-free (anchored grid)" begin
        me = [Date(2020, 1, 31) + Month(k) for k in 0:23]
        @test me[2] == Date(2020, 2, 29) && me[3] == Date(2020, 3, 31)
        # Stepping from Feb 29 would give Mar 29, a false gap.
        @test MachineLearningForecast.validate_time_column(me, :ds, Month(1)) === nothing
        fcm = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Recursive(),
                         freq=Month(1), target=:y, time=:ds)
        out = forecast(fit(fcm, (ds=me, y=Float64.(1:24))), 3)
        @test out.ds == [Date(2022, 1, 31), Date(2022, 2, 28), Date(2022, 3, 31)]
        gapped = vcat(me[1:5], me[7:end])
        @test_throws ArgumentError MachineLearningForecast.validate_time_column(
            gapped, :ds, Month(1))
        @test_throws "time column :ds has 1 gap for freq=1 month; first gap after " *
                     "2020-05-31. Reindex your data or resample before fitting." (
            MachineLearningForecast.validate_time_column(gapped, :ds, Month(1)))
    end

    @testset "gap count reports discontinuities, not off-grid rows" begin
        # One missing day is one gap, not one per later row.
        full = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 1) + Day(999))
        one = vcat(full[1:100], full[102:end])
        validate = MachineLearningForecast.validate_time_column
        err = try validate(one, :ds, Day(1)) catch e; e end
        @test err isa ArgumentError
        @test occursin("has 1 gap for", err.msg)
        @test occursin(string(full[100]), err.msg)        # "first gap after <ts>"

        three = vcat(full[1:100], full[102:200], full[202:300], full[302:end])
        err3 = try validate(three, :ds, Day(1)) catch e; e end
        @test occursin("has 3 gaps for", err3.msg)

        run5 = vcat(full[1:100], full[106:end])
        err5 = try validate(run5, :ds, Day(1)) catch e; e end
        @test occursin("has 1 gap for", err5.msg)
    end

    @testset "missing, off-grid and non-advancing time columns" begin
        days = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 20))
        withmissing = Vector{Union{Missing,Date}}(days); withmissing[5] = missing
        validate = MachineLearningForecast.validate_time_column
        err = try validate(withmissing, :ds, Day(1)) catch e; e end
        @test err isa ArgumentError                       # not a bare TypeError
        @test occursin("missing values", err.msg) && occursin("row 5", err.msg)

        offgrid = collect(DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 1) + Hour(9))
        offgrid[6] += Minute(30)
        err2 = try validate(offgrid, :ds, Hour(1)) catch e; e end
        @test err2 isa ArgumentError
        @test occursin("does not", err2.msg) && occursin("grid", err2.msg)

        err3 = try validate(days, :ds, Day(0)) catch e; e end
        @test err3 isa ArgumentError && occursin("does not advance", err3.msg)
    end

    @testset "tune validates horizon/initial/step like backtest" begin
        df2 = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 4, 10)), y=Float64.(1:100))
        base = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                          strategy=Recursive(), freq=Day(1))
        grid = (model=[TestModels.LinAR(1.0, 0.0)],)
        for (kw, word) in ((:horizon, "horizon"), (:initial, "initial"), (:step, "step"))
            args = Dict(:horizon => 10, :initial => 60, :step => 10)
            args[kw] = 0
            err = try tune(base, df2; grid=grid, args...) catch e; e end
            @test err isa ArgumentError
            @test occursin(word, err.msg)      # not a bare "step cannot be zero"
            @test occursin("tune $word must be ≥ 1, got 0. Pass", err.msg)
        end
    end

    @testset "BacktestResult show is correct with no metrics" begin
        df3 = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 4, 10)), y=Float64.(1:100))
        fcb = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                         strategy=Recursive(), freq=Day(1))
        r = backtest(fcb, df3; horizon=10, initial=60, step=10, metrics=())
        nfolds = length(unique(r.folds.origin))
        @test nfolds > 1
        @test occursin("$nfolds fold", sprint(show, MIME"text/plain"(), r))
        @test occursin("$nfolds folds", sprint(show, r))
    end

    @testset "export surface is pinned" begin
        # The README quotes these counts; keep them honest.
        exports = setdiff(names(MachineLearningForecast), [:MachineLearningForecast])
        @test length(exports) == 33
        @test :Forecaster in exports && :tune in exports && :mase in exports
    end

    @testset "Calendar sub-daily parts need a DateTime column" begin
        err = try fit(mk(features=FeatureSet(Lag(1), Calendar(:hour))), df) catch e; e end
        @test err isa ArgumentError
        @test occursin("sub-daily", err.msg)
        @test occursin("hour", err.msg)
        hds = collect(DateTime(2022, 1, 1):Hour(1):DateTime(2022, 1, 1) + Hour(49))
        fch = Forecaster(TestModels.EchoColumn(:hour);
                         features=FeatureSet(Lag(1), Calendar(:hour)),
                         strategy=Recursive(), freq=Hour(1), target=:y, time=:ds)
        got = forecast(fit(fch, (ds=hds, y=Float64.(1:50))), 3)
        @test got.y_hat == Float64.(hour.(hds[end] .+ Hour.(1:3)))
        # values are checked, so loosely typed DateTimes work
        for T in (Union{Missing,DateTime}, Any)
            @test forecast(fit(fch, (ds=Vector{T}(hds), y=Float64.(1:50))), 3) == got
            for m in (EvoTreeRegressor(nrounds=5),
                      DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
                fcm = Forecaster(m; features=FeatureSet(Lag(1), Calendar(:hour, :minute)),
                                 freq=Hour(1))
                @test forecast(fit(fcm, (ds=Vector{T}(hds), y=Float64.(1:50))), 3) ==
                      forecast(fit(fcm, (ds=hds, y=Float64.(1:50))), 3)
            end
        end
        msg = "Calendar(:hour) needs a sub-daily time column, but the time column " *
              "has element type Date. Use a DateTime time column, or drop :hour " *
              "from the Calendar feature."
        @test err.msg == msg
        for T in (Union{Missing,Date}, Any)
            @test_throws msg fit(mk(features=FeatureSet(Lag(1), Calendar(:hour))),
                                 (ds=Vector{T}(df.ds), y=df.y))
        end
    end

    @testset "new_data: duplicates and eltype mismatches are diagnosed" begin
        fce = Forecaster(TestModels.EchoColumn(:promo);
                         features=FeatureSet(Lag(1), Exogenous(:promo)),
                         strategy=Recursive(), freq=Day(1), target=:y, time=:ds)
        fitted = fit(fce, (ds=df.ds, y=df.y, promo=fill(1.0, 50)))
        grid = collect(df.ds[end] + Day(1):Day(1):df.ds[end] + Day(3))

        dup = (ds=[grid[1], grid[1], grid[2], grid[3]], promo=[1.0, 2.0, 3.0, 4.0])
        err = try forecast(fitted, 3; new_data=dup) catch e; e end
        @test err isa ArgumentError
        @test occursin("duplicate timestamps", err.msg)

        mism = (ds=DateTime.(grid), promo=[1.0, 2.0, 3.0])
        err2 = try forecast(fitted, 3; new_data=mism) catch e; e end
        @test err2 isa ArgumentError
        @test occursin("element type", err2.msg)
        @test !occursin("missing", err2.msg)

        shuffled = (ds=grid[[3, 1, 2]], promo=[30.0, 10.0, 20.0])
        @test forecast(fitted, 3; new_data=shuffled).y_hat == [10.0, 20.0, 30.0]
    end

    @testset "new_data's time column is checked by value, not element type" begin
        dfe = (ds=df.ds, y=df.y, promo=Float64.(1:50))
        grid = collect(df.ds[end] + Day(1):Day(1):df.ds[end] + Day(3))
        nd = (ds=grid, promo=[1.0, 2.0, 3.0])
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fce = Forecaster(m; features=FeatureSet(Lag(1), Exogenous(:promo)),
                             freq=Day(1))
            f = fit(fce, dfe)
            want = forecast(f, 3; new_data=nd)
            bt = backtest(fce, dfe; horizon=3, initial=40)
            for T in (Union{Missing,Date}, Any)
                @test forecast(f, 3; new_data=(ds=Vector{T}(grid), promo=nd.promo)) == want
                # backtest slices new_data from the user's own time column
                r = backtest(fce, merge(dfe, (ds=Vector{T}(df.ds),)); horizon=3,
                             initial=40)
                @test isequal(r.folds, bt.folds) && isequal(r.metrics, bt.metrics)
            end
            holed = (ds=Union{Missing,Date}[grid[1], missing, grid[2], grid[3]],
                     promo=[1.0, 9.0, 2.0, 3.0])
            @test_throws ArgumentError forecast(f, 3; new_data=holed)
            @test_throws "new_data's time column :ds has a missing value at row 2" forecast(
                f, 3; new_data=holed)
            mixed = (ds=Any[grid[1], DateTime(grid[2]), grid[3]], promo=nd.promo)
            @test_throws "has element type DateTime at row 2, but the training time " *
                         "column is Date" forecast(f, 3; new_data=mixed)
        end
    end

    @testset "FeatureSet accepts a plain vector of features" begin
        @test FeatureSet([Lag(k) for k in 1:3]) == FeatureSet(Lag(1), Lag(2), Lag(3))
        @test FeatureSet([Calendar(:dayofweek), Fourier(7, 1)]) isa FeatureSet
        @test FeatureSet(MachineLearningForecast.AbstractFeature[Lag(1)]) isa FeatureSet
    end

    @testset "single-row (forecast-time) paths: exact values" begin
        y = Float64[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        t = collect(Date(2021, 3, 1):Day(1):Date(2021, 3, 10))
        # Calendar single-row values match Dates and the batch row.
        cal = Calendar(:dayofweek, :month, :weekofyear)
        @test collect(MachineLearningForecast.featurevalues(cal, y, t[7], nothing)) ==
              Float64[dayofweek(t[7]), month(t[7]), week(t[7])]
        acc = MachineLearningForecast.ColumnAccumulator()
        MachineLearningForecast.materialize!(acc, cal, y, t, (ds=t, y=y))
        @test collect(MachineLearningForecast.featurevalues(cal, y[1:6], t[7], nothing)) ==
              [col[7] for (_, col) in acc]
        @test only(MachineLearningForecast.featurevalues(
                       Diff(3; lag=2), y, t[1], nothing)) ==
              y[10 + 1 - 2] - y[10 + 1 - 2 - 3]

        ds = collect(Date(2022, 1, 1):Day(1):Date(2022, 2, 19))
        d2 = (ds=ds, y=Float64.(1:50))
        fc_dw = Forecaster(TestModels.EchoColumn(:dayofweek);
                           features=FeatureSet(Lag(1), Calendar(:dayofweek)),
                           strategy=Recursive(), freq=Day(1))
        @test forecast(fit(fc_dw, d2), 4).y_hat ==
              Float64.(dayofweek.(ds[end] .+ Day.(1:4)))
        fc_df = Forecaster(TestModels.EchoColumn(:y_diff_1_lag_1);
                           features=FeatureSet(Lag(1), Diff(1)),
                           strategy=Recursive(), freq=Day(1))
        # Echoed Diff(1): 50-49, 1-50, -49-1, -50-(-49).
        @test forecast(fit(fc_df, d2), 4).y_hat == [1.0, -49.0, -50.0, -1.0]
    end

    @testset "mase is usable as a backtest/tune metric" begin
        fcb = Forecaster(TestModels.LinAR(1.0, 0.0); features=fs, strategy=Recursive(),
                         freq=Day(1))
        res = backtest(fcb, df; horizon=7, initial=30, step=10, metrics=(mae, mase))
        @test :mase in res.metrics.metric
        vals = [res.metrics.value[i] for i in eachindex(res.metrics.fold)
                if res.metrics.metric[i] == :mase && res.metrics.fold[i] > 0]
        @test all(isfinite, vals) && all(>(0), vals)
        @test MachineLearningForecast.needs_ytrain(mase)
        @test !MachineLearningForecast.needs_ytrain(mae)
    end

    @testset "a third-party ExogenousFeature must store its columns in cols" begin
        dfe = (ds=df.ds, y=df.y, promo=Float64.(1:50))
        future = (ds=collect(df.ds[end] + Day(1):Day(1):df.ds[end] + Day(3)),
                  promo=[1.0, 2.0, 3.0])
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            bad = Forecaster(m; features=FeatureSet(Lag(1), ThirdPartyExog.NoCols(:promo)),
                             freq=Day(1))
            f = fit(bad, dfe)                     # fitting never reads cols
            for run in (() -> forecast(f, 3; new_data=future),
                        () -> backtest(bad, dfe; horizon=3, initial=40))
                @test_throws ArgumentError run()
                @test_throws "NoCols is an ExogenousFeature without a cols field" run()
                @test_throws "in a field cols::Vector{Symbol}" run()
            end
            good = Forecaster(m; freq=Day(1),
                              features=FeatureSet(Lag(1), ThirdPartyExog.Doubled([:promo])))
            @test all(isfinite, forecast(fit(good, dfe), 3; new_data=future).y_hat)
            @test length(backtest(good, dfe; horizon=3, initial=40).folds.y_hat) == 9
            # A cols property is enough, as in 0.1.0.
            prop = Forecaster(m; freq=Day(1),
                              features=FeatureSet(Lag(1), ThirdPartyExog.PropCols(:promo)))
            @test forecast(fit(prop, dfe), 3; new_data=future) ==
                  forecast(fit(good, dfe), 3; new_data=future)
            bp = backtest(prop, dfe; horizon=3, initial=40)
            bg = backtest(good, dfe; horizon=3, initial=40)
            @test isequal(bp.folds, bg.folds) && isequal(bp.metrics, bg.metrics)
        end
        echo = Forecaster(TestModels.EchoColumn(:promo_x2); freq=Day(1),
                          features=FeatureSet(Lag(1), ThirdPartyExog.Doubled([:promo])))
        @test forecast(fit(echo, dfe), 3; new_data=future).y_hat == [2.0, 4.0, 6.0]
        fs3 = FeatureSet(Lag(1), Exogenous(:a, :b), ThirdPartyExog.Doubled([:c]))
        @test (@inferred MachineLearningForecast.exogenous_columns(fs3)) == [:a, :b, :c]
        @test MachineLearningForecast.exogenous_columns(FeatureSet(Lag(1))) == Symbol[]
        fsp = FeatureSet(Lag(1), ThirdPartyExog.PropCols(:c))
        @test MachineLearningForecast.exogenous_columns(fsp) == [:c]
        exogcols = MachineLearningForecast.exogenous_columns
        fsb = FeatureSet(Lag(1), ThirdPartyExog.BrokenCols(:c))
        @test_throws ErrorException("boom in getproperty") exogcols(fsb)
    end
end
