@testset "forecaster" begin
    n = 60
    t = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(n - 1))
    y = Float64.(1:n)
    df = (ds=t, y=y)

    @testset "construction validation" begin
        @test_throws ArgumentError Forecaster("not a model";
            features=FeatureSet(Lag(1)), freq=Day(1))
        @test_throws "got String. Pass e.g. EvoTreeRegressor()" Forecaster("not a model";
            features=FeatureSet(Lag(1)), freq=Day(1))
        for M in (EvoTreeRegressor, DecisionTreeRegressor)
            @test_throws ArgumentError Forecaster(M; features=FeatureSet(Lag(1)),
                                                  freq=Day(1))
            @test_throws "got the type $(nameof(M)). Pass an instance instead, e.g. " *
                         "$(nameof(M))()." Forecaster(M; features=FeatureSet(Lag(1)),
                                                      freq=Day(1))
        end
        @test_throws "got DataType" Forecaster(Int; features=FeatureSet(Lag(1)),
                                               freq=Day(1))
        @test_throws ArgumentError Forecaster(TestModels.LinAR(1.0, 0.0);
            features=FeatureSet(Lag(1)), freq=Day(1), target=:y, time=:y)
        @test_throws "target and time must be different columns, both were :y. Pass " *
                     "the names of your target and time columns, e.g. target=:y, " *
                     "time=:ds." Forecaster(TestModels.LinAR(1.0, 0.0);
            features=FeatureSet(Lag(1)), freq=Day(1), target=:y, time=:y)
        fc = @test_logs((:warn, r"neither an MLJModelInterface.Deterministic nor a " *
                                "Probabilistic"),
                        (:warn, r"target_scitype"),
                        Forecaster(TestModels.DummyInterval(); features=FeatureSet(Lag(1)),
                                   freq=Day(1)))
        @test fc isa Forecaster
        # A Probabilistic model warns only about its target_scitype.
        @test_logs (:warn, r"target_scitype") Forecaster(TestModels.DummyProb();
            features=FeatureSet(Lag(1)), freq=Day(1))
    end

    @testset "Forecaster has no positional constructor" begin
        m = TestModels.LinAR(1.0, 0.0)
        fs = FeatureSet(Lag(1))
        @test_throws MethodError Forecaster(m, fs, Recursive(), Day(1), :y, :ds,
                                            nothing, NoTransform())
        @test_throws MethodError Forecaster(m, fs, Recursive(), Day(1), :y, :ds)
        fc = Forecaster(m; features=fs, freq=Day(1))
        @test (fc.strategy, fc.target, fc.time, fc.id, fc.transform) ==
              (Recursive(), :y, :ds, nothing, NoTransform())
    end

    @testset "features must be a FeatureSet, freq a positive Period" begin
        for m in (TestModels.LinAR(1.0, 0.0), DecisionTreeRegressor(max_depth=2),
                  EvoTreeRegressor(nrounds=5))
            @test_throws ArgumentError Forecaster(m; features=[Lag(1)], freq=Day(1))
            @test_throws "features must be a FeatureSet such as FeatureSet(Lag(1), " *
                         "Lag(7)), got $(repr([Lag(1)])). Wrap the features in " *
                         "FeatureSet(...)." Forecaster(
                m; features=[Lag(1)], freq=Day(1))
            @test_throws "got Lag(1). Wrap" Forecaster(m; features=Lag(1), freq=Day(1))
            for bad in (Day(0), Day(-1), Month(0), Month(-2))
                @test_throws ArgumentError Forecaster(m; features=FeatureSet(Lag(1)),
                                                      freq=bad)
                @test_throws "freq must be a positive period, got $bad. Pass e.g. " *
                             "freq=Day(1) or freq=Month(1)." Forecaster(
                    m; features=FeatureSet(Lag(1)), freq=bad)
            end
            for bad in (1, "day", Day(1) + Hour(1))
                @test_throws ArgumentError Forecaster(m; features=FeatureSet(Lag(1)),
                                                      freq=bad)
                @test_throws "freq must be a Dates.Period such as Day(1) or Month(1), " *
                             "got $(repr(bad)); a sum such as Day(1) + Hour(1) is not " *
                             "accepted. Pass e.g. freq=Day(1)." Forecaster(
                    m; features=FeatureSet(Lag(1)), freq=bad)
            end
            @test Forecaster(m; features=FeatureSet(Lag(1)), freq=Hour(6)).freq == Hour(6)
        end
    end

    @testset "a Probabilistic model forecasts the mean of its prediction" begin
        @test_logs Forecaster(TestModels.ProbLinAR(0.8, 2.0); features=FeatureSet(Lag(1)),
                              freq=Day(1))
        panel = (unique_id=[fill("a", 40); fill("b", 30)], ds=[t[1:40]; t[1:30]],
                 y=[y[1:40]; 3 .* y[1:30]])
        for strat in (Recursive(), Direct(5)), (data, id) in ((df, nothing),
                                                              (panel, :unique_id))
            point(m) = forecast(fit(Forecaster(m; features=FeatureSet(Lag(1)),
                                               strategy=strat, freq=Day(1), id), data),
                                5).y_hat
            @test point(TestModels.ProbLinAR(0.8, 2.0)) ==
                  point(TestModels.LinAR(0.8, 2.0))
        end
        # A real one: EvoTrees' Gaussian model predicts Normal distributions.
        week = (ds=t, y=[10 + 5 * sin(2π * i / 7) for i in 1:n])
        fc = Forecaster(EvoTreeGaussian(nrounds=100, seed=1);
                        features=FeatureSet(Fourier(7, 1)), freq=Day(1))
        f = fit(fc, week)
        out = forecast(f, 7)
        vals = [rowvalues(Fourier(7, 1), Float64[], Date(0), n - 1 + s) for s in 1:7]
        future = (fourier_7_0_sin_1=first.(vals), fourier_7_0_cos_1=last.(vals))
        MMI = MachineLearningForecast.MLJModelInterface
        tm = only(f.models)
        dists = MMI.predict(tm.model, tm.fitresult, MMI.reformat(tm.model, future)...)
        @test out.y_hat isa Vector{Float64} && out.y_hat == mean.(dists)
        truth = [10 + 5 * sin(2π * i / 7) for i in n .+ (1:7)]
        @test maximum(abs.(out.y_hat .- truth)) < 1
        r = backtest(fc, week; horizon=7, windows=2)
        @test all(isfinite, r.folds.y_hat)
    end

    @testset "the precompile workload runs without logging" begin
        @test_logs MachineLearningForecast._precompile_workload()
    end

    @testset "recursive oracle: LinAR closed-form iteration" begin
        # ŷ_{n+1} = a*y_n + b, then iterated on its own predictions.
        a, b = 0.8, 2.0
        fc = Forecaster(TestModels.LinAR(a, b); features=FeatureSet(Lag(1)),
                        strategy=Recursive(), freq=Day(1))
        fitted = fit(fc, df)
        h = 10
        fcast = forecast(fitted, h)
        expected = Float64[]
        prev = y[end]
        for _ in 1:h
            prev = a * prev + b
            push!(expected, prev)
        end
        @test fcast.y_hat ≈ expected
        @test fcast.ds == [t[end] + Day(s) for s in 1:h]
        @test keys(fcast) == (:ds, :y_hat)
    end

    @testset "recursive oracle: naive ŷ = y_lag_1" begin
        fc = Forecaster(TestModels.EchoColumn(:y_lag_1); features=FeatureSet(Lag(1)),
                        strategy=Recursive(), freq=Day(1))
        fcast = forecast(fit(fc, df), 5)
        @test all(fcast.y_hat .== y[end])   # naive carries the last value forward
    end

    @testset "recursive feedback reaches rolling windows" begin
        # Each step's rolling mean includes earlier predictions.
        fc = Forecaster(TestModels.EchoColumn(:y_rollmean_2_lag_1);
                        features=FeatureSet(Lag(1), RollingMean(2)),
                        strategy=Recursive(), freq=Day(1))
        fcast = forecast(fit(fc, df), 3)
        e1 = (y[end-1] + y[end]) / 2
        e2 = (y[end] + e1) / 2
        e3 = (e1 + e2) / 2
        @test fcast.y_hat ≈ [e1, e2, e3]
    end

    @testset "Fourier continuity across the train/forecast boundary" begin
        f = Fourier(7, 1)
        name = MachineLearningForecast.outputnames(f)[1]   # the sin_1 column
        fc = Forecaster(TestModels.EchoColumn(name);
                        features=FeatureSet(Lag(1), f), strategy=Recursive(), freq=Day(1))
        fcast = forecast(fit(fc, df), 14)
        # Training index is 0..n-1, so step s has index n-1+s.
        expected = [sin(2π * (n - 1 + s) / 7) for s in 1:14]
        @test fcast.y_hat ≈ expected
        # the series y_t = sin(2πt/7) continues with no phase jump:
        full = [sin(2π * k / 7) for k in 0:(n + 13)]
        @test fcast.y_hat ≈ full[(n + 1):(n + 14)]
    end

    @testset "exogenous handling" begin
        dfe = (ds=t, y=y, promo=Float64.(iseven.(1:n)))
        fc = Forecaster(TestModels.EchoColumn(:promo);
                        features=FeatureSet(Lag(1), Exogenous(:promo)),
                        strategy=Recursive(), freq=Day(1))
        fitted = fit(fc, dfe)

        err = try forecast(fitted, 3) catch e; e end
        @test err isa ArgumentError
        @test occursin("Exogenous(:promo)", err.msg)
        @test occursin("future", err.msg)
        @test occursin(string(t[end] + Day(1)), err.msg)
        @test occursin(string(t[end] + Day(3)), err.msg)

        bad_cols = (ds=t[end] .+ Day.(1:3), other=zeros(3))
        err = try forecast(fitted, 3; future=bad_cols) catch e; e end
        @test err isa ArgumentError && occursin(":promo", err.msg)
        @test_throws "`future` is missing column :promo; it must contain the time " *
                     "column :ds and the exogenous column :promo." forecast(
            fitted, 3; future=bad_cols)

        short = (ds=t[end] .+ Day.(1:2), promo=zeros(2))
        err = try forecast(fitted, 3; future=short) catch e; e end
        @test err isa ArgumentError && occursin(string(t[end] + Day(3)), err.msg)

        holed = (ds=t[end] .+ Day.(1:3), promo=[1.0, missing, 0.0])
        @test_throws ArgumentError forecast(fitted, 3; future=holed)
        @test_throws "`future` has a missing value in exogenous column :promo at time " *
                     "$(t[end] + Day(2)). Provide complete exogenous values for every " *
                     "future step." forecast(fitted, 3; future=holed)

        future = (ds=[t[end] + Day(3), t[end] + Day(1), t[end] + Day(2), t[end] + Day(9)],
                  promo=[30.0, 10.0, 20.0, 99.0])
        fcast = forecast(fitted, 3; future=future)
        @test fcast.y_hat == [10.0, 20.0, 30.0]   # EchoColumn(:promo)

        fc2 = Forecaster(TestModels.LinAR(1.0, 0.0);
                         features=FeatureSet(Lag(1), Exogenous(:absent)), freq=Day(1))
        @test_throws ArgumentError fit(fc2, dfe)
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fc3 = Forecaster(m; features=FeatureSet(Lag(1), Exogenous(:absent)),
                             freq=Day(1))
            @test_throws "Exogenous(:absent) but column :absent is not present in the " *
                         "training data. Available columns: ds, y, promo. Add column " *
                         ":absent to the data" fit(fc3, dfe)
        end
    end

    @testset "future without exogenous features warns and is ignored" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        fitted = fit(fc, df)
        extra = (ds=t[end] .+ Day.(1:2), promo=zeros(2))
        fcast = @test_logs (:warn, r"ignored") forecast(fitted, 2; future=extra)
        @test length(fcast.y_hat) == 2
        warning = "`future` was passed but the feature set has no Exogenous " *
                  "features; it will be ignored."
        @test_logs (:warn, warning) forecast(fitted, 2; future=extra)
    end

    @testset "the keyword is future; new_data is rejected" begin
        fce = Forecaster(TestModels.EchoColumn(:promo);
                         features=FeatureSet(Lag(1), Exogenous(:promo)),
                         strategy=Recursive(), freq=Day(1))
        fitted = fit(fce, (ds=t, y=y, promo=Float64.(iseven.(1:n))))
        grid = t[end] .+ Day.(1:3)
        ok = (ds=grid, promo=[1.0, 2.0, 3.0])
        @test forecast(fitted, 3; future=ok).y_hat == [1.0, 2.0, 3.0]
        @test_throws MethodError forecast(fitted, 3; new_data=ok)
        @test_throws MethodError forecast(fitted, 3; future=ok, new_data=ok)

        dup = (ds=[grid[1], grid[1], grid[2], grid[3]], promo=[1.0, 2.0, 3.0, 4.0])
        @test_throws "`future` has duplicate timestamps in column :ds; first duplicate " *
                     "at $(grid[1]). Deduplicate `future` — otherwise which row " *
                     "supplies each forecast step is arbitrary." (
            forecast(fitted, 3; future=dup))
        short = (ds=grid[1:2], promo=[1.0, 2.0])
        @test_throws "`future` is missing 1 of the 3 timestamps required for the " *
                     "forecast grid $(grid[1]) … $(grid[3]); first missing: " *
                     "$(grid[3]). Provide exogenous values for every future step." (
            forecast(fitted, 3; future=short))
        @test_throws "without `future`. Pass a table with columns (:ds, :promo)" (
            forecast(fitted, 3))
        @test_throws "Every `future` row needs a timestamp; drop or fill that row." (
            forecast(fitted, 3; future=(
                ds=Union{Missing,Date}[grid[1], missing, grid[2], grid[3]],
                promo=[1.0, 9.0, 2.0, 3.0])))
    end

    @testset "time column validation" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        gap = (ds=[t[1:10]; t[12:20]], y=Float64.(1:19))
        err = try fit(fc, gap) catch e; e end
        @test err isa ArgumentError && occursin("gap", err.msg)
        @test occursin(string(t[10]), err.msg)   # first gap after this timestamp
        dup = (ds=[t[1:10]; [t[10]]; t[11:19]], y=Float64.(1:20))
        err = try fit(fc, dup) catch e; e end
        @test err isa ArgumentError && occursin("duplicate", err.msg)
        unsorted = (ds=reverse(t), y=y)
        err = try fit(fc, unsorted) catch e; e end
        @test err isa ArgumentError && occursin("not sorted", err.msg)
        # a wrong freq puts rows off the declared grid
        fc_w = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                          freq=Week(1))
        @test_throws ArgumentError fit(fc_w, df)
        @test_throws "time column :ds has the timestamp 2022-01-02 at row 2, which does " *
                     "not lie on the freq=1 week grid starting at 2022-01-01" fit(fc_w, df)
        @test_throws ArgumentError fit(fc, (when=t, y=y))
        @test_throws "time column :ds not found in the data. Available columns: when, " *
                     "y. Pass time= with the name of your time column, or rename it to " *
                     ":ds." fit(fc, (when=t, y=y))
        @test_throws "target column :y not found in the data. Available columns: ds, " *
                     "value. Pass target= with the name of your target column" fit(
            fc, (ds=t, value=y))
        @test_throws "Available columns: none, the table has no columns." fit(
            fc, NamedTuple())
        ym = Vector{Union{Missing,Float64}}(y)
        ym[7] = missing
        @test_throws ArgumentError fit(fc, (ds=t, y=ym))
        @test_throws "target column :y contains missing values (first at row 7). " *
                     "Impute or drop missing targets before fitting." fit(fc, (ds=t, y=ym))
    end

    @testset "empty time columns and non-numeric targets" begin
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fc = Forecaster(m; features=FeatureSet(Lag(1)), freq=Day(1))
            @test_throws ArgumentError fit(fc, (ds=Date[], y=Float64[]))
            @test_throws "time column :ds is empty. Provide at least one row." fit(
                fc, (ds=Date[], y=Float64[]))
            for (col, T) in ((string.(y), "String"), (Any[y...], "Any"))
                @test_throws ArgumentError fit(fc, (ds=t, y=col))
                @test_throws "target column :y has element type $T; expected a Real " *
                             "(numeric) target. Convert it to numbers before fitting, " *
                             "e.g. parse.(Float64, col) for strings or Float64.(col) " *
                             "for a Vector{Any} of numbers." fit(fc, (ds=t, y=col))
            end
        end
    end

    @testset "edge cases: horizon 1 and too little history" begin
        fc = Forecaster(TestModels.LinAR(0.8, 2.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        h1 = forecast(fit(fc, df), 1)
        @test h1.y_hat == [0.8 * y[end] + 2.0] && h1.ds == [t[end] + Day(1)]
        @test_throws MethodError forecast(fit(fc, df), 1.5)
        for m in (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            fs = FeatureSet(Lag(1), Lag(7))
            f = fit(Forecaster(m; features=fs, freq=Day(1)), df)
            @test forecast(f, 1).y_hat == forecast(f, 3).y_hat[1:1]
            # Direct's step-1 model trains on Recursive's rows.
            fd = fit(Forecaster(m; features=fs, strategy=Direct(1), freq=Day(1)), df)
            @test forecast(fd, 1).y_hat == forecast(f, 1).y_hat
            deep = Forecaster(m; features=FeatureSet(Lag(5)), freq=Day(1))
            for k in (3, 5)
                @test_throws ArgumentError fit(deep, (ds=t[1:k], y=y[1:k]))
                @test_throws "needs 5 history rows before the first usable training " *
                             "row, but the data has only $k rows. Provide at least 6 " *
                             "rows" fit(deep, (ds=t[1:k], y=y[1:k]))
            end
            @test fit(deep, (ds=t[1:6], y=y[1:6])).n_train == 6
        end
    end

    @testset "table genericity: columntable and rowtable agree" begin
        fc = Forecaster(TestModels.LinAR(0.9, 1.0);
                        features=FeatureSet(Lag(1), RollingMean(3), Calendar(:dayofweek)),
                        strategy=Recursive(), freq=Day(1))
        ref = forecast(fit(fc, df), 5)
        rt = Tables.rowtable(df)
        @test rt isa Vector{<:NamedTuple}
        @test forecast(fit(fc, rt), 5).y_hat == ref.y_hat
        dt = Tables.dictcolumntable(df)
        @test forecast(fit(fc, dt), 5).y_hat == ref.y_hat
        @test_throws ArgumentError fit(fc, 42)
        @test_throws "expected a Tables.jl-compatible table (NamedTuple of vectors, " *
                     "DataFrame, CSV.File, ...), got Int64. Convert your data to a " *
                     "table first." fit(fc, 42)
        @test Tables.istable(ref)
        @test Tables.columntable(ref) == ref
    end

    @testset "misc" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        fitted = fit(fc, df)
        @test_throws ArgumentError forecast(fitted, 0)
        @test_throws "forecast horizon h must be ≥ 1, got 0. Pass the number of steps " *
                     "to forecast, e.g. forecast(fitted, 28)." forecast(fitted, 0)
        @test_throws ArgumentError forecast(fitted, -2)
        for bad in (true, false)
            @test_throws "forecast horizon h must be a whole number, not the Bool " *
                         "$bad. Pass the number of steps to forecast, e.g. " *
                         "forecast(fitted, 28)." forecast(fitted, bad)
        end
        @test_throws "h must be ≥ 1, got -2" forecast(fitted, -2)
        df2 = deepcopy(df)
        fit(fc, df2)
        @test df2 == df
        @test occursin("LinAR", sprint(show, fc))
        @test occursin("Recursive()", sprint(show, fc))
        @test occursin("trained on 60 rows", sprint(show, fitted))
        th = collect(DateTime(2022, 1, 1):Hour(1):DateTime(2022, 1, 3, 11))
        dfh = (ds=th, y=Float64.(1:length(th)))
        fch = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                         freq=Hour(1))
        fh = forecast(fit(fch, dfh), 3)
        @test fh.ds == [th[end] + Hour(s) for s in 1:3]
    end

    @testset "fitted models go through MLJModelInterface directly" begin
        # XGBoost, LightGBM and others need MMI's full data interface.
        fm = fit(Forecaster(TestModels.MatrixMean(); features=FeatureSet(Lag(1), Lag(2)),
                            freq=Day(1)), df)
        @test only(fm.models).fitresult == sum(df.y[3:end]) / (length(df.y) - 2)
        @test forecast(fm, 2).y_hat == fill(only(fm.models).fitresult, 2)
        fc = Forecaster(TestModels.MeanModel(); features=FeatureSet(Lag(1)), freq=Day(1))
        f = fit(fc, df)
        m = only(f.models)
        @test m isa MachineLearningForecast.TrainedModel
        @test m.model isa TestModels.MeanModel && m.model !== fc.model
        @test m.fitresult == sum(df.y[2:end]) / (length(df.y) - 1)
        @test forecast(f, 2).y_hat == fill(m.fitresult, 2)
        fd = fit(Forecaster(TestModels.MeanModel(); features=FeatureSet(Lag(1)),
                            strategy=Direct(3), freq=Day(1)), df)
        @test length(fd.models) == 3 && all(x -> x isa MachineLearningForecast.TrainedModel,
                                            fd.models)
        # As MLJBase did, fit cleans each copy and warns, not the prototype.
        proto = TestModels.Clamped(0)
        fcc = Forecaster(proto; features=FeatureSet(Lag(1)), freq=Day(1))
        fitted = @test_logs (:warn, "k must be ≥ 1; resetting k to 1.") fit(fcc, df)
        @test proto.k == 0 && only(fitted.models).model.k == 1
        @test_logs (:warn, r"declares input_scitype .*Count") Forecaster(
            TestModels.CountInput(); features=FeatureSet(Lag(1)), freq=Day(1))
        @test_logs Forecaster(TestModels.NoInputTrait(); features=FeatureSet(Lag(1)),
                              freq=Day(1))
        for model in (EvoTreeRegressor(nrounds=5),
                      DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
            @test_logs Forecaster(model; features=FeatureSet(Lag(1)), freq=Day(1))
        end
    end

    @testset "fit, backtest and tune leave the model prototype unchanged" begin
        dfr = (ds=t, y=10 .+ sin.(2π .* (1:n) ./ 7) .+ randn(StableRNG(4), n))
        fs = FeatureSet(Lag(1), Lag(2), Lag(7), RollingMean(3))
        # MLJ's == ignores RNG state, so compare every field.
        same(a, b) = all(k -> isequal(getfield(a, k), getfield(b, k)),
                         fieldnames(typeof(a)))
        for m in (DecisionTreeRegressor(n_subfeatures=2, rng=StableRNG(1)),
                  EvoTreeRegressor(nrounds=10, rowsample=0.5, colsample=0.5)),
            s in (Recursive(), Direct(3))
            before = deepcopy(m)
            fc = Forecaster(m; features=fs, strategy=s, freq=Day(1))
            first_fit = forecast(fit(fc, dfr), 3).y_hat
            @test same(m, before)
            @test forecast(fit(fc, dfr), 3).y_hat == first_fit
            backtest(fc, dfr; horizon=3, initial=50, step=5)
            @test same(m, before)
            r = tune(fc, dfr; grid=(features=[fs, fs, fs],), horizon=3, initial=50,
                     step=5)
            @test same(m, before)
            @test allequal(r.table.mean_score)
        end
    end

    @testset "a missing exogenous value is reported at the user's row" begin
        m30 = 30
        t30 = collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(m30 - 1))
        promo = Vector{Union{Missing,Float64}}(1.0:m30)
        promo[10] = missing
        y30 = Float64.(1:m30) .+ 10
        single = (ds=t30, y=y30, promo=promo)
        phrase = "feature column :promo has a missing value at row 10 (time 2022-01-10)"
        fs = FeatureSet(Lag(1), Exogenous(:promo))
        models = (EvoTreeRegressor(nrounds=5),
                  DecisionTreeRegressor(max_depth=2, rng=StableRNG(1)))
        for m in models, strat in (Recursive(), Direct(2)),
            tr in (NoTransform(), Difference(3), LogRatio(3),
                   TransformChain(Difference(1), LogRatio(3)))
            fc = Forecaster(m; features=fs, strategy=strat, freq=Day(1), transform=tr)
            @test_throws ArgumentError fit(fc, single)
            @test_throws phrase fit(fc, single)
            # the same fit as a panel already reports the user's row
            panel = (unique_id=fill("a", m30), ds=t30, y=y30, promo=promo)
            fcp = Forecaster(m; features=fs, strategy=strat, freq=Day(1), id=:unique_id,
                             transform=tr)
            @test_throws "series unique_id=\"a\": " * phrase fit(fcp, panel)
        end
    end
    @testset "single-series fit errors keep their order and text" begin
        fc = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                        freq=Day(1))
        @test_throws "time column :ds not found in the data" fit(fc, (a=[1.0, 2.0],))
        d3, d1, d2 = Date(2022, 1, 3), Date(2022, 1, 1), Date(2022, 1, 2)
        @test_throws "time column :ds is not sorted; row 2 (2022-01-01) comes after " *
                     "2022-01-03" fit(fc, (ds=[d3, d1, d2],))
        @test_throws "target column :y not found" fit(fc, (ds=[d1, d2, d3],))
        kc = Forecaster(TestModels.KeepTrain(); features=FeatureSet(Lag(1)), freq=Day(1))
        f = fit(kc, (ds=[d1, d2], y=[1.0, 2.0]))
        @test only(f.models).fitresult.y == [2.0]
        @test only(f.models).fitresult.X.y_lag_1 == [1.0]
        for (tr, k) in ((Difference(5), 5), (LogRatio(7), 7))
            fct = Forecaster(TestModels.LinAR(1.0, 0.0); features=FeatureSet(Lag(1)),
                             freq=Day(1), transform=tr)
            tbl = (ds=collect(d1:Day(1):d1 + Day(k - 1)), y=Float64.(1:k))
            @test_throws("the feature set needs 1 history rows and the transform " *
                         "drops $k, so a series needs at least $(k + 2) rows, but " *
                         "the data has only $k rows. Provide at least $(k + 2) rows " *
                         "or reduce lags/windows or the transform's lag/window.",
                         fit(fct, tbl))
        end
        model = TestModels.LinAR(0.5, 1.0)
        tbl = (ds=collect(d1:Day(1):d1 + Day(19)), y=Float64.(1:20))
        same, keep = deepcopy(tbl), deepcopy(model)
        fit(Forecaster(model; features=FeatureSet(Lag(1)), freq=Day(1),
                       strategy=Direct(2)), tbl)
        @test tbl == same && model == keep
    end

    @testset "forecast allocates its buffers once, not per step" begin
        M = MachineLearningForecast
        bytes(f, args...) = (f(args...); @allocated f(args...))
        feats = FeatureSet(Lag(1), Lag(7), RollingMean(7))
        d = (ds=collect(Date(2022, 1, 1):Day(1):Date(2022, 1, 1) + Day(399)),
             y=50 .+ 2 .* sin.(1:400))
        f = fit(Forecaster(TestModels.LinAR(0.9, 1.0); features=feats, freq=Day(1)), d)
        perstep = (bytes(forecast, f, 200) - bytes(forecast, f, 100)) / 100
        row1 = NamedTuple{Tuple(f.feature_names)}(Tuple([[1.0] for _ in f.feature_names]))
        pbytes = bytes(M._predict, only(f.models), row1)
        @test perstep <= pbytes + 64 * length(feats) + 256

        # A copy of the history would cost 8n bytes.
        nbig = 40_000
        big = (ds=collect(Date(2000, 1, 1):Day(1):Date(2000, 1, 1) + Day(nbig - 1)),
               y=Float64.(1:nbig))
        fd = fit(Forecaster(TestModels.MeanModel(); features=FeatureSet(Lag.(1:3)...),
                            strategy=Direct(5), freq=Day(1)), big)
        @test bytes(forecast, fd, 1) < sizeof(Float64) * nbig / 4
    end
end
