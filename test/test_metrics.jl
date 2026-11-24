import MachineLearningForecast: metric_inputs, metric_name, MetricInput, Point, Scaled

@testset "metrics" begin
    y = [10.0, 20.0, 30.0]
    ŷ = [12.0, 18.0, 33.0]
    @test mae(y, ŷ) ≈ (2 + 2 + 3) / 3
    @test rmse(y, ŷ) ≈ sqrt((4 + 4 + 9) / 3)
    @test mape(y, ŷ) ≈ (2 / 10 + 2 / 20 + 3 / 30) / 3
    @test smape(y, ŷ) ≈ (2 * 2 / 22 + 2 * 2 / 38 + 2 * 3 / 63) / 3
    @test mae(y, y) == 0.0
    @test smape(y, y) == 0.0

    @test (@test_logs (:warn, r"mape is undefined") mape([0.0, 1.0], [1.0, 1.0])) == Inf

    # smape 0/0 terms contribute 0
    @test smape([0.0, 1.0], [0.0, 1.0]) == 0.0

    y_train = [1.0, 3.0, 1.0, 3.0, 1.0]   # naive m=1 error: mean(|2,2,2,2|) = 2
    @test mase([10.0, 12.0], [11.0, 11.0]; y_train=y_train) ≈ 1.0 / 2
    # seasonal m=2: y_t == y_{t-2} everywhere, so Inf
    @test (@test_logs (:warn, r"mase is undefined") mase(y, ŷ; y_train=y_train, m=2)) == Inf
    @test_throws ArgumentError mase(y, ŷ; y_train=[1.0], m=1)
    @test_throws "mase needs length(y_train) > m; got length(y_train)=1 with m=1. " *
                 "Pass a longer y_train or a smaller m." mase(y, ŷ; y_train=[1.0], m=1)
    @test_throws ArgumentError mase(y, ŷ; y_train=y_train, m=0)
    for bad in (true, false)
        @test_throws "mase seasonality m must be a whole number, not the Bool $bad. " *
                     "Pass the season length in steps, e.g. m=7 for weekly " *
                     "seasonality on daily data." mase(y, ŷ; y_train=y_train, m=bad)
    end
    @test_throws "mase seasonality m must be ≥ 1, got 0. Pass the season length in " *
                 "steps, e.g. m=7 for weekly seasonality on daily data, or m=1 (the " *
                 "default)." mase(y, ŷ; y_train=y_train, m=0)

    @testset "rmsse" begin
        yr = [10.0, 14.0]
        ŷr = [11.0, 11.0]
        # Naive m=1 steps 1, 2, 3: mean square 14/3; rmse² = 5.
        @test rmsse(yr, ŷr; y_train=[1.0, 2.0, 4.0, 7.0]) ≈ sqrt(15 / 14)
        # Lag-2 steps 3, 5, 7: mean square 83/3.
        @test rmsse(yr, ŷr; y_train=[1.0, 2.0, 4.0, 7.0, 11.0], m=2) ≈ sqrt(15 / 83)
        tr = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
        sc = mean(abs2(tr[i] - tr[i - 1]) for i in 2:6)
        @test rmsse(yr, ŷr; y_train=tr) ≈ rmse(yr, ŷr) / sqrt(sc)
        @test rmsse(yr, ŷr; y_train=tr) ≈ rmsse(yr, ŷr; y_train=tr, m=1)
        # Scale-free.
        @test rmsse(1000 .* yr, 1000 .* ŷr; y_train=1000 .* tr) ≈
              rmsse(yr, ŷr; y_train=tr)
        @test rmsse(yr, yr; y_train=tr) == 0.0
        # Ranks methods within a series as rmse does.
        @test (rmsse(yr, ŷr; y_train=tr) < rmsse(yr, [12.0, 11.0]; y_train=tr)) ==
              (rmse(yr, ŷr) < rmse(yr, [12.0, 11.0]))
        @test metric_inputs(rmsse) isa Scaled
        @test (@test_logs (:warn, r"rmsse is undefined for a constant") rmsse(
            yr, ŷr; y_train=[3.0, 3.0, 3.0])) == Inf
        @test (@test_logs (:warn, r"rmsse is undefined") rmsse(
            yr, ŷr; y_train=[1.0, 2.0, 1.0, 2.0, 1.0], m=2)) == Inf

        @test_throws ArgumentError rmsse(yr, ŷr; y_train=[1.0], m=1)
        @test_throws "rmsse needs length(y_train) > m; got length(y_train)=2 with m=2. " *
                     "Pass a longer y_train or a smaller m." rmsse(
            yr, ŷr; y_train=[1.0, 2.0], m=2)
        @test_throws "rmsse seasonality m must be ≥ 1, got 0. Pass the season length " *
                     "in steps, e.g. m=7 for weekly seasonality on daily data, or m=1 " *
                     "(the default)." rmsse(yr, ŷr; y_train=tr, m=0)
        for bad in (true, false)
            @test_throws "rmsse seasonality m must be a whole number, not the Bool " *
                         "$bad. Pass the season length in steps, e.g. m=7 for weekly " *
                         "seasonality on daily data." rmsse(yr, ŷr; y_train=tr, m=bad)
        end
        @test_throws "metric inputs must have equal length, got length(y)=1 and " *
                     "length(ŷ)=2." rmsse([1.0], [1.0, 2.0]; y_train=tr)
        @test_throws "metric inputs are empty" rmsse(Float64[], Float64[]; y_train=tr)
    end

    @test_throws ArgumentError mae([1.0], [1.0, 2.0])
    @test_throws "metric inputs must have equal length, got length(y)=1 and " *
                 "length(ŷ)=2. Pass one forecast per actual, aligned element by " *
                 "element." mae([1.0], [1.0, 2.0])
    @test_throws ArgumentError rmse(Float64[], Float64[])
    @test_throws "metric inputs are empty: got length(y)=0 and length(ŷ)=0. Pass at " *
                 "least one actual and its forecast." rmse(Float64[], Float64[])
    for f in (mae, rmse, mape, smape)
        @test_throws "got length(y)=2 and length(ŷ)=1" f([1.0, 2.0], [1.0])
        @test_throws "metric inputs are empty" f(Float64[], Float64[])
    end
end

@testset "metric protocol" begin
    named = MachineLearningForecast._named_metrics
    y_train = [1.0, 3.0, 1.0, 3.0, 1.0]
    yr, ŷr = [10.0, 14.0], [11.0, 11.0]
    tr = [1.0, 2.0, 4.0, 7.0, 11.0]

    @testset "metric_inputs" begin
        for f in (mae, rmse, mape, smape, (a, b) -> mae(a, b))
            @test metric_inputs(f) === Point()
        end
        for f in (mase, rmsse, Mase(), Rmsse(12), TestModels.NaiveScaled())
            @test metric_inputs(f) === Scaled()
        end
        @test Point() isa MetricInput && Scaled() isa MetricInput
        @test !isdefined(MachineLearningForecast, :needs_ytrain)
    end

    @testset "metric_name" begin
        @test metric_name(mae) === :mae && metric_name(smape) === :smape
        @test metric_name(mase) === :mase && metric_name(rmsse) === :rmsse
        @test metric_name(Mase()) === :mase && metric_name(Mase(1)) === :mase
        @test metric_name(Mase(12)) === :mase_12
        @test metric_name(Rmsse()) === :rmsse && metric_name(Rmsse(7)) === :rmsse_7
        @test metric_name(TestModels.NaiveScaled(3)) === :naive_scaled_3
        u = TestModels.UnnamedScaled()
        @test metric_name(u) === Symbol(u)
    end

    @testset "Mase and Rmsse call mase and rmsse" begin
        @test Mase()(yr, ŷr; y_train=y_train) === mase(yr, ŷr; y_train=y_train)
        @test Mase()(yr, ŷr; y_train=y_train) ≈ 2.0 / 2   # mae 2, naive MAE 2
        @test Mase(2)(yr, ŷr; y_train=tr) === mase(yr, ŷr; y_train=tr, m=2)
        @test Mase(2)(yr, ŷr; y_train=tr) ≈ 2.0 / 5   # lag-2 steps 3, 5, 7
        @test Rmsse()(yr, ŷr; y_train=[1.0, 2.0, 4.0, 7.0]) ≈ sqrt(15 / 14)
        @test Rmsse(2)(yr, ŷr; y_train=tr) === rmsse(yr, ŷr; y_train=tr, m=2)
        @test Rmsse(2)(yr, ŷr; y_train=tr) ≈ sqrt(15 / 83)
        @test Mase(2)(1000 .* yr, 1000 .* ŷr; y_train=1000 .* tr) ≈
              Mase(2)(yr, ŷr; y_train=tr)
        @test typeof(Mase(3).m) === Int && Mase(Int32(3)).m === 3
        @test Rmsse(Int8(4)).m === 4
        @test (@test_logs (:warn, r"mase is undefined") Mase(2)(
            yr, ŷr; y_train=y_train)) == Inf
        @test (@test_logs (:warn, r"rmsse is undefined") Rmsse(2)(
            yr, ŷr; y_train=y_train)) == Inf
        @test_throws MethodError Mase(1.5)
        @test_throws MethodError Rmsse(2.0)
    end

    @testset "Mase and Rmsse validate m" begin
        eg = "Pass the season length in steps, e.g. %s(7) for weekly seasonality on " *
             "daily data"
        for (T, name) in ((Mase, "Mase"), (Rmsse, "Rmsse")), bad in (0, -1)
            @test_throws ArgumentError T(bad)
            @test_throws "$name seasonality m must be ≥ 1, got $bad. " *
                         replace(eg, "%s" => name) * ", or $name() (the default)." T(bad)
        end
        for (T, name) in ((Mase, "Mase"), (Rmsse, "Rmsse")), bad in (true, false)
            @test_throws "$name seasonality m must be a whole number, not the Bool " *
                         "$bad. " * replace(eg, "%s" => name) * "." T(bad)
        end
        big_m = big(2)^70
        for (T, name) in ((Mase, "Mase"), (Rmsse, "Rmsse"))
            @test_throws ArgumentError T(big_m)
            @test_throws "$name seasonality m must be at most $(typemax(Int)), got " *
                         "$big_m. " * replace(eg, "%s" => name) * "." T(big_m)
        end
        @test_throws "mase needs length(y_train) > m; got length(y_train)=3 with " *
                     "m=12. Pass a longer y_train or a smaller m." Mase(12)(
            yr, ŷr; y_train=[1.0, 2.0, 3.0])
        @test_throws "rmsse needs length(y_train) > m; got length(y_train)=3 with " *
                     "m=12. Pass a longer y_train or a smaller m." Rmsse(12)(
            yr, ŷr; y_train=[1.0, 2.0, 3.0])
        # The function forms keep their own hint.
        @test_throws "or m=1 (the default)." mase(yr, ŷr; y_train=tr, m=0)
        @test_throws "or m=1 (the default)." rmsse(yr, ŷr; y_train=tr, m=0)
    end

    @testset "metrics collections are named" begin
        pairs_(x) = [n => m for (n, m) in named(x)]
        @test pairs_((mae, rmse)) == [:mae => mae, :rmse => rmse]
        @test first.(named([mae, Mase(12), Rmsse(7)])) == [:mae, :mase_12, :rmsse_7]
        @test first.(named(Set([mae]))) == [:mae]
        @test first.(named(f for f in (mae, rmse))) == [:mae, :rmse]
        @test first.(named(Iterators.Stateful((mae, rmse)))) == [:mae, :rmse]
        @test isempty(named(())) && isempty(named(Any[]))
        @test pairs_((a=mae, b=Mase(12))) == [:a => mae, :b => Mase(12)]
        # A key names the metric, whatever metric_name says.
        @test first.(named((x=Mase(12),))) == [:x]
        mkm(k) = (a, b) -> k * mae(a, b)
        @test first.(named((small=mkm(1), large=mkm(2)))) == [:small, :large]
        @test first.(named((a=mae, b=mae))) == [:a, :b]
        @test first.(named((Mase(1), Mase(12)))) == [:mase, :mase_12]
        # A one-shot iterator is read once.
        it = Iterators.Stateful((mae, rmse))
        @test length(named(it)) == 2
    end

    @testset "collections that are not metrics are refused" begin
        for bad in (mae, :mae, "mae", (:mae,), ("mae", "rmse"), Dict(:mae => mae),
                    [mae, :rmse], 42, (:a => mae,), (mae, "rmse"), (a=:mae,),
                    (a=:b => mae,), Dict(:a => mae))
            @test_throws ArgumentError named(bad)
            @test_throws "got $(repr(bad)). Pass e.g. metrics=(mae, rmse)" named(bad)
        end
        bad = (a=:b => mae,)
        @test_throws "backtest metrics must be a tuple or vector of metric functions, " *
                     "got $(repr(bad)). Pass e.g. metrics=(mae, rmse), or metrics=(mae,) " *
                     "for one metric. Name a metric with a NamedTuple key, e.g. " *
                     "metrics=(error=mae, scaled=Mase(12)), not with a Pair." named(bad)
    end

    @testset "a type is not a metric" begin
        for (T, name) in ((Mase, "Mase"), (Rmsse, "Rmsse"))
            for bad in ((T,), [T], (a=T,), (mae, T))
                @test_throws ArgumentError named(bad)
                @test_throws "got the type $name; pass $name() or $name(12)" named(bad)
            end
        end
        @test_throws "backtest metric :a: got the type Mase; pass Mase() or Mase(12)" named(
            (a=Mase,))
        @test_throws "got the type NaiveScaled; pass an instance of it" named(
            (TestModels.NaiveScaled,))
    end

    @testset "metric_inputs and metric_name must answer the protocol" begin
        @test_throws ArgumentError named((TestModels.badtrait,))
        @test_throws "backtest metric :badtrait: metric_inputs returned true, which " *
                     "is not Point() or Scaled(). Define metric_inputs for your " *
                     "metric, e.g. MachineLearningForecast.metric_inputs(::typeof(f)) " *
                     "= MachineLearningForecast.Scaled()." named((TestModels.badtrait,))
        @test_throws "backtest metric :k: metric_inputs returned true" named(
            (k=TestModels.badtrait,))
        @test_throws ArgumentError named((TestModels.BadName(),))
        @test_throws "metric_name returned \"bad\", which is not a Symbol. Define " *
                     "metric_name for your metric to return one, e.g. " *
                     "MachineLearningForecast.metric_name(::MyMetric) = :my_metric." named(
            (TestModels.BadName(),))
        @test_throws "metric_inputs returned $(repr(TestModels.OtherInput())), " *
                     "which is not Point() or Scaled()." named((TestModels.otherkind,))
        # Keys name the metric, so metric_name is not asked.
        @test first.(named((k=TestModels.BadName(),))) == [:k]
    end

    @testset "metric names must be distinct" begin
        msg(n, name) = "backtest metrics must have distinct names, but $n metrics are " *
                       "named $name, so their per-fold rows would be merged into one " *
                       "wrong overall score. Remove the repeated metric, or name each " *
                       "one with a NamedTuple key (closures from one factory share a " *
                       "name), e.g. metrics=(a=mae, b=Mase(12))."
        @test_throws ArgumentError named((mae, mae))
        @test_throws msg(2, ":mae") named((mae, mae))
        @test_throws msg(2, ":mase") named((mase, Mase()))
        @test_throws msg(2, ":mase") named((Mase(1), mase))
        @test_throws msg(3, ":mae") named((mae, rmse, mae, mae))
        @test_throws msg(2, ":mase_12") named([Mase(12), Mase(12)])
        mkm(k) = (a, b) -> k * mae(a, b)
        @test_throws "2 metrics are named $(repr(Symbol(mkm(1))))" named(
            (mkm(1), mkm(2)))
    end
end
