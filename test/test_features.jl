@testset "features" begin
    # A 10-point series with hand-computable feature values.
    y = Float64.([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    t = collect(Date(2021, 3, 1):Day(1):Date(2021, 3, 10))
    df = (ds=t, y=y)

    @testset "constructor validation" begin
        @test_throws ArgumentError Lag(0)
        @test_throws ArgumentError Lag(-3)
        @test_throws ArgumentError RollingMean(0)
        @test_throws ArgumentError RollingMean(3; lag=0)   # lag=0 would leak
        @test_throws ArgumentError RollingStd(1)           # std needs window ≥ 2
        @test_throws ArgumentError RollingMin(2; lag=-1)
        @test_throws ArgumentError Diff(0)
        @test_throws ArgumentError Diff(1; lag=0)
        @test_throws ArgumentError Fourier(0, 3)
        @test_throws ArgumentError Fourier(7, 0)
        @test_throws ArgumentError Calendar()
        @test_throws ArgumentError Calendar(:weekday)      # unknown part
        @test_throws ArgumentError Exogenous()
        @test_throws ArgumentError CustomFeature(:f, identity, -1)
    end

    @testset "equality (features are value objects)" begin
        @test Lag(7) == Lag(7)
        @test Lag(7) != Lag(6)
        @test Calendar(:month) == Calendar(:month)
        @test hash(Calendar(:month)) == hash(Calendar(:month))
        @test RollingMean(7) != RollingMax(7)
        @test FeatureSet(Lag(1), Calendar(:month)) == FeatureSet(Lag(1), Calendar(:month))
        @test FeatureSet(Lag(1)) != FeatureSet(Lag(2))
    end

    @testset "outputnames" begin
        @test MachineLearningForecast.outputnames(Lag(7)) == [:y_lag_7]
        @test MachineLearningForecast.outputnames(RollingMean(7)) == [:y_rollmean_7_lag_1]
        @test MachineLearningForecast.outputnames(RollingStd(28; lag=2)) == [:y_rollstd_28_lag_2]
        @test MachineLearningForecast.outputnames(Diff(7)) == [:y_diff_7_lag_1]
        @test MachineLearningForecast.outputnames(Calendar(:dayofweek, :month)) == [:dayofweek, :month]
        @test MachineLearningForecast.outputnames(Fourier(7, 2)) ==
              [:fourier_7_0_sin_1, :fourier_7_0_sin_2, :fourier_7_0_cos_1, :fourier_7_0_cos_2]
        @test MachineLearningForecast.outputnames(Exogenous(:promo, :price)) == [:promo, :price]
        @test MachineLearningForecast.outputnames(CustomFeature(:hm, mean, 1)) == [:hm]
    end

    @testset "minhistory" begin
        @test MachineLearningForecast.minhistory(Lag(7)) == 7
        @test MachineLearningForecast.minhistory(RollingMean(7)) == 7            # window + lag - 1
        @test MachineLearningForecast.minhistory(RollingMean(7; lag=3)) == 9
        @test MachineLearningForecast.minhistory(Diff(7)) == 8                   # k + lag
        @test MachineLearningForecast.minhistory(Calendar(:month)) == 0
        @test MachineLearningForecast.minhistory(Fourier(7, 2)) == 0
        @test MachineLearningForecast.minhistory(Exogenous(:promo)) == 0
        @test MachineLearningForecast.minhistory(CustomFeature(:hm, mean, 4)) == 4
        fs = FeatureSet(Lag(1), RollingMean(3; lag=2), Calendar(:month))
        @test MachineLearningForecast.minhistory(fs) == 4
    end

    # Materialize one feature (with a given y) into a columntable.
    materialize(f, yv=y, data=df) = begin
        out = MachineLearningForecast.ColumnAccumulator()
        MachineLearningForecast.materialize!(out, f, yv, t, data)
        (; out...)
    end

    @testset "Lag exact values" begin
        col = materialize(Lag(2)).y_lag_2
        @test all(ismissing, col[1:2])
        @test collect(skipmissing(col)) == y[1:8]
    end

    @testset "RollingMean exact values (hand-computed)" begin
        col = materialize(RollingMean(3)).y_rollmean_3_lag_1
        @test all(ismissing, col[1:3])
        # row 4 covers y[1:3], row 10 covers y[7:9]
        @test col[4] ≈ mean(y[1:3])
        @test col[5] ≈ mean(y[2:4])
        @test col[10] ≈ mean(y[7:9])
    end

    @testset "leakage: RollingMean(3) at row t excludes y[t]" begin
        # Make y[t] an outlier; the feature at t must not move.
        y2 = copy(y); y2[6] = 1e6
        for f in (RollingMean(3), Lag(1), RollingStd(3), RollingMin(3), RollingMax(3),
                  Diff(2), CustomFeature(:hm, mean, 1))
            name = only(MachineLearningForecast.outputnames(f))
            @test materialize(f)[name][6] == materialize(f, y2)[name][6]
        end
    end

    @testset "rolling std/min/max and Diff exact values" begin
        @test materialize(RollingStd(3)).y_rollstd_3_lag_1[5] ≈ std(y[2:4])
        @test materialize(RollingMin(4)).y_rollmin_4_lag_1[6] == minimum(y[2:5])
        @test materialize(RollingMax(4; lag=2)).y_rollmax_4_lag_2[7] == maximum(y[2:5])
        col = materialize(Diff(3; lag=2)).y_diff_3_lag_2
        @test all(ismissing, col[1:5])
        @test col[6] == y[4] - y[1]
    end

    @testset "Calendar values" begin
        out = materialize(Calendar(:dayofweek, :month, :weekofyear))
        @test out.dayofweek[1] == Dates.dayofweek(Date(2021, 3, 1))
        @test all(out.month .== 3)
        @test out.weekofyear[1] == Dates.week(Date(2021, 3, 1))
    end

    @testset "Fourier values and index continuity" begin
        f = Fourier(7, 2)
        out = materialize(f)
        # training index is 0-based: row i has n = i-1
        @test out.fourier_7_0_sin_1[1] ≈ sin(0.0)
        @test out.fourier_7_0_sin_1[4] ≈ sin(2π * 3 / 7)
        @test out.fourier_7_0_cos_2[4] ≈ cos(2π * 2 * 3 / 7)
        # featurevalues continues the same index: value at index n equals the
        # materialized value at row n+1 — no phase jump at the boundary.
        vals = MachineLearningForecast.featurevalues(f, y, 9, nothing)
        @test collect(vals) ≈ [out[c][10] for c in MachineLearningForecast.outputnames(f)]
    end

    @testset "Exogenous materialization and errors" begin
        dfe = (ds=t, y=y, promo=collect(1:10) .% 2 .== 0)
        out = materialize(Exogenous(:promo), y, dfe)
        @test out.promo == Float64.(dfe.promo)
        @test_throws ArgumentError materialize(Exogenous(:absent), y, dfe)
        dfs = (ds=t, y=y, label=fill("a", 10))
        @test_throws ArgumentError materialize(Exogenous(:label), y, dfs)
        # featurevalues: missing column in the exogenous row errors clearly
        @test_throws ArgumentError MachineLearningForecast.featurevalues(Exogenous(:promo), y, t[1], (other=1.0,))
        @test MachineLearningForecast.featurevalues(Exogenous(:promo), y, t[1], (promo=true,)) == (1.0,)
    end

    @testset "CustomFeature" begin
        f = CustomFeature(:hist_mean, mean, 2)
        out = materialize(f)
        @test all(ismissing, out.hist_mean[1:2])
        @test out.hist_mean[3] ≈ mean(y[1:2])
        @test out.hist_mean[10] ≈ mean(y[1:9])
        @test MachineLearningForecast.featurevalues(f, y, t[1], nothing) == (mean(y),)
    end

    @testset "FeatureSet" begin
        fs = FeatureSet(Lag(1), Lag(7), Calendar(:month))
        @test length(fs) == 3
        @test collect(fs) == fs.features
        @test MachineLearningForecast.outputnames(fs) == [:y_lag_1, :y_lag_7, :month]
        @test_throws ArgumentError FeatureSet()
        @test_throws ArgumentError FeatureSet(Lag(1), Lag(1))          # duplicate columns
        @test_throws ArgumentError FeatureSet(Calendar(:month), Exogenous(:month))
        @test occursin("Lag(1)", sprint(show, FeatureSet(Lag(1))))
    end

    @testset "build_training_frame" begin
        fs = FeatureSet(Lag(2), RollingMean(2), Calendar(:dayofweek))
        X, ykept, keep = MachineLearningForecast.build_training_frame(fs, df, :y, :ds)
        mh = MachineLearningForecast.minhistory(fs)   # max(2, 2) = 2
        @test mh == 2
        @test length(ykept) == 8
        @test ykept == y[3:end]
        @test keep == [falses(2); trues(8)]
        @test keys(X) == (:y_lag_2, :y_rollmean_2_lag_1, :dayofweek)
        @test all(eltype(X[c]) == Float64 for c in keys(X))
        @test X.y_lag_2 == y[1:8]
        # too-short data errors with row counts in the message
        short = (ds=t[1:3], y=y[1:3])
        @test_throws ArgumentError MachineLearningForecast.build_training_frame(FeatureSet(Lag(5)), short, :y, :ds)
        # missing exogenous values error (interior missing)
        promo = Vector{Union{Missing,Float64}}(1.0:10.0)
        promo[5] = missing
        dfm = (ds=t, y=y, promo=promo)
        @test_throws ArgumentError MachineLearningForecast.build_training_frame(
            FeatureSet(Lag(1), Exogenous(:promo)), dfm, :y, :ds)
    end
end
