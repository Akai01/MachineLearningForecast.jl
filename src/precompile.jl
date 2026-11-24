# A trivial model lets the workload run fit and forecast.
struct _PrecompileMean <: MLJModelInterface.Deterministic end
MLJModelInterface.fit(::_PrecompileMean, verbosity, X, y) =
    (Statistics.mean(y), nothing, NamedTuple())
MLJModelInterface.predict(::_PrecompileMean, m, Xnew) = fill(m, length(first(Xnew)))
MLJModelInterface.target_scitype(::Type{_PrecompileMean}) =
    AbstractVector{MLJModelInterface.Continuous}

function _precompile_workload()
    t = collect(Date(2000, 1, 1):Month(1):Date(2004, 12, 1))
    y = [10 + sin(2π * i / 12) + i / 10 for i in eachindex(t)]
    single = (ds=t, y=y)
    panel = (unique_id=[fill("a", 60); fill("b", 48)], ds=[t; t[13:60]],
             y=[y; y[13:60] .+ 1])
    feats = FeatureSet(Lag(1), Lag(12), RollingMean(3), RollingStd(3), Diff(1),
                       Calendar(:month), Fourier(12, 1))
    chain = TransformChain(LastScale(12), Deseasonalize(12), LogRatio(12))
    for strategy in (Recursive(), Direct(3)), transform in (NoTransform(), chain),
        id in (nothing, :unique_id)
        data = id === nothing ? single : panel
        fc = Forecaster(_PrecompileMean(); features=feats, freq=Month(1), strategy,
                        transform, id)
        forecast(fit(fc, data), 3)
        metrics = id === nothing ? (mae, smape, mase, Mase(12)) :
                  (mae=mae, smape=smape, mase=mase, mase_12=Mase(12))
        backtest(fc, data; horizon=3, windows=2, metrics)
    end
    d = collect(Date(2020, 1, 1):Day(1):Date(2020, 3, 31))
    daily = (ds=d, y=[10 + sin(i) + i / 10 for i in eachindex(d)],
             promo=[Float64(i % 7 == 0) for i in eachindex(d)])
    ahead = d[end] .+ Day.(1:3)
    future = (ds=ahead, promo=zeros(3))
    dfeats = FeatureSet(Lag(1), Lag(7), RollingMean(7), Diff(1), Calendar(:dayofweek),
                        Fourier(7, 1), Exogenous(:promo))
    for strategy in (Recursive(), Direct(3))
        fc = Forecaster(_PrecompileMean(); features=dfeats, freq=Day(1), strategy)
        forecast(fit(fc, daily), 3; future)
    end
    return nothing
end

PrecompileTools.@compile_workload _precompile_workload()
