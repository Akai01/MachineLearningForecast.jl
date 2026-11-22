# Closed-form MLJ models to test without real learners.
module TestModels

using MLJModelInterface
using Tables
import Statistics
const MMI = MLJModelInterface

"Predicts `a * y_lag_1 + b`; ignores training entirely."
mutable struct LinAR <: MMI.Deterministic
    a::Float64
    b::Float64
end
MMI.fit(::LinAR, verbosity, X, y) = (nothing, nothing, NamedTuple())
MMI.predict(m::LinAR, fitresult, Xnew) =
    m.a .* Float64.(collect(Tables.getcolumn(Xnew, :y_lag_1))) .+ m.b
MMI.input_scitype(::Type{LinAR}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{LinAR}) = AbstractVector{MMI.Continuous}

"Echoes one feature column as the prediction; ignores training entirely."
mutable struct EchoColumn <: MMI.Deterministic
    col::Symbol
end
MMI.fit(::EchoColumn, verbosity, X, y) = (nothing, nothing, NamedTuple())
MMI.predict(m::EchoColumn, fitresult, Xnew) =
    Float64.(collect(Tables.getcolumn(Xnew, m.col)))
MMI.input_scitype(::Type{EchoColumn}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{EchoColumn}) = AbstractVector{MMI.Continuous}

"Always predicts the training-target mean (a real, if trivial, learner)."
mutable struct MeanModel <: MMI.Deterministic end
MMI.fit(::MeanModel, verbosity, X, y) = (sum(y) / length(y), nothing, NamedTuple())
MMI.predict(::MeanModel, fitresult, Xnew) =
    fill(Float64(fitresult), length(Tables.rows(Xnew)))
MMI.input_scitype(::Type{MeanModel}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{MeanModel}) = AbstractVector{MMI.Continuous}

"""
Memorises the training pairs `(X[:, col] => y)` and, at predict time, looks the
query row's `col` value up in that map (`NaN` if absent).

This is the only test model that *reveals which target each feature row was
trained against*, so it is what pins `Direct`'s column anchoring: if the
time/exogenous columns are shifted relative to the target, the memorised map is
shifted too and the looked-up prediction moves with it. Use distinct `col`
values so the map has no colliding keys.
"""
mutable struct PairLookup <: MMI.Deterministic
    col::Symbol
end
function MMI.fit(m::PairLookup, verbosity, X, y)
    keys_ = Float64.(collect(Tables.getcolumn(X, m.col)))
    map_ = Dict{Float64,Float64}(k => Float64(v) for (k, v) in zip(keys_, y))
    return (map_, nothing, NamedTuple())
end
MMI.predict(m::PairLookup, fitresult, Xnew) =
    [get(fitresult, Float64(v), NaN) for v in Tables.getcolumn(Xnew, m.col)]
MMI.input_scitype(::Type{PairLookup}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{PairLookup}) = AbstractVector{MMI.Continuous}

"A Probabilistic (non-Deterministic) model, to exercise construction warnings."
mutable struct DummyProb <: MMI.Probabilistic end
MMI.fit(::DummyProb, verbosity, X, y) = (nothing, nothing, NamedTuple())

"Neither Deterministic nor Probabilistic, to exercise construction warnings."
mutable struct DummyInterval <: MMI.Interval end
MMI.fit(::DummyInterval, verbosity, X, y) = (nothing, nothing, NamedTuple())

"A predicted distribution that only knows its mean."
struct PointDist
    μ::Float64
end
Statistics.mean(d::PointDist) = d.μ

"Probabilistic `LinAR`: predicts a `PointDist` at `a * y_lag_1 + b`."
mutable struct ProbLinAR <: MMI.Probabilistic
    a::Float64
    b::Float64
end
MMI.fit(::ProbLinAR, verbosity, X, y) = (nothing, nothing, NamedTuple())
MMI.predict(m::ProbLinAR, fitresult, Xnew) =
    PointDist.(m.a .* Float64.(collect(Tables.getcolumn(Xnew, :y_lag_1))) .+ m.b)
MMI.input_scitype(::Type{ProbLinAR}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{ProbLinAR}) = AbstractVector{MMI.Continuous}

"Always predicts `NaN`, standing in for a learner that diverged."
mutable struct NaNModel <: MMI.Deterministic end
MMI.fit(::NaNModel, verbosity, X, y) = (nothing, nothing, NamedTuple())
MMI.predict(::NaNModel, fitresult, Xnew) = fill(NaN, length(Tables.rows(Xnew)))
MMI.input_scitype(::Type{NaNModel}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{NaNModel}) = AbstractVector{MMI.Continuous}

"Mean model whose `clean!` resets an invalid `k` to 1, with a warning message."
mutable struct Clamped <: MMI.Deterministic
    k::Int
end
function MMI.clean!(m::Clamped)
    m.k ≥ 1 && return ""
    m.k = 1
    return "k must be ≥ 1; resetting k to 1."
end
MMI.fit(m::Clamped, verbosity, X, y) = (m.k * sum(y) / length(y), nothing, NamedTuple())
MMI.predict(::Clamped, fitresult, Xnew) = fill(fitresult, length(Tables.rows(Xnew)))
MMI.input_scitype(::Type{Clamped}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{Clamped}) = AbstractVector{MMI.Continuous}

"Declares that it only takes Count inputs, unlike the package's Continuous ones."
mutable struct CountInput <: MMI.Deterministic end
MMI.input_scitype(::Type{CountInput}) = MMI.Table(MMI.Count)
MMI.target_scitype(::Type{CountInput}) = AbstractVector{MMI.Continuous}

"""
Mean model that reads its features with `MMI.matrix`, as XGBoost, LightGBM,
MLJLinearModels and NearestNeighborModels do; that needs MMI's full interface.
"""
mutable struct MatrixMean <: MMI.Deterministic end
function MMI.fit(::MatrixMean, verbosity, X, y)
    size(MMI.matrix(X), 1) == length(y) || error("rows and targets disagree")
    return (sum(y) / length(y), nothing, NamedTuple())
end
MMI.predict(::MatrixMean, fitresult, Xnew) = fill(fitresult, size(MMI.matrix(Xnew), 1))
MMI.input_scitype(::Type{MatrixMean}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{MatrixMean}) = AbstractVector{MMI.Continuous}

"""
Keeps what it was trained on: `fitresult` is `(X=copy of the training columns,
y=copy of the targets)`, and it predicts zeros. Read it back through
`fitted.models[i].fitresult`.
"""
mutable struct KeepTrain <: MMI.Deterministic end
MMI.fit(::KeepTrain, verbosity, X, y) =
    ((X=map(copy, Tables.columntable(X)), y=copy(y)), nothing, NamedTuple())
MMI.predict(::KeepTrain, fitresult, Xnew) = zeros(length(Tables.rows(Xnew)))
MMI.input_scitype(::Type{KeepTrain}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{KeepTrain}) = AbstractVector{MMI.Continuous}

"""
Records a copy of every `Xnew` it is asked to predict on, in `calls[]`, and
predicts zeros. `fit` trains a deep copy, so read the calls back through
`fitted.models[i].model.calls[]`.
"""
mutable struct KeepPredict <: MMI.Deterministic
    calls::Base.RefValue{Vector{Any}}
end
MMI.fit(::KeepPredict, verbosity, X, y) = (nothing, nothing, NamedTuple())
function MMI.predict(m::KeepPredict, fitresult, Xnew)
    push!(m.calls[], map(copy, Tables.columntable(Xnew)))
    return zeros(length(Tables.rows(Xnew)))
end
MMI.input_scitype(::Type{KeepPredict}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{KeepPredict}) = AbstractVector{MMI.Continuous}

"Declares no input scitype, so it is Unknown."
mutable struct NoInputTrait <: MMI.Deterministic end
MMI.target_scitype(::Type{NoInputTrait}) = AbstractVector{MMI.Continuous}

end

# Test-side views of the feature contract.
module FeatureHelpers

import MachineLearningForecast as M

export rowvalues, columns

"One row of `f`: its `featurevalues!` on a context built from the arguments."
function rowvalues(f, y, t, step=0, exog=nothing)
    out = fill(NaN, length(M.outputnames(f)))
    M.featurevalues!(out, f, M.FeatureContext(y, t, step, exog))
    return out
end

"The training columns of `f` over one series, as a `ColumnAccumulator`."
function columns(f, y, t, data; step0=0)
    out = M.ColumnAccumulator()
    M.materialize!(out, f, M.TrainingSeries(y, t, data, step0))
    return out
end

end
