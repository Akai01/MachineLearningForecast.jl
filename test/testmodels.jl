# Closed-form MLJ models to test without real learners.
module TestModels

using MLJModelInterface
import MachineLearningForecast
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

"Has a `missing` and a `NaN` default hyperparameter, to render them."
mutable struct Hyper <: MMI.Deterministic
    a::Union{Missing,Float64}
    b::Float64
end
Hyper(; a=missing, b=NaN) = Hyper(a, b)

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

"""
Predicts zeros, but `extra` predictions too many (too few if negative): a
learner that breaks the one-prediction-per-row contract.
"""
mutable struct WrongCount <: MMI.Deterministic
    extra::Int
end
MMI.fit(::WrongCount, verbosity, X, y) = (nothing, nothing, NamedTuple())
MMI.predict(m::WrongCount, fitresult, Xnew) =
    zeros(max(length(Tables.rows(Xnew)) + m.extra, 0))
MMI.input_scitype(::Type{WrongCount}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{WrongCount}) = AbstractVector{MMI.Continuous}

"""
Predicts like `LinAR(a, b)`, but reads its input through `reshape` (which flags
the column as shared data on Julia 1.10) and, if `asmatrix`, returns an n×1
Matrix. `extra` predictions too many (too few if negative) break the
one-prediction-per-row contract.
"""
mutable struct ReshapeLinAR <: MMI.Deterministic
    a::Float64
    b::Float64
    extra::Int
    asmatrix::Bool
end
MMI.fit(::ReshapeLinAR, verbosity, X, y) = (nothing, nothing, NamedTuple())
function MMI.predict(m::ReshapeLinAR, fitresult, Xnew)
    out = m.a .* reshape(Tables.getcolumn(Xnew, :y_lag_1), :, 1) .+ m.b
    out = m.extra ≥ 0 ? vcat(out, zeros(m.extra, 1)) : out[1:(end + m.extra), :]
    return m.asmatrix ? out : vec(out)
end
MMI.input_scitype(::Type{ReshapeLinAR}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{ReshapeLinAR}) = AbstractVector{MMI.Continuous}

"""
Predicts zeros and keeps the `y_lag_1` column itself, not a copy, of every
`Xnew` it is asked about, in `calls[]`: it shows which arrays the forecast loop
hands to a model. Read them back through `fitted.models[1].model.calls[]`.
"""
mutable struct ColumnProbe <: MMI.Deterministic
    calls::Base.RefValue{Vector{Any}}
end
MMI.fit(::ColumnProbe, verbosity, X, y) = (nothing, nothing, NamedTuple())
function MMI.predict(m::ColumnProbe, fitresult, Xnew)
    push!(m.calls[], Tables.getcolumn(Xnew, :y_lag_1))
    return zeros(length(Tables.rows(Xnew)))
end
MMI.input_scitype(::Type{ColumnProbe}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{ColumnProbe}) = AbstractVector{MMI.Continuous}

"Declares no input scitype, so it is Unknown."
mutable struct NoInputTrait <: MMI.Deterministic end
MMI.target_scitype(::Type{NoInputTrait}) = AbstractVector{MMI.Continuous}

"""
A transform whose state type varies per series: a series starting above 50 is
left as is (`z === y`, state `nothing`), any other is shifted down by 5
(state `5.0`).
"""
struct VaryingState <: MachineLearningForecast.TargetTransform end
MachineLearningForecast.fit_transform(::VaryingState, y::AbstractVector{Float64}) =
    y[1] > 50 ? (y, nothing) : (y .- 5.0, 5.0)
MachineLearningForecast.untransform(::VaryingState, z::AbstractVector{Float64},
                                    ::Nothing) = collect(z)
MachineLearningForecast.untransform(::VaryingState, z::AbstractVector{Float64},
                                    m::Float64) = z .+ m

"""
A transform that drops its first `k` rows and declares it with `dropped_rows`.
It refuses a series of `k` rows or fewer, as `LastScale` does on an empty
series, so a test sees whether `fit_transform` ran on a series that should be
skipped.
"""
struct FussyDrop <: MachineLearningForecast.TargetTransform
    k::Int
end
function MachineLearningForecast.fit_transform(t::FussyDrop, y::AbstractVector{Float64})
    length(y) > t.k || throw(ArgumentError(
        "FussyDrop($(t.k)) was given $(length(y)) rows"))
    return y[(t.k + 1):end], nothing
end
MachineLearningForecast.untransform(::FussyDrop, z::AbstractVector{Float64},
                                    ::Nothing) = collect(z)
MachineLearningForecast.dropped_rows(t::FussyDrop) = t.k

"A transform that drops its first `k` rows without declaring it."
struct QuietDrop <: MachineLearningForecast.TargetTransform
    k::Int
end
MachineLearningForecast.fit_transform(t::QuietDrop, y::AbstractVector{Float64}) =
    (y[(t.k + 1):end], nothing)
MachineLearningForecast.untransform(::QuietDrop, z::AbstractVector{Float64},
                                    ::Nothing) = collect(z)

"A transform that drops `k` rows but declares only `d` of them."
struct UnderDeclared <: MachineLearningForecast.TargetTransform
    k::Int
    d::Int
end
MachineLearningForecast.fit_transform(t::UnderDeclared, y::AbstractVector{Float64}) =
    (y[(t.k + 1):end], nothing)
MachineLearningForecast.untransform(::UnderDeclared, z::AbstractVector{Float64},
                                    ::Nothing) = collect(z)
MachineLearningForecast.dropped_rows(t::UnderDeclared) = t.d

"""
A user scaled metric: MAE over the in-sample naive MAE of `y_train`.
Records every `y_train` it receives, and names itself after `k`.
"""
struct NaiveScaled
    k::Int
    seen::Vector{Vector{Float64}}
end
NaiveScaled(k::Int=1) = NaiveScaled(k, Vector{Float64}[])
function (s::NaiveScaled)(y, ŷ; y_train)
    push!(s.seen, collect(Float64, y_train))
    return MachineLearningForecast.mae(y, ŷ) /
           MachineLearningForecast.mae(y_train[2:end], y_train[1:(end - 1)])
end
MachineLearningForecast.metric_inputs(::NaiveScaled) = MachineLearningForecast.Scaled()
MachineLearningForecast.metric_name(s::NaiveScaled) = Symbol("naive_scaled_", s.k)

"A scaled metric left on the default name, `Symbol(x)`."
struct UnnamedScaled end
(::UnnamedScaled)(y, ŷ; y_train) = 1.0
MachineLearningForecast.metric_inputs(::UnnamedScaled) = MachineLearningForecast.Scaled()

"Metrics that break the protocol."
badtrait(y, ŷ) = MachineLearningForecast.mae(y, ŷ)
MachineLearningForecast.metric_inputs(::typeof(badtrait)) = true
struct OtherInput <: MachineLearningForecast.MetricInput end
otherkind(y, ŷ) = MachineLearningForecast.mae(y, ŷ)
MachineLearningForecast.metric_inputs(::typeof(otherkind)) = OtherInput()
struct BadName end
(::BadName)(y, ŷ) = 1.0
MachineLearningForecast.metric_name(::BadName) = "bad"

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

module AnchorProbes

using Dates
import MachineLearningForecast as M
import MachineLearningForecast: outputnames, minhistory, featurevalues!, anchor

"The last value of the history; anchored at the origin by its supertype."
struct LagOrigin <: M.TargetFeature end
"The same value as `LagOrigin`, anchored at the origin by `anchor`."
struct YOrigin <: M.AbstractFeature end
"The same value, but anchored at the target row against its supertype."
struct LagAtTarget <: M.TargetFeature end
"The day of week of the target row."
struct DowTarget <: M.AbstractFeature end
"A feature of the abstract type with no anchor."
struct NoAnchor <: M.AbstractFeature end

const Ylast = Union{LagOrigin,YOrigin,LagAtTarget,NoAnchor}
outputnames(::Ylast) = [:ylast]
minhistory(::Ylast) = 1
featurevalues!(out, ::Ylast, ctx) = (out[1] = ctx.y[end]; nothing)
anchor(::YOrigin) = M.AtOrigin()
anchor(::LagAtTarget) = M.AtTarget()

outputnames(::DowTarget) = [:dayofweek]
minhistory(::DowTarget) = 0
featurevalues!(out, ::DowTarget, ctx) = (out[1] = dayofweek(ctx.t); nothing)
anchor(::DowTarget) = M.AtTarget()

end
