# Minimal MLJ models with fully deterministic, closed-form behavior, used to
# test the forecasting machinery independent of any real learner.
module TestModels

using MLJModelInterface
using Tables
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

end # module
