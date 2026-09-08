# Pipeline tuning on backtest score, with an ask/tell strategy interface.
#
# The loop in `tune` is a pure ask/tell driver: it repeatedly calls
# `ask(strategy)` for the next candidate, evaluates it with `backtest`, and
# reports the score back via `tell!(strategy, candidate, score)`. Batch
# strategies (GridSearch, RandomSearch) are adapted onto this loop through an
# internal queue; sequential strategies (e.g. a Bayesian optimizer defined
# outside this package) drive it directly.

"""
    TuningStrategy

Abstract supertype of tuning search strategies for [`tune`](@ref). Shipped
strategies: [`GridSearch`](@ref) and [`RandomSearch`](@ref).

# Extending with your own (sequential) strategy

Subtype `TuningStrategy` and implement two methods; `tune` will drive them in
an ask/tell loop:

- [`ask`](@ref)`(s::MyStrategy) -> Union{NamedTuple,Nothing}` — return the next
  candidate: a NamedTuple of `Forecaster` field overrides (keys among `:model`,
  `:features`, `:strategy`, `:freq`, `:target`, `:time`), drawn from your
  search space. Return `nothing` to signal termination.
- [`tell!`](@ref)`(s::MyStrategy, candidate, score)` — receive the result of
  evaluating `candidate`: a `Float64` backtest score (lower is better), or
  `missing` if the candidate failed to evaluate.

Termination is signalled either by `ask` returning `nothing`, or by the
`max_evals` budget enforced by `tune`. Your strategy owns its search-space
representation (construct it with whatever space description it needs).

# Example
```julia
mutable struct FirstK <: TuningStrategy
    candidates::Vector{NamedTuple}
    i::Int
end
MachineLearningForecast.ask(s::FirstK) = s.i > length(s.candidates) ? nothing :
                             (c = s.candidates[s.i]; s.i += 1; c)
MachineLearningForecast.tell!(s::FirstK, candidate, score) = nothing
```
"""
abstract type TuningStrategy end

"""
    GridSearch()

Exhaustive search over the Cartesian product of the `grid` values passed to
[`tune`](@ref). Deterministic candidate order (first grid key varies fastest);
ties in score are broken by first-seen order.

# Example
```julia
tune(fc, df; grid=(model=[m1, m2], strategy=[Recursive(), Direct(28)]),
     tuner=GridSearch(), horizon=28, initial=730)
```
"""
struct GridSearch <: TuningStrategy end

"""
    RandomSearch(n; rng=Random.default_rng())

Random search: `n` candidates drawn from the `grid` passed to [`tune`](@ref),
sampling each grid key independently and uniformly (with replacement, so
duplicates are possible). Pass an explicit `rng` for reproducibility.

# Example
```julia
tune(fc, df; grid=(model=models, features=featuresets),
     tuner=RandomSearch(10; rng=StableRNG(1)), horizon=28, initial=730)
```
"""
struct RandomSearch{R<:Random.AbstractRNG} <: TuningStrategy
    n::Int
    rng::R
    function RandomSearch(n::Integer, rng::Random.AbstractRNG)
        n ≥ 1 || throw(ArgumentError("RandomSearch n must be ≥ 1, got $n."))
        new{typeof(rng)}(Int(n), rng)
    end
end
RandomSearch(n::Integer; rng::Random.AbstractRNG=Random.default_rng()) =
    RandomSearch(n, rng)

"""
    ask(strategy) -> Union{NamedTuple, Nothing}

Ask a [`TuningStrategy`](@ref) for its next candidate: a NamedTuple of
[`Forecaster`](@ref) field overrides (keys among `:model`, `:features`,
`:strategy`, `:freq`, `:target`, `:time`). Return `nothing` to signal that the
search is finished. Part of the public tuning-extension API — implement it for
your own strategy subtypes; see [`TuningStrategy`](@ref) for the full contract.
"""
function ask end

"""
    tell!(strategy, candidate, score)

Report an evaluation result back to a [`TuningStrategy`](@ref): `candidate` is
the NamedTuple previously returned by [`ask`](@ref), and `score` is the
`Float64` mean backtest metric (lower is better) or `missing` if the candidate
failed to evaluate. Part of the public tuning-extension API — implement it for
your own strategy subtypes; see [`TuningStrategy`](@ref) for the full contract.
"""
function tell! end

# ---------------------------------------------------------------------------
# Batch strategies: candidate generation + internal queue adapter
# ---------------------------------------------------------------------------

function _validate_grid(grid)
    grid isa NamedTuple || throw(ArgumentError(
        "grid must be a NamedTuple of Forecaster field names to candidate value " *
        "lists, e.g. grid=(model=[m1, m2], strategy=[Recursive(), Direct(28)])."))
    valid = fieldnames(Forecaster)
    bad = setdiff(keys(grid), valid)
    isempty(bad) || throw(ArgumentError(
        "grid has unknown key$(length(bad) == 1 ? "" : "s") " *
        "$(join(":" .* string.(bad), ", ")); valid keys are " *
        "$(join(":" .* string.(valid), ", "))."))
    for (k, v) in pairs(grid)
        (v isa Union{AbstractVector,Tuple}) && !isempty(v) || throw(ArgumentError(
            "grid key :$k must map to a nonempty vector or tuple of candidate " *
            "values, got $(repr(v))."))
    end
    return nothing
end

"Candidates proposed by a batch strategy over a declared grid, in order."
function candidates(::GridSearch, grid::NamedTuple)
    _validate_grid(grid)
    ks = keys(grid)
    return [NamedTuple{ks}(combo) for combo in vec(collect(Iterators.product(values(grid)...)))]
end

function candidates(t::RandomSearch, grid::NamedTuple)
    _validate_grid(grid)
    ks = keys(grid)
    return [NamedTuple{ks}(map(v -> rand(t.rng, v), values(grid))) for _ in 1:t.n]
end

# Adapter that serves a precomputed candidate list through the ask/tell loop.
mutable struct CandidateQueue
    candidates::Vector{NamedTuple}
    i::Int
end
CandidateQueue(cands::Vector{<:NamedTuple}) = CandidateQueue(collect(NamedTuple, cands), 1)

ask(q::CandidateQueue) = q.i > length(q.candidates) ? nothing :
                         (c = q.candidates[q.i]; q.i += 1; c)
tell!(::CandidateQueue, candidate, score) = nothing

# How `tune` turns a strategy into an ask/tell sequence. Sequential strategies
# are their own sequence (default); batch strategies materialize a queue.
_sequence(t::TuningStrategy, grid) = t
function _sequence(t::Union{GridSearch,RandomSearch}, grid)
    grid === nothing && throw(ArgumentError(
        "$(nameof(typeof(t)))() requires the grid keyword: pass " *
        "grid=(model=[...], features=[...], ...) to tune."))
    _validate_grid(grid)
    return CandidateQueue(candidates(t, grid))
end

# ---------------------------------------------------------------------------
# Spec reconstruction
# ---------------------------------------------------------------------------

# Rebuild the immutable Forecaster with some fields overridden. Hand-written
# constructor call (only 6 fields) — no Setfield dependency.
function reconstruct(fc::Forecaster; kwargs...)
    bad = setdiff(keys(kwargs), fieldnames(Forecaster))
    isempty(bad) || throw(ArgumentError(
        "cannot reconstruct Forecaster with unknown field$(length(bad) == 1 ? "" : "s") " *
        "$(join(":" .* string.(bad), ", ")); valid fields are " *
        "$(join(":" .* string.(fieldnames(Forecaster)), ", "))."))
    return Forecaster(get(kwargs, :model, fc.model),
                      get(kwargs, :features, fc.features),
                      get(kwargs, :strategy, fc.strategy),
                      get(kwargs, :freq, fc.freq),
                      get(kwargs, :target, fc.target),
                      get(kwargs, :time, fc.time))
end

# ---------------------------------------------------------------------------
# TuneResult
# ---------------------------------------------------------------------------

"""
    TuneResult

Result of [`tune`](@ref):

- `table`: a columntable (NamedTuple of vectors) with one row per evaluated
  candidate — the overridden parameters (one column per key), `:mean_score`
  and `:std_score` (the mean and standard deviation of the metric across
  backtest folds; `std_score` is `missing` with a single fold), and `:error`
  (the error message if the candidate failed, `missing` otherwise). Failed
  candidates are excluded from ranking.
- `best::Forecaster`: the best candidate spec (NOT fitted).
- `best_fitted::FittedForecaster`: the best spec refit on ALL of the data.
"""
struct TuneResult
    table::NamedTuple
    best::Forecaster
    best_fitted::FittedForecaster
end

_ncandidates(r::TuneResult) = length(r.table.mean_score)

# Show a model candidate by the hyperparameters that actually DIFFER from a
# default-constructed prototype, so competing candidates in the top-5 table are
# distinguishable (showing the first two numeric fields made them identical).
function _short(x)
    x isa MLJModelInterface.Model || return string(x)
    T = typeof(x)
    fs = collect(fieldnames(T))
    proto = try T() catch; nothing end
    changed = proto === nothing ? fs :
              [f for f in fs if getfield(x, f) != getfield(proto, f)]
    isempty(changed) && (changed = fs)
    shown = first(changed, 3)
    body = join(("$f=$(getfield(x, f))" for f in shown), ", ")
    return string(nameof(T), "(", body, length(changed) > length(shown) ? ", …)" : ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::TuneResult)
    tbl = r.table
    n = _ncandidates(r)
    nfail = count(!ismissing, tbl.error)
    println(io, "TuneResult: $n candidate$(n == 1 ? "" : "s") evaluated" *
                (nfail > 0 ? " ($nfail failed)" : ""))
    paramkeys = [k for k in keys(tbl) if k ∉ (:mean_score, :std_score, :error)]
    okidx = findall(ismissing, tbl.error)
    order = okidx[sortperm(collect(Float64, tbl.mean_score[okidx]))]
    println(io, "Top $(min(5, length(order))) candidates (lower is better):")
    for (rank, i) in enumerate(order[1:min(5, end)])
        params = join(["$k=$(_short(tbl[k][i]))" for k in paramkeys], ", ")
        println(io, "  $rank. $params → ", round(tbl.mean_score[i]; sigdigits=5))
    end
    print(io, "Best spec: ", r.best)
end

Base.show(io::IO, r::TuneResult) =
    print(io, "TuneResult(", _ncandidates(r), " candidates)")

# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

"""
    tune(fc::Forecaster, data; grid=nothing, tuner=GridSearch(), max_evals=nothing,
         horizon, initial, step=horizon, metric=smape) -> TuneResult

Tune the whole forecasting pipeline on backtest score. Each candidate is a
NamedTuple of [`Forecaster`](@ref) field overrides (e.g. `model`, `features`,
`strategy`); the candidate spec is rebuilt from `fc` with those overrides,
backtested with [`backtest`](@ref)`(spec, data; horizon, initial, step,
metrics=(metric,))`, and scored by the mean of `metric` across folds (lower is
better).

- `grid`: NamedTuple mapping `Forecaster` field names to candidate value lists.
  Required for the batch strategies ([`GridSearch`](@ref), the Cartesian
  product; [`RandomSearch`](@ref), `n` uniform draws). A sequential
  user-defined [`TuningStrategy`](@ref) owns its own space and may ignore it.
- `max_evals`: optional evaluation budget; the loop stops after this many
  candidates even if the strategy proposes more.
- Candidate failures never abort the search: the error is caught, the score
  recorded as `missing` (and reported to the strategy via [`tell!`](@ref)),
  the message stored in the result table's `:error` column, and the candidate
  excluded from ranking.

Returns a [`TuneResult`](@ref) with the score table, the best spec, and the
best spec refit on all of `data`. Ties are broken by first-seen order.

# Example
```julia
result = tune(fc, df;
    grid = (model    = [EvoTreeRegressor(eta=0.05), EvoTreeRegressor(eta=0.1)],
            features = [FeatureSet(Lag(1), Lag(7)),
                        FeatureSet(Lag(1), Lag(7), Fourier(365.25, 3))]),
    horizon=28, initial=730, step=28, metric=smape)
result.best_fitted   # ready to forecast
```
"""
function tune(fc::Forecaster, data; grid=nothing, tuner::TuningStrategy=GridSearch(),
              max_evals::Union{Nothing,Integer}=nothing,
              horizon::Integer, initial::Integer, step::Integer=horizon,
              metric=smape)
    max_evals === nothing || max_evals ≥ 1 || throw(ArgumentError(
        "max_evals must be ≥ 1 (or nothing for no budget), got $max_evals."))
    # Validate here too, so a bad value is diagnosed before any candidate is built
    # rather than surfacing as a bare `step cannot be zero` from the range below.
    horizon ≥ 1 || throw(ArgumentError("tune horizon must be ≥ 1, got $horizon."))
    initial ≥ 1 || throw(ArgumentError("tune initial must be ≥ 1, got $initial."))
    step ≥ 1 || throw(ArgumentError("tune step must be ≥ 1, got $step."))
    tbl = normalize_table(data)
    seq = _sequence(tuner, grid)
    # Only warn about fits that will actually run: max_evals truncates the queue.
    if seq isa CandidateQueue
        planned = max_evals === nothing ? seq.candidates :
                  seq.candidates[1:min(length(seq.candidates), max_evals)]
        _warn_fit_count(fc, planned, tbl, horizon, initial, step)
    end

    cands = NamedTuple[]
    means = Union{Missing,Float64}[]
    stds = Union{Missing,Float64}[]
    errors = Union{Missing,String}[]
    while max_evals === nothing || length(cands) < max_evals
        cand = ask(seq)
        cand === nothing && break
        cand isa NamedTuple || throw(ArgumentError(
            "ask($(nameof(typeof(tuner)))) returned $(repr(cand)); the ask/tell " *
            "contract requires a NamedTuple of Forecaster field overrides, or " *
            "nothing to stop."))
        m, sd, err = _evaluate_candidate(fc, cand, tbl, horizon, initial, step, metric)
        tell!(seq, cand, m)
        push!(cands, cand); push!(means, m); push!(stds, sd); push!(errors, err)
    end
    isempty(cands) && throw(ArgumentError(
        "the tuning strategy proposed no candidates ($(nameof(typeof(tuner)))); " *
        "nothing to tune."))

    table = _tune_table(cands, means, stds, errors)
    ok = findall(!ismissing, means)
    if isempty(ok)
        msgs = join(unique(skipmissing(errors)), "\n  - ")
        throw(ErrorException(
            "all $(length(cands)) tuning candidates failed to evaluate. Errors:\n  - $msgs"))
    end
    best_idx = ok[argmin([means[i] for i in ok])]   # ties: first-seen wins (argmin)
    best = reconstruct(fc; cands[best_idx]...)
    best_fitted = fit(best, tbl)
    return TuneResult(table, best, best_fitted)
end

# TODO: Threads.@threads over candidates (backtest folds inside a candidate stay
# serial — MLJ machines aren't guaranteed thread-safe across all model packages).
function _evaluate_candidate(fc, cand, tbl, horizon, initial, step, metric)
    try
        cfc = reconstruct(fc; cand...)
        res = backtest(cfc, tbl; horizon, initial, step, metrics=(metric,))
        vals = [res.metrics.value[i] for i in eachindex(res.metrics.fold)
                if res.metrics.fold[i] > 0]
        sd = length(vals) > 1 ? Statistics.std(vals) : missing
        return Float64(Statistics.mean(vals)), sd, missing
    catch e
        e isa InterruptException && rethrow()
        return missing, missing, sprint(showerror, e)
    end
end

function _tune_table(cands, means, stds, errors)
    ks = Symbol[]
    for c in cands, k in keys(c)
        k in ks || push!(ks, k)
    end
    names = Tuple(vcat(ks, [:mean_score, :std_score, :error]))
    cols = Any[[haskey(c, k) ? c[k] : missing for c in cands] for k in ks]
    push!(cols, means); push!(cols, stds); push!(cols, errors)
    return NamedTuple{names}(Tuple(cols))
end

# Warn before large batch searches: candidates × folds × machines-per-fit.
function _warn_fit_count(fc, cands, tbl, horizon, initial, step)
    nfolds = length(initial:step:(nrows(tbl) - horizon))
    total = 0
    for c in cands
        total += nfolds * nmachines(get(c, :strategy, fc.strategy))
    end
    total > 500 && @warn(
        "tune is about to run $total model fits ($(length(cands)) candidates × " *
        "$nfolds backtest folds × machines per fit). This may take a while; " *
        "reduce the grid, increase step, or lower max_evals to shrink it.")
    return nothing
end

# TODO: continuous range types for sequential strategies (needed for a real
# TPE/Bayesian optimizer) would slot in here: a `ParamRange` declaration
# accepted in `grid` values and interpreted by the strategy's `candidates`/`ask`.
# v1 supports discrete candidate lists only.
