# FeatureSet: an ordered collection of features, plus training-frame materialization.

"""
    FeatureSet(features::AbstractFeature...)
    FeatureSet(features::AbstractVector{<:AbstractFeature})

An ordered collection of features. Order matters: it fixes the column order of
the design matrix fed to the model. Output column names must be unique across
the whole set.

# Example
```julia
FeatureSet(Lag(1), Lag(7), RollingMean(7), Calendar(:dayofweek), Fourier(365.25, 3))
```
"""
struct FeatureSet
    features::Vector{AbstractFeature}
    function FeatureSet(features::Vector{AbstractFeature})
        isempty(features) && throw(ArgumentError(
            "FeatureSet needs at least one feature, e.g. FeatureSet(Lag(1))."))
        names = reduce(vcat, outputnames.(features))
        if !allunique(names)
            dups = unique([n for n in names if count(==(n), names) > 1])
            throw(ArgumentError(
                "FeatureSet produces duplicate column name$(length(dups) == 1 ? "" : "s") " *
                "$(join(repr.(dups), ", ")). Remove the duplicated feature(s) or rename " *
                "exogenous/custom columns."))
        end
        new(features)
    end
end
FeatureSet(fs::AbstractFeature...) = FeatureSet(collect(AbstractFeature, fs))
# Julia arrays are invariant, so `Vector{Lag}` is not a `Vector{AbstractFeature}`.
# Accept any vector of features so that `FeatureSet([Lag(k) for k in 1:7])` works
# (useful when building tuning grids programmatically).
FeatureSet(fs::AbstractVector{<:AbstractFeature}) = FeatureSet(collect(AbstractFeature, fs))

Base.iterate(fs::FeatureSet, s...) = iterate(fs.features, s...)
Base.length(fs::FeatureSet) = length(fs.features)
Base.eltype(::Type{FeatureSet}) = AbstractFeature
Base.:(==)(a::FeatureSet, b::FeatureSet) = a.features == b.features
Base.hash(fs::FeatureSet, h::UInt) = hash(fs.features, hash(FeatureSet, h))
Base.show(io::IO, fs::FeatureSet) =
    print(io, "FeatureSet(", join(string.(fs.features), ", "), ")")

"""
    outputnames(fs::FeatureSet) -> Vector{Symbol}

All output column names produced by the set, in stable order (feature order,
then each feature's own column order).
"""
outputnames(fs::FeatureSet) = reduce(vcat, outputnames.(fs.features))

"""
    minhistory(fs::FeatureSet) -> Int

The number of leading rows for which at least one feature is undefined; these
rows are dropped from the training frame.
"""
minhistory(fs::FeatureSet) = maximum(minhistory.(fs.features); init=0)

"""
    targetcolumnmask(fs::FeatureSet) -> Vector{Bool}

Mask over [`outputnames`](@ref)`(fs)`: `true` where the column is produced by a
[`TargetFeature`](@ref) (so it describes the target's own history and is
anchored to the *feature* row), `false` for time/exogenous columns (which are
anchored to the *target* row). `Direct` fitting needs this to shift the two
groups independently — see `src/direct.jl`.
"""
function targetcolumnmask(fs::FeatureSet)
    mask = Bool[]
    for f in fs
        istarget = f isa TargetFeature
        for _ in outputnames(f)
            push!(mask, istarget)
        end
    end
    return mask
end

"The exogenous columns required by the set (empty if none)."
exogenouscolumns(fs::FeatureSet) =
    reduce(vcat, [f.cols for f in fs.features if f isa ExogenousFeature]; init=Symbol[])

"""
    build_training_frame(fs, tbl, target, time) -> (X, y, keep)

Materialize every feature over the training columntable `tbl` and assemble the
design matrix. Returns `X` (a columntable of the feature columns only, all
`Float64`, in stable order), `y::Vector{Float64}` (the target for the kept
rows), and `keep::BitVector` (the kept-row mask over the original rows). The
first `minhistory(fs)` rows are dropped because at least one target feature is
undefined there.
"""
function build_training_frame(fs::FeatureSet, tbl::NamedTuple, target::Symbol, time::Symbol)
    y = target_vector(tbl, target)
    t = require_column(tbl, time, "time")
    out = ColumnAccumulator()
    for f in fs
        materialize!(out, f, y, t, tbl)
    end
    mh = minhistory(fs)
    n = length(y)
    n > mh || throw(ArgumentError(
        "not enough data: the feature set needs $mh history rows before the first " *
        "usable training row, but the data has only $n rows. Provide at least " *
        "$(mh + 1) rows or reduce lags/windows."))
    keep = falses(n)
    keep[(mh + 1):n] .= true
    # Interior missings can only come from missing exogenous values (target
    # features produce leading missings only, by construction).
    for (name, col) in out
        for i in (mh + 1):n
            ismissing(col[i]) && throw(ArgumentError(
                "feature column :$name has a missing value at row $i (time $(t[i])). " *
                "Missing values in exogenous columns are not supported; impute or " *
                "drop them before fitting."))
        end
    end
    X = NamedTuple{Tuple(first.(out))}(
        Tuple(convert(Vector{Float64}, col[(mh + 1):n]) for (_, col) in out))
    return X, y[keep], keep
end
