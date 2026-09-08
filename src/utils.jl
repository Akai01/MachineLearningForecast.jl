# Table normalization, validation helpers, and the future time grid.
#
# MachineLearningForecast is Tables.jl-native: any Tables.jl-compatible source is accepted
# and normalized to a columntable (a NamedTuple of column vectors). All tabular
# return values are columntables too — convert them to your favorite table type
# (e.g. a DataFrame) if you prefer.

"""
    normalize_table(data) -> NamedTuple

Normalize any Tables.jl-compatible source (NamedTuple of vectors, vector of
NamedTuples, DataFrame, CSV.File, ...) to a columntable — a `NamedTuple` of
column vectors. Throws an `ArgumentError` if `data` does not satisfy the
Tables.jl interface. Columns are never mutated by MachineLearningForecast.
"""
function normalize_table(data)
    Tables.istable(data) || throw(ArgumentError(
        "expected a Tables.jl-compatible table (NamedTuple of vectors, DataFrame, " *
        "CSV.File, ...), got $(typeof(data)). Convert your data to a table first."))
    tbl = Tables.columntable(data)
    if !isempty(tbl)
        lens = map(length, values(tbl))
        if !allequal(lens)
            pairs_ = join((":$k has $(length(v)) row$(length(v) == 1 ? "" : "s")"
                           for (k, v) in pairs(tbl)), ", ")
            throw(ArgumentError(
                "table columns have unequal lengths ($pairs_). Every column must " *
                "have the same number of rows; check how the table was built."))
        end
    end
    return tbl
end

"Number of rows of a columntable."
nrows(tbl::NamedTuple) = isempty(tbl) ? 0 : length(first(tbl))

"Rows `r` of a columntable, as a columntable (columns are copied slices)."
rowsubset(tbl::NamedTuple, r) = map(v -> v[r], tbl)

function require_column(tbl::NamedTuple, col::Symbol, what::AbstractString)
    haskey(tbl, col) || throw(ArgumentError(
        "$what column :$col not found in the data. Available columns: " *
        "$(join(keys(tbl), ", "))."))
    return tbl[col]
end

"""
    validate_time_column(t, time, freq)

Validate that the time column `t` (named `time`, for error messages) is sorted,
free of duplicates, and gap-free with respect to the step `freq`. Throws an
`ArgumentError` describing the first offending timestamp otherwise.
"""
function validate_time_column(t::AbstractVector, time::Symbol, freq)
    n = length(t)
    n ≥ 1 || throw(ArgumentError("time column :$time is empty. Provide at least one row."))
    applicable(+, t[1], freq) || throw(ArgumentError(
        "time column :$time has element type $(eltype(t)), which does not support " *
        "`+ $freq`. The time column must be a Date/DateTime (or another type " *
        "supporting `+` with $(typeof(freq))) — if these are strings, parse them " *
        "first, e.g. `Date.(col, dateformat\"yyyy-mm-dd\")`."))
    issorted(t) || begin
        i = findfirst(i -> t[i] < t[i-1], 2:n) + 1
        throw(ArgumentError(
            "time column :$time is not sorted; row $i ($(t[i])) comes after $(t[i-1]). " *
            "Sort your data by :$time before fitting."))
    end
    ndup = 0
    firstdup = nothing
    ngap = 0
    firstgap = nothing
    # Compare against the grid ANCHORED at t[1] rather than stepping from the
    # previous row. For Month/Year steps these differ: a month-end series
    # (Jan 31, Feb 29, Mar 31, ...) is gap-free anchored, but stepping would
    # read Feb 29 + Month(1) = Mar 29 and report a spurious gap. `future_grid`
    # is anchored the same way, so validation and forecasting agree.
    for i in 2:n
        if t[i] == t[i-1]
            ndup += 1
            firstdup === nothing && (firstdup = t[i])
        elseif t[i] != t[1] + (i - 1) * freq
            ngap += 1
            firstgap === nothing && (firstgap = t[i-1])
        end
    end
    ndup == 0 || throw(ArgumentError(
        "time column :$time has $ndup duplicate timestamp$(ndup == 1 ? "" : "s"); " *
        "first duplicate at $firstdup. Aggregate or deduplicate your data before fitting."))
    ngap == 0 || throw(ArgumentError(
        "time column :$time has $ngap gap$(ngap == 1 ? "" : "s") for freq=$freq; " *
        "first gap after $firstgap. Reindex your data or resample before fitting."))
    return nothing
end

"""
    future_grid(t_start, freq, n_train, h) -> Vector

The `h` timestamps continuing a training series of `n_train` rows that began at
`t_start`: `t_start + (n_train - 1 + s) * freq` for `s in 1:h`.

Anchored at `t_start` rather than stepping from the last timestamp, so that
Month/Year steps behave consistently with [`validate_time_column`](@ref): a
month-end series (Jan 31, Feb 29, Mar 31, ...) continues to Apr 30, May 31
instead of drifting to the 29th/30th. For Day/Week/Hour steps the two are
identical.
"""
future_grid(t_start, freq, n_train::Integer, h::Integer) =
    [t_start + (n_train - 1 + s) * freq for s in 1:h]

"""
    target_vector(tbl, target) -> Vector{Float64}

Extract and validate the target column: must exist, contain no `missing`
values, and be convertible to `Float64`.
"""
function target_vector(tbl::NamedTuple, target::Symbol)
    col = require_column(tbl, target, "target")
    if any(ismissing, col)
        i = findfirst(ismissing, col)
        throw(ArgumentError(
            "target column :$target contains missing values (first at row $i). " *
            "Impute or drop missing targets before fitting."))
    end
    eltype(col) <: Union{Real,Missing} || throw(ArgumentError(
        "target column :$target has element type $(eltype(col)); expected a Real " *
        "(numeric) target."))
    vals = Float64.(col)
    j = findfirst(!isfinite, vals)
    j === nothing || throw(ArgumentError(
        "target column :$target contains the non-finite value $(vals[j]) at row $j. " *
        "Impute, drop, or clip non-finite targets before fitting — they propagate " *
        "silently through lag features into every forecast."))
    return vals
end
