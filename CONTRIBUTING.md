# Contributing to MachineLearningForecast.jl

Thanks for your interest in improving MachineLearningForecast. Bug reports, documentation
fixes, new features and new tuning strategies are all welcome.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

```julia
julia> ]                        # enter Pkg mode
pkg> dev /path/to/MachineLearningForecast.jl
pkg> activate /path/to/MachineLearningForecast.jl
pkg> test                       # runs the full suite
```

Building the docs locally:

```bash
julia --project=docs docs/make.jl   # output in docs/build/
```

The docs build is warning-free and `checkdocs = :exports` is enabled, so a new
export without a docstring entry in `docs/src/index.md` will fail the build.

## Opening an issue

For a bug, please include:

- the MachineLearningForecast version and `versioninfo()`,
- the base model you used (any MLJ regressor) and its package version,
- a minimal reproducing script — a short synthetic series is ideal,
- what you expected and what happened.

## Pull requests

1. Open an issue first for anything larger than a fix, so the design can be
   discussed before you spend time on it.
2. Branch from `main`.
3. Add tests. A PR that changes behaviour without a test that would have caught
   the old behaviour will be asked for one.
4. Add or update docstrings; every exported name needs one **with an example**.
5. Run `Pkg.test()` and the docs build before pushing.

CI runs the test suite on Julia 1.10 (oldest supported) and the current stable
release, plus `Aqua.jl` quality checks (ambiguities, unbound type parameters,
stale dependencies, and the like).

## Design principles

These are load-bearing; please follow them:

- **Multiple dispatch over configuration flags.** Strategies, features and
  tuning strategies are *types*. Behaviour is selected by dispatch, never by
  `if strategy == "recursive"`.
- **Tables.jl for all tabular I/O.** Any Tables.jl-compatible input is accepted
  and normalised to a columntable (a `NamedTuple` of vectors); every tabular
  return value is a columntable too. The package deliberately has **no
  DataFrames.jl dependency** — do not add one.
- **Model genericity via MLJ.** Never hard-code a learner. Anything
  implementing the MLJ model interface must work as a base model.
- **Immutable specs, explicit fitted state.** `Forecaster` is immutable; `fit`
  returns a `FittedForecaster` and never mutates user data or model prototypes.
- **No leakage.** Every `TargetFeature` may use only information strictly
  before the current row's target. This is what the rolling/diff features'
  `lag ≥ 1` requirement enforces, and it is covered by tests.
- **Actionable errors.** Every user-facing validation throws an `ArgumentError`
  stating what was wrong, the offending value, and the fix. See the error
  message tests for the expected quality bar.

## Adding a new feature type

A feature is a subtype of `TargetFeature`, `TimeFeature` or `ExogenousFeature`,
and must implement four methods — this is the whole contract:

```julia
outputnames(f)                          # Vector{Symbol}: the columns it produces
minhistory(f)                           # Int: rows needed before it is defined
materialize!(out, f, y, t, tbl)         # batch/training path, vectorised
featurevalues(f, y_hist, t_next, exog)  # single-row path, used when forecasting
```

The two paths **must compute the same function**. A mismatch between them is a
silent accuracy bug: training sees one thing, forecasting another. Add a test
asserting the single-row value equals the materialised value at the same row —
`test/test_validation.jl` has examples.

If your feature is a `TargetFeature`, it is anchored to the *feature* row; time
and exogenous features are anchored to the *target* row. `Direct` fitting relies
on this split via `targetcolumnmask`, so getting the category right matters.

## Adding a tuning strategy

`TuningStrategy`, `ask` and `tell!` are public, documented API precisely so that
strategies can live outside this package. Subtype `TuningStrategy` and implement
`ask` (return a `NamedTuple` of `Forecaster` field overrides, or `nothing` to
stop) and `tell!` (receive the candidate and its `Float64` score, or `missing`
if it failed). `tune` will drive the loop. See the `TuningStrategy` docstring
for the exact contract, and `test/test_tune.jl` for a third-party strategy
implemented using only exported names.

## Style

Match the surrounding code: 4-space indent, ~92 column limit, lowercase
function names, `snake_case` for internals and `CamelCase` for types. Comments
should explain *why*, not restate the code.
