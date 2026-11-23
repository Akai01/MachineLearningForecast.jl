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
- **No leakage.** Every `AtOrigin` feature (every `TargetFeature` by default)
  may use only information strictly before the current row's target. This is
  what the rolling/diff features' `lag ≥ 1` requirement enforces, and it is
  covered by tests.
- **Actionable errors.** Every user-facing validation throws an `ArgumentError`
  stating what was wrong, the offending value, and the fix. See the error
  message tests for the expected quality bar.

## Adding a new feature type

A feature is a subtype of `TargetFeature`, `TimeFeature` or `ExogenousFeature`, all
exported (or of `AbstractFeature` directly, then it also defines `anchor`). To extend
the package, import the generic functions and implement three methods (plus `anchor`
for a direct `AbstractFeature` subtype):

```julia
import MachineLearningForecast: outputnames, minhistory, featurevalues!, inputcolumns,
    anchor

outputnames(f)                  # Vector{Symbol}: the columns it produces
minhistory(f)                   # Int: history rows needed before it is defined
featurevalues!(out, f, ctx)     # write the values for the row ctx describes
anchor(f)                       # AtOrigin() or AtTarget() (only for direct AbstractFeature subtypes)
```

`AtOrigin` and `AtTarget` need the `MachineLearningForecast.` prefix or an import.

`ctx` is a `MachineLearningForecast.FeatureContext`:

- `ctx.y`: the target history strictly before the row's target (a `Vector` when
  forecasting, a view in training: read it, do not keep it). It is empty at the first
  row when `minhistory(f) = 0`. Under `Direct` the history is fixed at the forecast
  origin, so `ctx.y` is the same for every horizon.
- `ctx.t`: the row's timestamp; `ctx.step`: its 0-based step index (shared by all
  series of a panel).
- `ctx.exog`: a `NamedTuple` of the user's columns at the row. The values are the raw
  user values (not converted to `Float64`: convert them yourself and reject what you
  cannot use). At forecast time it may hold more columns than `inputcolumns(f)`, so
  index it by name. When `inputcolumns(f)` is empty it may be `nothing` or a
  `NamedTuple` (forecasting with another exogenous feature): do not read it then.

A feature's `anchor` is `AtOrigin` or `AtTarget`; see below.

`out` is preallocated with `length(outputnames(f))` entries: write every one of them as
a `Float64`. The return value is ignored. If the feature reads user columns, also
define `inputcolumns(f)`, the `Vector{Symbol}` of their names (the default is
`Symbol[]`; an `ExogenousFeature` without it throws an `ArgumentError`). `forecast` and
`backtest` use it to pick the columns from the `future` keyword.

Training calls `featurevalues!` for every row, so by default (unless the feature
overrides `materialize!`) batch and single-row values cannot diverge. Define
`MachineLearningForecast.materialize!(out, f::MyFeature,
s::MachineLearningForecast.TrainingSeries)` only for speed or to validate a column
before any row is read (`Calendar`, `Fourier` and `Exogenous` do). Add a test that
`featurevalues!` at row i equals the training column at row i, and a leakage test
(changing `y` from row t on leaves the values at rows up to t unchanged):
`test/test_features.jl` has examples.

A feature's `anchor` says which row its columns describe. A `TargetFeature` defaults to
`AtOrigin` (the feature row: target history); a `TimeFeature` or `ExogenousFeature`
defaults to `AtTarget` (the target row). `Direct` fitting shifts the two groups
independently, so getting the anchor right matters. An `AtOrigin` feature may read only
`ctx.y`: its training column comes from the feature row and its forecast value from the
origin, while `ctx.t`, `ctx.step` and `ctx.exog` describe the context's row (the feature
row in training, the target row when forecasting). An `AtTarget` feature should not read
`ctx.y`: under `Direct` it is trained on history up to the target row but served the
history at the forecast origin. A feature that reads `ctx.t`, `ctx.step` or `ctx.exog`
and not `ctx.y` is `AtTarget`: subtype `TimeFeature` or `ExogenousFeature`, or define
`anchor(f) = AtTarget()`; one that reads both is correct only under `Recursive`.

## Adding a target transform

A transform is a subtype of `TargetTransform` and implements two methods, which are
public but not exported, so import them:

```julia
import MachineLearningForecast: fit_transform, untransform

fit_transform(t, y)        # -> (z, state): z may be shorter, aligned to y's end
untransform(t, ẑ, state)   # -> the forecasts of y for the forecasts ẑ of z
```

`fit_transform` sees one series' training history only, and its `state` must
hold everything `untransform` needs. Add a test that `untransform` recovers a
known future exactly from its forward-mapped values, as
`test/test_transforms.jl` does for the built-ins.

## Adding a tuning strategy

`TuningStrategy`, `ask!` and `tell!` are public, documented API precisely so that
strategies can live outside this package. `ask!` and `tell!` are not exported, so
import them (`import MachineLearningForecast: ask!, tell!`). Subtype `TuningStrategy`
and implement `ask!` (return a `NamedTuple` of `Forecaster` field overrides, or
`nothing` to stop) and `tell!` (receive the candidate and its `Float64` score, or `missing`
if it failed). `tune` will drive the loop. See the `TuningStrategy` docstring
for the exact contract, and `test/test_tune.jl` for a third-party strategy
implemented using only the public API.

## Style

Match the surrounding code: 4-space indent, ~92 column limit, lowercase
function names, `snake_case` for internals and `CamelCase` for types. Comments
should explain *why*, not restate the code.
