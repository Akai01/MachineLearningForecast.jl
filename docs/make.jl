# Make the docs environment resolvable from any clean checkout: MachineLearningForecast is
# not registered yet, so `docs/Project.toml` cannot resolve it by UUID alone.
using Pkg
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))

using Documenter
using MachineLearningForecast

const REPO_URL = "https://github.com/Akai01/MachineLearningForecast.jl"

# Documenter can only build "Edit on GitHub" / source links when git can resolve
# a commit, which requires the repository to have at least one. Fall back
# gracefully so that `julia --project=docs docs/make.jl` also works in a fresh
# checkout that has not been committed yet.
const HAS_COMMIT = try
    success(`git -C $(dirname(@__DIR__)) rev-parse --verify --quiet HEAD`)
catch
    false
end

format = HAS_COMMIT ?
    Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                    canonical = "https://Akai01.github.io/MachineLearningForecast.jl",
                    repolink = REPO_URL) :
    Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                    edit_link = nothing, repolink = nothing)

repo_kwargs = HAS_COMMIT ?
    (repo = "$(REPO_URL)/blob/{commit}{path}#{line}",) :
    (remotes = nothing,)

makedocs(;
    sitename = "MachineLearningForecast.jl",
    modules = [MachineLearningForecast],
    authors = "Resul Akay and contributors",
    format,
    pages = ["Home" => "index.md"],
    checkdocs = :exports,
    repo_kwargs...,
)

# Only does anything when running in CI with the deploy key/token configured.
deploydocs(
    repo = "github.com/Akai01/MachineLearningForecast.jl.git",
    devbranch = "main",
    push_preview = true,
)
