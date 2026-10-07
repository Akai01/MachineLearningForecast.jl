# Unregistered: develop the package so docs resolve.
using Pkg
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))

using Documenter
using MachineLearningForecast

const REPO_URL = "https://github.com/Akai01/MachineLearningForecast.jl"

# Source links need a git commit; skip them without one.
const HAS_COMMIT = try
    success(`git -C $(dirname(@__DIR__)) rev-parse --verify --quiet HEAD`)
catch
    false
end

const format = HAS_COMMIT ?
    Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                    canonical = "https://Akai01.github.io/MachineLearningForecast.jl",
                    repolink = REPO_URL) :
    Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                    edit_link = nothing, repolink = nothing)

const repo_kwargs = HAS_COMMIT ?
    (repo = "$(REPO_URL)/blob/{commit}{path}#{line}",) :
    (remotes = nothing,)

makedocs(;
    sitename = "MachineLearningForecast.jl",
    modules = [MachineLearningForecast],
    authors = "Resul Akay and contributors",
    format,
    pages = ["Home" => "index.md", "Using different models" => "models.md"],
    checkdocs = :exports,
    repo_kwargs...,
)

# Deploys only in CI with a deploy key or token.
deploydocs(
    repo = "github.com/Akai01/MachineLearningForecast.jl.git",
    devbranch = "main",
    push_preview = true,
)
