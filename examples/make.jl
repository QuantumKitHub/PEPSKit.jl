# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

using PEPSKit
using Literate
using TOML, SHA

# ---------------------------------------------------------------------------------------- #
# Caching
# ---------------------------------------------------------------------------------------- #

const CACHEFILE = joinpath(@__DIR__, "Cache.toml")

getcache() = isfile(CACHEFILE) ? TOML.parsefile(CACHEFILE) : Dict{String,Any}()

function iscached(name)
    cache = getcache()
    return haskey(cache, name) && cache[name] == checksum(name)
end

function setcached(name)
    cache = getcache()
    cache[name] = checksum(name)
    return open(f -> TOML.print(f, cache), CACHEFILE, "w")
end

# generate checksum based on path relative to ~/.../PEPSKit.jl
# such that different users do not have to rerun already cached examples
function checksum(name)
    project_path = joinpath(@__DIR__, name, "main.jl")
    @assert isfile(project_path)
    return open(project_path, "r") do io
         bytes2hex(sha256(io))
    end
end

# ---------------------------------------------------------------------------------------- #
# Building
# ---------------------------------------------------------------------------------------- #

attach_notebook_badge(name) = str -> attach_notebook_badge(name, str)
function attach_notebook_badge(name, str)
    mybinder_badge_url = "https://mybinder.org/badge_logo.svg"
    nbviewer_badge_url = "https://img.shields.io/badge/show-nbviewer-579ACA.svg"
    download_badge_url = "https://img.shields.io/badge/download-project-orange"
    mybinder = "[![]($mybinder_badge_url)](@__BINDER_ROOT_URL__/examples/$name/main.ipynb)"
    nbviewer = "[![]($nbviewer_badge_url)](@__NBVIEWER_ROOT_URL__/examples/$name/main.ipynb)"
    download = "[![]($download_badge_url)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/$name)"

    markdown_only(x) = "#md # " * x
    return join(map(markdown_only, (mybinder, nbviewer, download)), "\n") * "\n\n" * str
end

function build_example(name)
    source_dir = joinpath(@__DIR__, "..", "examples", name)
    source_file = joinpath(source_dir, "main.jl")
    target_dir = joinpath(@__DIR__, "..", "docs", "src", "examples", name)

    if !iscached(name)
        Literate.markdown(
            source_file,
            target_dir;
            execute=true,
            name="index",
            preprocess=attach_notebook_badge(name),
            mdstrings=true,
            nbviewer_root_url="https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev",
            binder_root_url="https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev",
            credits=false,
            repo_root_url="https://github.com/QuantumKitHub/PEPSKit.jl",
        )
        Literate.notebook(
            source_file,
            target_dir;
            execute=false,
            name="main",
            preprocess=str -> replace(str, r"(?<!`)``(?!`)" => "\$"),
            mdstrings=true,
            credits=false,
        )

        foreach(filter(!=("main.jl"), readdir(source_dir))) do f
            return cp(joinpath(source_dir, f), joinpath(target_dir, f); force=true)
        end
        setcached(name)
    end
end

function build()
    return cd(@__DIR__) do
        examples = filter(isdir, readdir(@__DIR__)) # filter out directories to ignore Cache.toml, etc.
        return map(ex -> build_example(ex), examples)
    end
end

# ---------------------------------------------------------------------------------------- #
# Scripts
# ---------------------------------------------------------------------------------------- #

build()
