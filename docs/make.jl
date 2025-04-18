# if docs is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
    Pkg.resolve()
    Pkg.instantiate()
end

using Documenter
using DocumenterCitations
using DocumenterInterLinks
using PEPSKit

# examples
example_dir = joinpath(@__DIR__, "src", "examples")

# bibliography
bibpath = joinpath(@__DIR__, "src", "assets", "pepskit.bib")
bib = CitationBibliography(bibpath; style=:authoryear)

# interlinks
# Zygote and MPSKitModels didn't update to documenter v1 yet...
links = InterLinks(
    "TensorKit" => "https://jutho.github.io/TensorKit.jl/stable/",
    "KrylovKit" => "https://jutho.github.io/KrylovKit.jl/stable/",
    "MPSKit" => "https://quantumkithub.github.io/MPSKit.jl/stable/",
    # "MPSKitModels" => "https://quantumkithub.github.io/MPSKitModels.jl/",
    # "Zygote" => "https://fluxml.ai/Zygote.jl/stable/",
    "ChainRulesCore" => "https://juliadiff.org/ChainRulesCore.jl/stable/",
)

# explicitly set math engine
mathengine = MathJax3(
    Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"],
        ),
    ),
)

# built examples target directories
example_pages = map(readdir(joinpath(@__DIR__, "src", "examples"))) do dir
    return joinpath("examples", dir, "index.md")
end

makedocs(;
    modules=[PEPSKit],
    sitename="PEPSKit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true", mathengine, size_threshold=1024000
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "man/states.md",
            "man/environments.md",
            "man/operators.md",
            "man/peps_optimization.md",
            "man/symmetries.md",
            "man/multi_threading.md",
            "man/precompilation.md",
        ],
        "Examples" => example_pages,
        "Library" => "lib/lib.md",
        "References" => "references.md",
    ],
    checkdocs=:exports,
    plugins=[bib, links],
)

deploydocs(; repo="github.com/QuantumKitHub/PEPSKit.jl.git", push_preview=true)
