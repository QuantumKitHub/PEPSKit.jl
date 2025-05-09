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
using MPSKitModels: MPSKitModels # used for docstrings

# bibliography
bibpath = joinpath(@__DIR__, "src", "assets", "pepskit.bib")
bib = CitationBibliography(bibpath; style=:authoryear)

# interlinks
# Zygote didn't update to documenter v1 yet...
links = InterLinks(
    "TensorKit" => "https://jutho.github.io/TensorKit.jl/stable/",
    "KrylovKit" => "https://jutho.github.io/KrylovKit.jl/stable/",
    "MPSKit" => "https://quantumkithub.github.io/MPSKit.jl/stable/",
    "MPSKitModels" => "https://quantumkithub.github.io/MPSKitModels.jl/dev/",
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

# examples pages
examples_optimization = joinpath.(
    ["heisenberg", "bose_hubbard", "xxz", "fermi_hubbard"], Ref("index.md")
)
examples_time_evolution = joinpath.(["heisenberg_su", "hubbard_su"], Ref("index.md"))
examples_partition_functions = joinpath.(
    ["2d_ising_partition_function", "3d_ising_partition_function"], Ref("index.md")
)
examples_boundary_mps = joinpath.(["boundary_mps"], Ref("index.md"))

makedocs(;
    modules=[PEPSKit, MPSKitModels],
    sitename="PEPSKit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true", mathengine, size_threshold=1024000
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => ["man/models.md", "man/multithreading.md", "man/precompilation.md"],
        "Examples" => [
            "examples/index.md",
            "Optimization" => joinpath.(Ref("examples"), examples_optimization),
            "Time Evolution" => joinpath.(Ref("examples"), examples_time_evolution),
            "Partition Functions" =>
                joinpath.(Ref("examples"), examples_partition_functions),
            "Boundary MPS" => joinpath.(Ref("examples"), examples_boundary_mps),
        ],
        "Library" => "lib/lib.md",
        "References" => "references.md",
    ],
    checkdocs=:none,
    # checkdocs_ignored_modules=[MPSKitModels], # doesn't seem to work...
    plugins=[bib, links],
)

deploydocs(; repo="github.com/QuantumKitHub/PEPSKit.jl.git", push_preview=true)
