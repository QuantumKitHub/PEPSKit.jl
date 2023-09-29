using Documenter
using PEPSKit

makedocs(;
    modules=[PEPSKit],
    sitename="PEPSKit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        mathengine=MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/physics"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "physics"],
                ),
            ),
        ),
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => "man/intro.md",
        "Examples" => "examples/index.md",
        "Library" => "lib/lib.md",
    ],
)

deploydocs(; repo="https://github.com/quantumghent/PEPSKit.jl.git")
