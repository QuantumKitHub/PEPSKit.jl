module CTMRGBenchmarks

export CTMRGSpec

using BenchmarkTools
using TOML
using TensorKit
using PEPSKit
using ..BenchUtils
import ..BenchUtils: tomlify, untomlify

const SUITE = BenchmarkGroup()

include("ctmrg_iteration_benchmarks.jl")

const allparams = Dict(
    "default" => TOML.parsefile(joinpath(@__DIR__, "default.toml")),
    "su3_hubbard" => TOML.parsefile(joinpath(@__DIR__, "su3_hubbard.toml")),
)

const CTMRG_ALGS = (
    "SequentialCTMRG_HalfInfinite" => SequentialCTMRG(; projector_alg = :HalfInfiniteProjector),
    "SimultaneousCTMRG_HalfInfinite" => SimultaneousCTMRG(; projector_alg = :HalfInfiniteProjector),
)

T = ComplexF64

for (scenario, params) in allparams
    g_scen = addgroup!(SUITE, scenario)
    for (alg_name, alg) in CTMRG_ALGS
        g_alg = addgroup!(g_scen, alg_name)
        for (sym_name, specs) in params
            g_sym = addgroup!(g_alg, sym_name)
            for spec_dict in specs
                spec = untomlify(CTMRGSpec, spec_dict)
                g_sym[benchname(spec)] = ctmrg_iteration_benchmark(spec, alg; T)
            end
        end
    end
end

end # module
