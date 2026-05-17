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

const RESERVED_KEYS = ("unitcell", "algorithms")

T = ComplexF64

for (scenario, params) in allparams
    unitcell = (Int(params["unitcell"][1]), Int(params["unitcell"][2]))
    algs = [untomlify(PEPSKit.CTMRGAlgorithm, a) for a in params["algorithms"]]
    g_scen = addgroup!(SUITE, scenario)
    for alg in algs
        g_alg = addgroup!(g_scen, algname(alg))
        for (sym_name, specs) in params
            sym_name in RESERVED_KEYS && continue
            g_sym = addgroup!(g_alg, sym_name)
            for spec_dict in specs
                spec = untomlify(CTMRGSpec, spec_dict; unitcell)
                g_sym[benchname(spec)] = ctmrg_iteration_benchmark(spec, alg; T)
            end
        end
    end
end

end # module
