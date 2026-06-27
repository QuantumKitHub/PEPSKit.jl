module PEPSKitBenchmarks

using BenchmarkTools
using PEPSKit
using TOML

include("utils/BenchUtils.jl")

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60.0
BenchmarkTools.DEFAULT_PARAMETERS.samples = 5
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1

const PARAMS_PATH = joinpath(@__DIR__, "etc", "params.json")
const SUITE = BenchmarkGroup()
const MODULES = Dict{String, Symbol}(
    "ctmrg" => :CTMRGBenchmarks,
)

include("ctmrg/CTMRGBenchmarks.jl")

load!(id::AbstractString; kwargs...) = load!(SUITE, id; kwargs...)

function load!(group::BenchmarkGroup, id::AbstractString; tune::Bool = false)
    modsym = MODULES[id]
    mod = Core.eval(@__MODULE__, modsym)
    modsuite = @invokelatest getglobal(mod, :SUITE)
    group[id] = modsuite
    if tune
        results = BenchmarkTools.load(PARAMS_PATH)[1]
        haskey(results, id) && loadparams!(modsuite, results[id], :evals)
    end
    return group
end

loadall!(; kwargs...) = loadall!(SUITE; kwargs...)

function loadall!(group::BenchmarkGroup; verbose::Bool = true, tune::Bool = false)
    for id in keys(MODULES)
        if verbose
            print("loading group $(repr(id))... ")
            time = @elapsed load!(group, id; tune = false)
            println("done (took $time seconds)")
        else
            load!(group, id; tune = false)
        end
    end
    if tune
        results = BenchmarkTools.load(PARAMS_PATH)[1]
        for (id, suite) in group
            haskey(results, id) && loadparams!(suite, results[id], :evals)
        end
    end
    return group
end

end # module
