# Load benchmark code
include("PEPSKitBenchmarks/PEPSKitBenchmarks.jl")
const SUITE = PEPSKitBenchmarks.SUITE

# Populate benchmarks
# Detect if user supplied extra arguments to load only specific modules
# e.g. julia benchmarks.jl --modules=ctmrg
modules_pattern = r"(?:--modules=)([\w,]+)"
arg_id = findfirst(contains(modules_pattern), ARGS)
if isnothing(arg_id)
    PEPSKitBenchmarks.loadall!()
else
    modules = split(match(modules_pattern, ARGS[arg_id]).captures[1], ",")
    for m in modules
        PEPSKitBenchmarks.load!(m)
    end
end
