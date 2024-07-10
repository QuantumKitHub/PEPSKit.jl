using SafeTestsets

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end

@time begin
    if GROUP == "ALL" || GROUP == "CTMRG"
        @time @safetestset "Gauge Fixing" begin
            include("ctmrg/gaugefix.jl")
        end
        @time @safetestset "Gradients" begin
            include("ctmrg/gradients.jl")
        end
        @time @safetestset "SVD wrapper" begin
            include("ctmrg/svd_wrapper.jl")
        end
        @time @safetestset "SVD wrapper" begin
            include("ctmrg/unitcell.jl")
        end
        @time @safetestset "SVD wrapper" begin
            include("ctmrg/leftmoves_allsides.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "MPS"
        @time @safetestset "VUMPS" begin
            include("boundarymps/vumps.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "EXAMPLES"
        @time @safetestset "Transverse Field Ising model" begin
            include("tf_ising.jl")
        end
        @time @safetestset "Heisenberg model" begin
            include("heisenberg.jl")
        end
        @time @safetestset "P-wave superconductor" begin
            include("pwave.jl")
        end
    end
end
