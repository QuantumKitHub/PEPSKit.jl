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
        @time @safetestset "Unit cell" begin
            include("ctmrg/unitcell.jl")
        end
        @time @safetestset ":fixed CTMRG iteration scheme" begin
            include("ctmrg/fixed_iterscheme.jl")
        end
        @time @safetestset "Unit cells" begin
            include("ctmrg/unitcell.jl")
        end
        @time @safetestset "Flavors" begin
            include("ctmrg/flavors.jl")
        end
        @time @safetestset "Partition function" begin
            include("ctmrg/partition_function.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "BOUNDARYMPS"
        @time @safetestset "VUMPS" begin
            include("boundarymps/vumps.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "BONDENV"
        @time @safetestset "Positive-definite bond environment" begin
            include("bondenv/positive.jl")
        end
        @time @safetestset "Iterative optimization after truncation" begin
            include("bondenv/optimize.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "UTILITY"
        @time @safetestset "SVD wrapper" begin
            include("utility/svd_wrapper.jl")
        end
        @time @safetestset "Symmetrization" begin
            include("utility/symmetrization.jl")
        end
        @time @safetestset "Differentiable tmap" begin
            include("utility/diff_maps.jl")
        end
    end
    if GROUP == "ALL" || GROUP == "EXAMPLES"
        @time @safetestset "Transverse Field Ising model" begin
            include("tf_ising.jl")
        end
        @time @safetestset "Heisenberg model" begin
            include("heisenberg.jl")
        end
        @time @safetestset "J1-J2 model" begin
            include("j1j2_model.jl")
        end
        @time @safetestset "P-wave superconductor" begin
            include("pwave.jl")
        end
    end
end
