using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "CTMRG"
        @time @safetestset "Gauge Fixing" begin
            include("ctmrg/gaugefix.jl")
        end
        @time @safetestset "Gradients" begin
            include("ctmrg/gradients.jl")
        end
        @time @safetestset "SVD wrapper" begin
            include("ctmrg/svd_wrapper.jl")
        end
    end
    if GROUP == "All" || GROUP == "MPS"
        @time @safetestset "VUMPS" begin
            include("boundarymps/vumps.jl")
        end
    end
    if GROUP == "All" || GROUP == "EXAMPLES"
        @time @safetestset "Heisenberg model" begin
            include("heisenberg.jl")
        end
        @time @safetestset "P-wave superconductor" begin
            include("pwave.jl")
        end
    end
end
