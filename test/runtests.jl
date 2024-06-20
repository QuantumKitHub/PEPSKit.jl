using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "CTMRG"
        @time @safetestset "Gauge Fixing" begin
            include("ctmrg/gaugefix.jl")
        end
        @time @safetestset "Gradients" begin
            include("ctmrg/gradients.jl")
            #include("ctmrg/gradparts.jl")
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
            include("pwave.jl")
        end
    end
end
