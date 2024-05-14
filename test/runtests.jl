using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "CTMRG"
        @time @safetestset "Gauge Fixing" begin include("ctmrg/gaugefix.jl") end
        @time @safetestset "Gradients" begin include("ctmrg/gradients.jl") end
    end
    if GROUP == "All" || GROUP == "EXAMPLES"
        @time @safetestset "Heisenberg model" begin
            include("heisenberg.jl")
        end
    end
end

