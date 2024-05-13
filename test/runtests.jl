using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "CTMRG"
        @time @safetestset "Gauge Fixing" begin include("ctmrg/gaugefix.jl") end
    end
end

