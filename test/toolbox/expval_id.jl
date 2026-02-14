using Test
using TensorKit
using PEPSKit
using Random

const syms = (FermionParity, U1Irrep)
Random.seed!(0)

function get_spaces(sym::Type{<:Sector})
    if sym == U1Irrep
        Vp = Vect[U1Irrep](-1 => 1, 0 => 1, 1 => 2)
        Vv = Vect[U1Irrep](-1 => 1, 0 => 3, 1 => 2)
        Ve = Vect[U1Irrep](-1 => 1, 0 => 3, 1 => 2)
        return Vp, Vv, Ve
    elseif sym == FermionParity
        Vp = Vect[FermionParity](0 => 1, 1 => 2)
        Vv = Vect[FermionParity](0 => 3, 1 => 2)
        Ve = Vect[FermionParity](0 => 3, 1 => 2)
        return Vp, Vv, Ve
    else
        error("Got a sector not intended to be tested.")
    end
end

@testset "⟨ψ|1|ψ⟩ (InfinitePEPS, $(sym))" for sym in syms
    Vp, Vv, Ve = get_spaces(sym)
    Pspaces = [Vp Vp'; Vp Vp']
    Vspaces = fill(Vv, (2, 2))
    ψ = InfinitePEPS(Pspaces, Vspaces)
    env = CTMRGEnv(ψ, Ve)
    for site1 in Tuple.(CartesianIndices((2, 2)))
        # 1-site
        id1 = TensorKit.id(physicalspace(ψ, site1...))
        O1 = LocalOperator(Pspaces, (site1,) => id1)
        val1 = expectation_value(ψ, O1, env)
        @info "$((site1,)): $(val1)"
        @test val1 ≈ 1
        # 2-site
        for d in [(1, 0), (0, 1), (1, 1), (-1, 1)]
            site2 = site1 .+ d
            id2 = TensorKit.id(physicalspace(ψ, site2...))
            O2 = LocalOperator(Pspaces, (site1, site2) => id1 ⊗ id2)
            val2 = expectation_value(ψ, O2, env)
            @info "$((site1, site2)): $(val2)"
            @test val2 ≈ 1
        end
    end
end

@testset "tr(ρ1) (one-layer InfinitePEPO, $(sym))" for sym in syms
    Vp, Vv, Ve = get_spaces(sym)
    Pspaces = [Vp Vp'; Vp Vp']
    Vspaces = fill(Vv, (2, 2))
    ρ = InfinitePEPO(Pspaces, Vspaces)
    env = CTMRGEnv(InfinitePartitionFunction(ρ), Ve)
    for site1 in Tuple.(CartesianIndices((2, 2)))
        # 1-site
        id1 = TensorKit.id(physicalspace(ρ, site1...))
        O1 = LocalOperator(Pspaces, (site1,) => id1)
        val1 = expectation_value(ρ, O1, env)
        @info "$((site1,)): $(val1)"
        @test val1 ≈ 1
        # 2-site
        for d in [(1, 0), (0, 1), (1, 1), (-1, 1)]
            site2 = site1 .+ d
            id2 = TensorKit.id(physicalspace(ρ, site2...))
            O2 = LocalOperator(Pspaces, (site1, site2) => id1 ⊗ id2)
            val2 = expectation_value(ρ, O2, env)
            @info "$((site1, site2)): $(val2)"
            @test val2 ≈ 1
        end
    end
end

@testset "⟨ρ|1|ρ⟩ (InfinitePEPS with ancilla, $(sym))" for sym in syms
    Vp, Vv, Ve = get_spaces(sym)
    Pspaces = [Vp Vp'; Vp Vp']
    Vspaces = fill(Vv, (2, 2))
    ρ = InfinitePEPO(Pspaces, Vspaces)
    env = CTMRGEnv(InfinitePEPS(ρ), Ve)
    for site1 in Tuple.(CartesianIndices((2, 2)))
        # 1-site
        id1 = TensorKit.id(physicalspace(ρ, site1...))
        O1 = LocalOperator(Pspaces, (site1,) => id1)
        val1 = expectation_value(ρ, O1, ρ, env)
        @info "$((site1,)): $(val1)"
        @test val1 ≈ 1
        # 2-site
        for d in [(1, 0), (0, 1), (1, 1), (-1, 1)]
            site2 = site1 .+ d
            id2 = TensorKit.id(physicalspace(ρ, site2...))
            O2 = LocalOperator(Pspaces, (site1, site2) => id1 ⊗ id2)
            val2 = expectation_value(ρ, O2, ρ, env)
            @info "$((site1, site2)): $(val2)"
            @test val2 ≈ 1
        end
    end
end
