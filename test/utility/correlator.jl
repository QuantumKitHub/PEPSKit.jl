using Test
using Random
using TensorKit
using PEPSKit

const syms = (Z2Irrep, FermionParity)

function get_spaces(sym::Type{<:Sector})
    @assert sym in syms
    Nr, Nc = 2, 2
    Vphy = Vect[sym](0 => 1, 1 => 1)
    V = Vect[sym](0 => 1, 1 => 2)
    Venv = Vect[sym](0 => 2, 1 => 2)
    Nspaces = [V' V; V V']
    Espaces = [V V'; V' V]
    return Vphy, Venv, Nspaces, Espaces
end

site0 = CartesianIndex(1, 1)
maxsep = 6
site1xs = collect(site0 + CartesianIndex(0, i) for i in 2:2:maxsep)
site1ys = collect(site0 + CartesianIndex(i, 0) for i in 2:2:maxsep)

@testset "Correlator in InfinitePEPS ($(sym))" for sym in syms
    Random.seed!(100)
    Vphy, Venv, Nspaces, Espaces = get_spaces(sym)
    for Vp in [Vphy', Vphy]
        op = randn(ComplexF64, Vp ⊗ Vp → Vp ⊗ Vp)
        Pspaces = fill(Vp, size(Nspaces))
        peps = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
        env = CTMRGEnv(randn, ComplexF64, peps, Venv)
        for site1s in (site1xs, site1ys)
            vals1 = correlator(peps, op, site0, site1s, env)
            vals2 = map(site1s) do site1
                O = LocalOperator(Pspaces, (site0, site1) => op)
                return expectation_value(peps, O, env)
            end
            @info vals1
            @info vals2
            @test vals1 ≈ vals2
        end
    end
end

@testset "Correlator in 1-layer InfinitePEPO ($(sym))" for sym in syms
    Random.seed!(100)
    Vphy, Venv, Nspaces, Espaces = get_spaces(sym)
    for Vp in [Vphy', Vphy]
        op = randn(ComplexF64, Vp ⊗ Vp → Vp ⊗ Vp)
        Pspaces = fill(Vp, size(Nspaces))
        pepo = InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces)
        pf = InfinitePartitionFunction(pepo)
        env = CTMRGEnv(randn, ComplexF64, pf, Venv)
        for site1s in (site1xs, site1ys)
            vals1 = correlator(pepo, op, site0, site1s, env)
            vals2 = map(site1s) do site1
                O = LocalOperator(Pspaces, (site0, site1) => op)
                return expectation_value(pepo, O, env)
            end
            @info vals1
            @info vals2
            @test vals1 ≈ vals2
        end
    end
end
