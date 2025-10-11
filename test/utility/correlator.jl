using Test
using Random
using TensorKit
using PEPSKit

Nr, Nc = 2, 2
Vphy = Vect[FermionParity](0 => 1, 1 => 1)
Pspaces = fill(Vphy, (Nr, Nc))
op = randn(ComplexF64, Vphy ⊗ Vphy → Vphy ⊗ Vphy)

V = Vect[FermionParity](0 => 1, 1 => 2)
Venv = Vect[FermionParity](0 => 2, 1 => 2)
Nspaces = [V' V; V V']
Espaces = [V V'; V' V]

site0 = CartesianIndex(1, 1)
maxsep = 6
site1xs = collect(site0 + CartesianIndex(0, i) for i in 2:2:maxsep)
site1ys = collect(site0 + CartesianIndex(i, 0) for i in 2:2:maxsep)

@testset "Correlator in InfinitePEPS" begin
    Random.seed!(100)
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

@testset "Correlator in InfinitePEPO (1-layer)" begin
    Random.seed!(100)
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
