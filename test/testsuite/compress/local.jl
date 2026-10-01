using Test
using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using PEPSKit: virtual_projector
using Adapt

"""
Cost function of LocalTruncation.
For test convenience, open virtual indices are made trivial and removed.
"""
function localcompress_cost(A1, A2, B1, B2, P1, P2)
    @tensor net1[pa1 pb1; pa2′ pb2′] :=
        A1[pa1 pa; D1] * A2[pa pa2′; D2] * B1[pb1 pb; D1] * B2[pb pb2′; D2]
    @tensor net2[pa1 pb1; pa2′ pb2′] := P1[Da1 Da2; D] * P2[D; Db1 Db2] *
        A1[pa1 pa; Da1] * A2[pa pa2′; Da2] * B1[pb1 pb; Db1] * B2[pb pb2′; Db2]
    return norm(net1 - net2)
end

function compress_fermionic_twists(AT)
    return @testset "Fermionic twists ($AT)" begin
        Vphy = Vect[FermionParity](0 => 2, 1 => 2)
        Vvir = Vect[FermionParity](0 => 2, 1 => 2)
        for _ in 1:4 # multiple trials without setting seed
            Aspace = (Vphy ⊗ Vphy' ← Vvir ⊗ Vvir ⊗ Vvir' ⊗ Vvir')
            A1 = adapt(AT, randn(ComplexF64, Aspace))
            A2 = adapt(AT, randn(ComplexF64, Aspace))
            for MM in [PEPSKit._get_MMdag(A1, A2), PEPSKit._get_MdagM(A1, A2)]
                @test isposdef(MM)
            end
        end
    end
end

function compress_cost_function(AT)
    return @testset "Cost function of LocalTruncation ($AT)" begin
        Random.seed!(0)
        Vaux, Vphy, V = ℂ^1, ℂ^10, ℂ^4
        A1 = normalize(adapt(AT, randn(Vphy ⊗ Vphy' ← Vaux ⊗ V ⊗ Vaux' ⊗ Vaux')), Inf)
        A2 = normalize(adapt(AT, randn(Vphy ⊗ Vphy' ← Vaux ⊗ V ⊗ Vaux' ⊗ Vaux')), Inf)
        B1 = normalize(adapt(AT, randn(Vphy ⊗ Vphy' ← Vaux ⊗ Vaux ⊗ Vaux' ⊗ V')), Inf)
        B2 = normalize(adapt(AT, randn(Vphy ⊗ Vphy' ← Vaux ⊗ Vaux ⊗ Vaux' ⊗ V')), Inf)

        P1, P2, info = virtual_projector(A1, A2, B1, B2; trunc = notrunc())
        @test P1 * P2 ≈ adapt(AT, TensorKit.id(domain(P2)))

        P1, P2, info = virtual_projector(A1, A2, B1, B2; trunc = truncrank(8))
        A1 = removeunit(removeunit(removeunit(A1, 6), 5), 3)
        A2 = removeunit(removeunit(removeunit(A2, 6), 5), 3)
        B1 = removeunit(removeunit(removeunit(B1, 5), 4), 3)
        B2 = removeunit(removeunit(removeunit(B2, 5), 4), 3)
        @info "Truncation error = $(info.ϵ)."
        @test info.ϵ ≈ localcompress_cost(A1, A2, B1, B2, P1, P2)
    end
end

function compress_virtual_space_matching(AT)
    return @testset "Virtual space matching ($AT)" begin
        Vps = ComplexSpace.([2 2; 2 2])
        Vns = ComplexSpace.([2 4; 5 3])
        Ves = ComplexSpace.([3 5; 4 2])
        ρ = adapt(AT, InfinitePEPO(randn, ComplexF64, Vps, Vns, Ves))
        alg = LocalTruncation(truncrank(2))
        ρ2, = compress((ρ, ρ), alg)
        @test ρ2 isa InfinitePEPO
    end
end
