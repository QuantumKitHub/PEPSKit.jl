using Test
using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using PEPSKit: localapprox_projector

"""
Cost function of LocalApprox
```
        ↓ ╱      ↓ ╱            ↓ ╱                 ↓ ╱
    ----A2---←---B2---      ----A2-←-|╲       ╱|--←-B2---
      ╱ |      ╱ |            ╱ |    | ╲     ╱ |  ╱ |
        ↓        ↓       -      ↓    |P1├-←-┤P2|    ↓
        | ╱      | ╱            | ╱  | ╱     ╲ |    | ╱
    ----A1---←---B1---      ----A1-←-|╱       ╲|--←-B1---
      ╱ ↓      ╱ ↓            ╱ ↓                 ╱ ↓
```
For test convenience, open virtual indices are made trivial and removed.
"""
function localapprox_cost(A1, A2, B1, B2, P1, P2)
    @tensor net1[pa1 pb1; pa2′ pb2′] :=
        A1[pa1 pa; D1] * A2[pa pa2′; D2] * B1[pb1 pb; D1] * B2[pb pb2′; D2]
    @tensor net2[pa1 pb1; pa2′ pb2′] := P1[Da1 Da2; D] * P2[D; Db1 Db2] *
        A1[pa1 pa; Da1] * A2[pa pa2′; Da2] * B1[pb1 pb; Db1] * B2[pb pb2′; Db2]
    return norm(net1 - net2)
end

@testset "Cost function of LocalApprox" begin
    Random.seed!(0)
    Vaux, Vphy, V = ℂ^1, ℂ^10, ℂ^4
    A1 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ V ⊗ Vaux' ⊗ Vaux'), Inf)
    A2 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ V ⊗ Vaux' ⊗ Vaux'), Inf)
    B1 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ Vaux ⊗ Vaux' ⊗ V'), Inf)
    B2 = normalize(randn(Vphy ⊗ Vphy' ← Vaux ⊗ Vaux ⊗ Vaux' ⊗ V'), Inf)

    P1, s, P2, ϵ = localapprox_projector(A1, A2, B1, B2; trunc = notrunc())
    @test P1 * P2 ≈ TensorKit.id(domain(P2))

    P1, sc, P2, ϵ = localapprox_projector(A1, A2, B1, B2; trunc = truncrank(8))
    A1 = removeunit(removeunit(removeunit(A1, 6), 5), 3)
    A2 = removeunit(removeunit(removeunit(A2, 6), 5), 3)
    B1 = removeunit(removeunit(removeunit(B1, 5), 4), 3)
    B2 = removeunit(removeunit(removeunit(B2, 5), 4), 3)
    @info "Truncation error = $(ϵ)."
    @test ϵ ≈ localapprox_cost(A1, A2, B1, B2, P1, P2)
end
