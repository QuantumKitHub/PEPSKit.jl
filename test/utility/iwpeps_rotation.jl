using Test
using TensorKit
using PEPSKit

function compose_n(f, n)
    n == 0 && return identity
    return f ∘ compose_n(f, n - 1)
end

function test_rotation(obj)
    println(compose_n(rotl90, 2)(obj) ≈ compose_n(rotr90, 2)(obj))
    println(compose_n(rotr90, 2)(obj) ≈ rot180(obj))
    println((rotl90 ∘ rotr90)(obj) ≈ obj)
    println((rotr90 ∘ rotl90)(obj) ≈ obj)
    println(compose_n(rotl90, 4)(obj) ≈ obj)
    println(compose_n(rotl90, 4)(obj) ≈ obj)
    println(compose_n(rot180, 2)(obj) ≈ obj)
    return nothing
end

Vphy = Vect[FermionParity](0 => 1, 1 => 1)
V = Vect[FermionParity](0 => 2, 1 => 2)
# Vphy = ℂ^3
# V = ℂ^4
Nr, Nc = 1, 1
vertices = collect(rand(Float64, Vphy ← V ⊗ V ⊗ V' ⊗ V') for r in 1:Nr, c in 1:Nc)
weights = SUWeight(
    collect(tsvd(rand(Float64, V ← V))[2] for dir in 1:2, r in 1:Nr, c in 1:Nc)
)
pepswt = InfiniteWeightPEPS(vertices, weights)

test_rotation(weights)
println("----")
test_rotation(pepswt)
