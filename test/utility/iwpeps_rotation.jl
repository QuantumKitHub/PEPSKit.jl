using Test
using TensorKit
using PEPSKit

function compose_n(f, n)
    n == 0 && return identity
    return f ∘ compose_n(f, n - 1)
end

function test_rotation(wts::SUWeight)
    @test compose_n(rotl90, 2)(wts) ≈ compose_n(rotr90, 2)(wts)
    @test compose_n(rotr90, 2)(wts) ≈ rot180(wts)
    @test (rotl90 ∘ rotr90)(wts) ≈ wts
    @test (rotr90 ∘ rotl90)(wts) ≈ wts
    @test compose_n(rotl90, 4)(wts) ≈ wts
    @test compose_n(rotl90, 4)(wts) ≈ wts
    @test compose_n(rot180, 2)(wts) ≈ wts
    return nothing
end

function test_rotation(psi::InfiniteWeightPEPS)
    peps = InfinitePEPS(psi)
    @test compose_n(rotl90, 2)(psi) ≈ compose_n(rotr90, 2)(psi)
    @test compose_n(rotr90, 2)(psi) ≈ rot180(psi)
    # flipping twice results in a twist
    psi_lr = (rotl90 ∘ rotr90)(psi)
    psi_rl = (rotr90 ∘ rotl90)(psi)
    @test psi_lr.weights ≈ psi_rl.weights ≈ psi.weights
    @test all(twist(v1, (3, 5)) ≈ v2 for (v1, v2) in zip(psi_lr.vertices, psi.vertices))
    @test all(twist(v1, (2, 4)) ≈ v2 for (v1, v2) in zip(psi_rl.vertices, psi.vertices))
    psi_l4 = compose_n(rotl90, 4)(psi)
    psi_r4 = compose_n(rotr90, 4)(psi)
    psi_2 = compose_n(rot180, 2)(psi)
    @test psi_l4 ≈ psi_r4 ≈ psi_2
    @test all(twist(v1, Tuple(2:5)) ≈ v2 for (v1, v2) in zip(psi_2.vertices, psi.vertices))
    # conversion to InfinitePEPS
    psi_l = rotl90(psi)
    peps_l = InfinitePEPS(psi_l)
    for (i, t) in enumerate(peps_l.A)
        peps_l.A[i] = flip(t, (3, 5); inv=true)
    end
    @test peps_l ≈ rotl90(peps)
    psi_r = rotr90(psi)
    peps_r = InfinitePEPS(psi_r)
    for (i, t) in enumerate(peps_r.A)
        peps_r.A[i] = flip(t, (2, 4); inv=true)
    end
    @test peps_r ≈ rotr90(peps)
    psi_2 = rot180(psi)
    peps_2 = InfinitePEPS(psi_2)
    for (i, t) in enumerate(peps_2.A)
        peps_2.A[i] = flip(t, Tuple(2:5); inv=true)
    end
    @test peps_2 ≈ rot180(peps)
end

Vphy = Vect[FermionParity](0 => 1, 1 => 1)
V = Vect[FermionParity](0 => 2, 1 => 2)
Nr, Nc = 2, 3
vertices = collect(rand(Float64, Vphy ← V ⊗ V ⊗ V' ⊗ V') for r in 1:Nr, c in 1:Nc)
weights = collect(tsvd(rand(Float64, V ← V))[2] for dir in 1:2, r in 1:Nr, c in 1:Nc)
weights = SUWeight(weights)
pepswt = InfiniteWeightPEPS(vertices, weights)

test_rotation(weights)
@static if pkgversion(TensorKit) >= v"0.14.6"
    test_rotation(pepswt)
end
