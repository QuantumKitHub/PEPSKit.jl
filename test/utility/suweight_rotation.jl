using Test
using TensorKit
using PEPSKit
using PEPSKit: _absorb_weights

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

function test_rotation(peps::InfinitePEPS, wts::SUWeight)
    for n in 1:4
        rot = compose_n(rotl90, n)
        A1 = InfinitePEPS(
            collect(
                _absorb_weights(peps.A[idx], idx[1], idx[2], Tuple(1:4), wts) for
                idx in CartesianIndices(peps.A)
            ),
        )
        peps2, wts2 = rot(peps), rot(wts)
        A2 = InfinitePEPS(
            collect(
                _absorb_weights(peps2.A[idx], idx[1], idx[2], Tuple(1:4), wts2) for
                idx in CartesianIndices(peps2.A)
            ),
        )
        @test A2 ≈ rot(A1)
    end
end

Vphy = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1//2) => 1, (1, -1//2) => 2)
Vs = (
    # Espace
    Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1//2) => 3, (1, -1//2) => 2),
    # Nspace
    Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1//2) => 1, (1, -1//2) => 4),
)
Nr, Nc = 2, 3
peps = InfinitePEPS(rand, Float64, Vphy, Vs[2], Vs[1]'; unitcell=(Nr, Nc))
wts = collect(
    tsvd(rand(Float64, Vs[dir] ← Vs[dir]))[2] for dir in 1:2, r in 1:Nr, c in 1:Nc
)
wts = SUWeight(wts)

@test sectortype(wts) === sectortype(Vs[1])
@test spacetype(wts) === spacetype(Vs[1])

test_rotation(wts)
test_rotation(peps, wts)
