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

function test_rotation(state::Union{InfinitePEPS, InfinitePEPO}, wts::SUWeight)
    InfiniteState = (state isa InfinitePEPS) ? InfinitePEPS : InfinitePEPO
    A1 = InfiniteState(
        map(CartesianIndices(state.A)) do idx
            return absorb_weight(state.A[idx], wts, idx[1], idx[2], Tuple(1:4))
        end
    )
    for n in 1:4
        rot = compose_n(rotl90, n)
        state2, wts2 = rot(state), rot(wts)
        A2 = InfiniteState(
            map(CartesianIndices(state2.A)) do idx
                return absorb_weight(state2.A[idx], wts2, idx[1], idx[2], Tuple(1:4))
            end
        )
        @test A2 ≈ rot(A1)
    end
    return
end

Vphy = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 2)
Vs = (
    # Espace
    Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 3, (1, -1 // 2) => 2),
    # Nspace
    Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 4),
)
Nr, Nc = 2, 3
peps = InfinitePEPS(rand, Float64, Vphy, Vs[2], Vs[1]'; unitcell = (Nr, Nc))
pepo = InfinitePEPO(rand, Float64, Vphy, Vs[2], Vs[1]'; unitcell = (Nr, Nc, 1))
wts = SUWeight(peps)
rand!(wts)
normalize!.(wts.data, Inf)
# check that elements of wts are successfully randomized
@test !(wts ≈ SUWeight(peps))

@test sectortype(wts) === sectortype(Vs[1])
@test spacetype(wts) === spacetype(Vs[1])

test_rotation(wts)
test_rotation(peps, wts)

wts = SUWeight(pepo)
rand!(wts)
normalize!.(wts.data, Inf)
test_rotation(pepo, wts)
