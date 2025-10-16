using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: str

function rand_wts(peps::InfinitePEPS)
    wts = SUWeight(peps)
    for idx in CartesianIndices(wts.data)
        n = length(wts.data[idx].data)
        wts.data[idx].data[:] = sort(rand(Float64, n); lt = !isless)
    end
    return wts
end

function su_rdm_1x1(row::Int, col::Int, peps::InfinitePEPS, wts::SUWeight)
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    t = peps.A[row, col]
    t = absorb_weight(t, wts, row, col, Tuple(1:4))
    ρ = t * t'
    return ρ / str(ρ)
end

Vps = [Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1), Vect[FermionParity](0 => 1, 1 => 2)]
Vvs = [Vect[U1Irrep](0 => 3, 1 => 1, -1 => 2), Vect[FermionParity](0 => 2, 1 => 2)]

for (Vp, Vv) in zip(Vps, Vvs)
    Nspaces = [Vv Vv' Vv; Vv' Vv Vv']
    Espaces = [Vv Vv Vv'; Vv Vv' Vv']
    Pspaces = fill(Vp, size(Nspaces))
    peps = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
    wts = rand_wts(peps)
    env = CTMRGEnv(wts, peps)
    for (r, c) in Tuple.(CartesianIndices(peps.A))
        ρ1 = su_rdm_1x1(r, c, peps, wts)
        ρ2 = reduced_densitymatrix(((r, c),), peps, env)
        @test ρ1 ≈ ρ2
    end
end
