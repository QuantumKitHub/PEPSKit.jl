using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: str, twistdual, _prev, _next

Vps = Dict(
    Z2Irrep => Vect[Z2Irrep](0 => 1, 1 => 2),
    U1Irrep => Vect[U1Irrep](0 => 2, 1 => 2, -1 => 1),
    FermionParity => Vect[FermionParity](0 => 1, 1 => 2),
)
Vvs = Dict(
    Z2Irrep => Vect[Z2Irrep](0 => 2, 1 => 2),
    U1Irrep => Vect[U1Irrep](0 => 3, 1 => 1, -1 => 2),
    FermionParity => Vect[FermionParity](0 => 2, 1 => 2),
)

function su_rdm_1x1(
        row::Int, col::Int, peps::InfinitePEPS, wts::Union{Nothing, SUWeight} = nothing
    )
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    t = peps.A[row, col]
    if !(wts === nothing)
        t = absorb_weight(t, wts, row, col, Tuple(1:4))
    end
    # contract local ⟨t|t⟩ without virtual twists
    @tensor ρ[k; b] := conj(t[b; n e s w]) * twistdual(t, 2:5)[k; n e s w]
    return ρ / str(ρ)
end

@testset "SUWeight ($(init) init, $(sect))" for (init, sect) in
    Iterators.product([:trivial, :random], keys(Vps))

    Vp, Vv = Vps[sect], Vvs[sect]
    Nspaces = [Vv Vv' Vv; Vv' Vv Vv']
    Espaces = [Vv Vv Vv'; Vv Vv' Vv']
    Pspaces = fill(Vp, size(Nspaces))
    peps = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
    wts = SUWeight(peps)
    if init != :trivial
        rand!(wts)
        normalize!.(wts.data, Inf)
    end
    env = CTMRGEnv(wts)
    for (r, c) in Tuple.(CartesianIndices(peps.A))
        ρ1 = su_rdm_1x1(r, c, peps, wts)
        if init == :trivial
            @test ρ1 ≈ su_rdm_1x1(r, c, peps, nothing)
        end
        ρ2 = reduced_densitymatrix(((r, c),), peps, env)
        @test ρ1 ≈ ρ2
    end
end
