using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: random_dual!, eachcoordinate, NORTH

ds = Dict(
    U1Irrep => U1Space(i => d for (i, d) in zip(-1:1, (1, 1, 2))),
    FermionParity => Vect[FermionParity](0 => 2, 1 => 1)
)
Ds = Dict(
    U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (1, 3, 2))),
    FermionParity => Vect[FermionParity](0 => 3, 1 => 2)
)
χs = Dict(
    U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (2, 4, 2))),
    FermionParity => Vect[FermionParity](0 => 4, 1 => 4)
)

@testset "CTMRGEnv of InfinitePEPS ($S)" for S in keys(ds)
    d, D, χ, uc = ds[S], Ds[S], χs[S], (2, 3)
    ψds = fill(d, uc)
    ψDNs = random_dual!(fill(D, uc))
    ψDEs = random_dual!(fill(D, uc))
    ψ0 = InfinitePEPS(ψds, ψDNs, ψDEs)
    env0 = CTMRGEnv(randn, ComplexF64, ψ0, χ)
    # create a random set of gauge transformation on each bond
    XXinv = map(eachcoordinate(ψ0, 1:2)) do (d, r, c)
        V = (d == NORTH) ? ψDNs[r, c] : ψDEs[r, c]
        X = randn(ComplexF64, V → V)
        return (X, inv(X))
    end
    # gauge transform ψ and env
    ψ1 = gauge_transform(ψ0, XXinv)
    env1 = gauge_transform(env0, XXinv)
    # reduced density matrices should remain unchanged
    for (r, c) in eachcoordinate(ψ0)
        ρ0 = reduced_densitymatrix(((r, c),), ψ0, env0)
        ρ1 = reduced_densitymatrix(((r, c),), ψ1, env1)
        @test ρ0 ≈ ρ1
    end
end
