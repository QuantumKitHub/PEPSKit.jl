using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: random_dual!

ds = Dict(
    Trivial => ℂ^2,
    U1Irrep => U1Space(i => d for (i, d) in zip(-1:1, (1, 1, 2))),
    FermionParity => Vect[FermionParity](0 => 2, 1 => 1)
)
Ds = Dict(
    Trivial => ℂ^3,
    U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (1, 3, 2))),
    FermionParity => Vect[FermionParity](0 => 3, 1 => 2)
)
Random.seed!(41973582)

function meas_sites(
        op::O, ψ::InfinitePEPS, env::Union{BPEnv, CTMRGEnv}
    ) where {O <: AbstractTensorMap{<:Any, <:Any, 1, 1}}
    lattice = physicalspace(ψ)
    return map(CartesianIndices(ψ.A)) do site1
        lo = LocalOperator(lattice, (site1,) => op)
        return expectation_value(ψ, lo, env)
    end
end

@testset "Rotation of BPEnv ($S)" for S in keys(ds)
    d, D, unitcell = ds[S], Ds[S], (2, 3)
    ψds = fill(d, unitcell)
    ψDNs = random_dual!(fill(D, unitcell))
    ψDEs = random_dual!(fill(D, unitcell))
    ψ = InfinitePEPS(ψds, ψDNs, ψDEs)
    env = BPEnv(ψ)

    op = randn(d → d)
    meas1 = meas_sites(op, ψ, env)
    # rotated peps and env
    for f in (rotl90, rotr90, rot180)
        ψ′, env′ = f(ψ), f(env)
        meas1′ = meas_sites(op, ψ′, env′)
        @test meas1′ ≈ f(meas1)
    end
end
