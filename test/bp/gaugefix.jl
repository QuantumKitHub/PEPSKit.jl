using Test, TestExtras
using Random
using TensorKit
using PEPSKit
using PEPSKit: SUState, gauge_fix, compare_weights, random_dual!

"""
A dummy Hamiltonian containing identity gates on all nearest neighbor bonds.
"""
function dummy_ham(elt::Type{<:Number}, lattice::Matrix{S}) where {S <: ElementarySpace}
    terms = []
    for site1 in CartesianIndices(lattice)
        r1, c1 = mod1.(Tuple(site1), size(lattice))
        for d in (CartesianIndex(1, 0), CartesianIndex(0, 1))
            site2 = site1 + d
            r2, c2 = mod1.(Tuple(site2), size(lattice))
            V1, V2 = lattice[r1, c1], lattice[r2, c2]
            h = TensorKit.id(elt, V1 ⊗ V2)
            push!(terms, (site1, site2) => h)
        end
    end
    return LocalOperator(lattice, terms...)
end

function gauge_fix_su(peps0::InfinitePEPS; maxiter::Int = 100, tol::Float64 = 1.0e-6)
    H = dummy_ham(scalartype(peps0), physicalspace(peps0))
    alg = SimpleUpdate(; trunc = FixedSpaceTruncation())
    wts0 = SUWeight(peps0)
    # use default constructor to avoid calculation of exp(-H * 0)
    evolver = TimeEvolver(alg, 0.0, maxiter, H, SUState(0, 0.0, peps0, wts0))
    for (i, (peps0, wts, info)) in enumerate(evolver)
        ϵ = compare_weights(wts, wts0)
        if i % 10 == 0 || ϵ < tol
            @info "SU gauging step $i: ϵ = $ϵ."
            (ϵ < tol) && return peps0, wts, ϵ
        end
        wts0 = deepcopy(wts)
    end
    return
end

isapproxone(X; kwargs...) = isapprox(X, id!(similar(X)); kwargs...)

@testset "Compare BP and SU $(S)" for S in [U1Irrep, FermionParity]
    unitcell = (2, 3)
    stype = ComplexF64
    maxiter, tol = 100, 1.0e-9

    Random.seed!(0)
    Pspaces, Nspaces, Espaces = if S == U1Irrep
        map(zip(rand(1:2, unitcell), rand(1:2, unitcell), rand(1:2, unitcell))) do (d0, d1, d2)
                Vect[S](0 => d0, 1 => d1, -1 => d2)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1, d2)
                Vect[S](0 => d0, 1 => d1, -1 => d2)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1, d2)
                Vect[S](0 => d0, 1 => d1, -1 => d2)
        end
    else
        map(zip(rand(2:3, unitcell), rand(2:3, unitcell))) do (d0, d1)
                Vect[S](0 => d0, 1 => d1)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1)
                Vect[S](0 => d0, 1 => d1)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1)
                Vect[S](0 => d0, 1 => d1)
        end
    end
    Nspaces, Espaces = random_dual!(Nspaces), random_dual!(Espaces)
    peps0 = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)

    # start by gauging with SU
    peps1, wts1 = gauge_fix_su(peps0; maxiter, tol)
    for (a0, a1) in zip(peps0.A, peps1.A)
        @test space(a0) == space(a1)
    end
    normalize!.(wts1.data)

    # find BP fixed point and SUWeight
    bp_alg = BeliefPropagation(; maxiter, tol)
    env = BPEnv(ones, stype, peps1)
    env, err = leading_boundary(env, peps1, bp_alg)
    wts2 = SUWeight(env)
    normalize!.(wts2.data)
    @test compare_weights(wts1, wts2) < 1.0e-9

    # BP should differ from SU only by a unitary gauge transformation
    bpg_alg = BPGauge()
    peps2, XXinv = @constinferred gauge_fix(peps1, bpg_alg, env)
    for (a1, a2) in zip(peps1.A, peps2.A)
        @test space(a1) == space(a2)
    end
    for (X, Xinv) in XXinv
        @test inv(X) ≈ adjoint(X) ≈ Xinv
    end
end
