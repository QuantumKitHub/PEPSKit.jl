using Test, TestExtras
using Random
using TensorKit
using PEPSKit
using PEPSKit: SUState, gauge_fix, compare_weights

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

function random_dual!(Vs::AbstractMatrix{E}) where {E <: ElementarySpace}
    for (i, V) in enumerate(Vs)
        (rand() < 0.7) && (Vs[i] = V')
    end
    return Vs
end

function gauge_fix_su(peps::InfinitePEPS; maxiter::Int = 100, tol::Float64 = 1.0e-6)
    H = dummy_ham(scalartype(peps), physicalspace(peps))
    alg = SimpleUpdate(; trunc = FixedSpaceTruncation())
    wts0 = SUWeight(peps)
    # use default constructor to avoid calculation of exp(-H * 0)
    evolver = TimeEvolver(alg, 0.0, maxiter, H, SUState(0, 0.0, peps, wts0))
    for (i, (peps, wts, info)) in enumerate(evolver)
        ϵ = compare_weights(wts, wts0)
        if i % 10 == 0 || ϵ < tol
            @info "SU gauging step $i: ϵ = $ϵ."
            (ϵ < tol) && return peps, wts, ϵ
        end
        wts0 = deepcopy(wts)
    end
    return
end

isapproxone(X; kwargs...) = isapprox(X, id!(similar(X)); kwargs...)

@testset "Compare BP and SU (no symmetry)" begin
    unitcell = (3, 3)
    stype = ComplexF64
    maxiter, tol = 100, 1.0e-9

    Random.seed!(0)
    Pspaces = ComplexSpace.(rand(2:3, unitcell...))
    Nspaces = random_dual!(ComplexSpace.(rand(2:4, unitcell...)))
    Espaces = random_dual!(ComplexSpace.(rand(2:4, unitcell...)))
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)

    # start by gauging with SU
    peps1, wts1 = gauge_fix_su(peps; maxiter, tol)
    for (a1, a2) in zip(peps1.A, peps.A)
        @test space(a1) == space(a2)
    end
    normalize!.(wts1.data)

    # gauging again with BP should give unitary gauge
    bp_alg = BeliefPropagation(; maxiter, tol)
    env0 = BPEnv(ones, stype, peps)
    peps2, XXinv, env = @constinferred gauge_fix(peps1, bp_alg, env0)
    for (a1, a2) in zip(peps1.A, peps.A)
        @test space(a1) == space(a2)
    end
    for (X, Xinv) in XXinv
        @test inv(X) ≈ adjoint(X) ≈ Xinv
    end
end
