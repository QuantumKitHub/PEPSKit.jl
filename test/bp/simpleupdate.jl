using Test
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
        r1, c1 = (mod1(x, N) for (x, N) in zip(site1.I, size(lattice)))
        for d in (CartesianIndex(1, 0), CartesianIndex(0, 1))
            site2 = site1 + d
            r2, c2 = (mod1(x, N) for (x, N) in zip(site2.I, size(lattice)))
            V1, V2 = lattice[r1, c1], lattice[r2, c2]
            h = TensorKit.id(elt, V1 ⊗ V2)
            push!(terms, (site1, site2) => h)
        end
    end
    return LocalOperator(lattice, terms...)
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

@testset "Compare BP and SU (no symmetry)" begin
    unitcell = (3, 3)
    stype = ComplexF64
    maxiter, tol = 100, 1.0e-9

    Random.seed!(0)
    Pspaces = ComplexSpace.(rand(2:3, unitcell...))
    Nspaces = ComplexSpace.(rand(2:4, unitcell...))
    Espaces = ComplexSpace.(rand(2:4, unitcell...))
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)

    peps1, wts1 = gauge_fix_su(peps; maxiter, tol)
    normalize!.(wts1.data)

    bp_alg = BeliefPropagation(; maxiter, tol)
    env0 = BPEnv(ones, stype, peps)
    peps2, wts2 = gauge_fix(peps, bp_alg, env0)
    normalize!.(wts2.data)

    # Even with the same bond weights, the PEPS can still
    # differ by a unitary gauge transformation on virtual legs.
    @test isapprox(wts1, wts2)
end
