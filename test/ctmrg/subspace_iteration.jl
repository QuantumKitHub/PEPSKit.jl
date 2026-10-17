using Test
using Random
using LinearAlgebra
using MatrixAlgebraKit
using TensorKit
using PEPSKit
using PEPSKit: ctmrg_iteration, initialize_projector_cache

"""Construct a CTMRG flavor configured with the SI projector for testing."""
function si_ctmrg(flavor; subspace, projector_kwargs = (;), kwargs...)
    projector_alg = merge(
        (; alg = :SubspaceIterationProjector, subspace), projector_kwargs
    )
    return flavor(; projector_alg, verbosity = 0, kwargs...)
end

@testset "SubspaceIterationProjector configuration" begin
    alg = SubspaceIterationProjector(; subspace = ℂ^10)
    @test alg.subspace == ℂ^10
    @test alg.subspace_tol == PEPSKit.Defaults.subspace_tol
    @test alg.min_subspace_iters == PEPSKit.Defaults.min_subspace_iters
    @test alg.decomposition_alg isa SVDAdjoint
    @test alg.orth_alg isa QRAdjoint
    @test alg.trunc isa FixedSpaceTruncation

    configured = SimultaneousCTMRG(;
        projector_alg = (;
            alg = :SubspaceIterationProjector,
            subspace = ℂ^9,
            subspace_tol = 1.0e-4,
            min_subspace_iters = 4,
            orth_alg = (; fwd_alg = (; alg = :Householder)),
        ),
    )
    @test configured.projector_alg.subspace == ℂ^9
    @test configured.projector_alg.subspace_tol == 1.0e-4
    @test configured.projector_alg.min_subspace_iters == 4
    @test configured.projector_alg.orth_alg.fwd_alg isa Householder

    @test_throws ArgumentError SimultaneousCTMRG(; projector_alg = :SubspaceIterationProjector)
    @test_throws ArgumentError SubspaceIterationProjector(; subspace = 10)
    @test_throws ArgumentError SubspaceIterationProjector(; subspace = ℂ^10, subspace_tol = -1)
    @test_throws ArgumentError SubspaceIterationProjector(; subspace = ℂ^10, min_subspace_iters = 0)
    @test_throws ArgumentError SubspaceIterationProjector(;
        subspace = ℂ^10, trunc = truncerror(; rtol = 1.0e-8)
    )
end

@testset "SI subspace convergence" begin
    Random.seed!(0x51ab)
    V = ComplexSpace(4)
    L = TensorMap(Matrix(Diagonal(ComplexF64[5, 3, 0.05, 0.01])), V ← V)
    R = id(V)
    L /= norm(L)
    R /= norm(R)
    trunc = truncspace(ComplexSpace(2))
    full_alg = MatrixAlgebraKit.TruncatedAlgorithm(DefaultAlgorithm(), trunc)
    Ufull, Sfull, Vfull, = svd_trunc(L * R, full_alg)
    alg = SubspaceIterationProjector(; subspace = ℂ^3)
    rangefinders = (; X = nothing, Y = nothing, trunc, recycle = false)
    local si
    for _ in 1:5
        si = PEPSKit._subspace_iteration_decomposition(L, R, alg, rangefinders)
        rangefinders = (; X = si.X, Y = si.Y, trunc, recycle = false)
    end

    @test norm(si.U * si.U' - Ufull * Ufull') < 1.0e-4
    @test norm(si.V' * si.V - Vfull' * Vfull) < 1.0e-4
    @test si.S / norm(si.S) ≈ Sfull / norm(Sfull) rtol = 1.0e-6
end

@testset "SI TensorMap rangefinders" begin
    Random.seed!(0x51c7)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell = (1, 1))
    network = InfiniteSquareNetwork(psi)
    env = CTMRGEnv(psi, ComplexSpace(4))
    alg = si_ctmrg(SimultaneousCTMRG; subspace = ℂ^104)

    for coordinate in PEPSKit.eachcoordinate(env, 1:4)
        corner = PEPSKit.EnlargedCorner(network, env, coordinate)
        materialized_corner = TensorMap(corner)
        @test domain(corner) == domain(materialized_corner)
        @test codomain(corner) == codomain(materialized_corner)
    end

    cache = initialize_projector_cache(network, env, alg)
    @test all(
        grid -> all(x -> x isa AbstractTensorMap, grid),
        (cache.X, cache.Y, cache.U, cache.V),
    )
    _, info, cache = ctmrg_iteration(network, env, alg, cache)
    @test all(x -> x isa AbstractTensorMap, (info.U..., info.V...))
    @test all(s -> s isa DiagonalTensorMap, info.S)
    X = first(cache.X)
    @test dim(domain(X)) < dim(alg.projector_alg.subspace)

    larger_env = CTMRGEnv(psi, ComplexSpace(5))
    _, _, reset_cache = ctmrg_iteration(network, larger_env, alg, cache)
    @test space(first(reset_cache.X)) != space(first(cache.X))
end

@testset "Sequential SI phase caches" begin
    Random.seed!(0x5e90)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(1); unitcell = (2, 3))
    network = InfiniteSquareNetwork(psi)
    env = CTMRGEnv(psi, ComplexSpace(2))
    alg = si_ctmrg(SequentialCTMRG; subspace = ℂ^7)
    cache = initialize_projector_cache(network, env, alg)
    env, _, cache = ctmrg_iteration(network, env, alg, cache)
    _, _, cache = ctmrg_iteration(network, env, alg, cache)
    @test all(grid -> size(grid) == (4, 2, 3), (cache.X, cache.Y, cache.U, cache.V))

    values = reshape(collect(1:24), 4, 2, 3)
    rotated = PEPSKit.SubspaceIterationCache(2, true, values, -values, 2values, -2values)
    for _ in 1:4
        rotated = rotl90(rotated)
    end
    @test (rotated.X, rotated.Y, rotated.U, rotated.V) ==
        (values, -values, 2values, -2values)
    @test (rotated.iteration, rotated.recycle) == (2, true)
end

@testset "SI rangefinder recycling" begin
    Random.seed!(0x51ec)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(1); unitcell = (1, 1))
    network = InfiniteSquareNetwork(psi)
    for flavor in (SimultaneousCTMRG, SequentialCTMRG)
        env = CTMRGEnv(psi, ComplexSpace(2))
        alg = si_ctmrg(
            flavor;
            subspace = ℂ^3,
            projector_kwargs = (; subspace_tol = 2.0, min_subspace_iters = 2),
        )
        cache = initialize_projector_cache(network, env, alg)
        for _ in 1:2
            env, info, cache = ctmrg_iteration(network, env, alg, cache)
        end
        @test !info.contraction_metrics.recycled
        @test cache.recycle
        _, info, cache = ctmrg_iteration(network, env, alg, cache)
        @test info.contraction_metrics.recycled
    end
end

@testset "SI unequal dense spaces" begin
    Pspaces = fill(ComplexSpace(2), 2, 2)
    Nspaces = ComplexSpace.([2 3; 3 2])
    Espaces = ComplexSpace.([3 2; 2 3])
    chis_north = ComplexSpace.([3 4; 4 3])
    chis_east = ComplexSpace.([4 3; 3 4])
    chis_south = ComplexSpace.([3 4; 4 3])
    chis_west = ComplexSpace.([4 3; 3 4])
    psi = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
    env = CTMRGEnv(
        randn,
        ComplexF64,
        psi,
        chis_north,
        chis_east,
        chis_south,
        chis_west,
    )
    for flavor in (SimultaneousCTMRG, SequentialCTMRG)
        alg = si_ctmrg(flavor; subspace = ℂ^9)
        env′, = ctmrg_iteration(InfiniteSquareNetwork(psi), env, alg)
        @test all(space.(env′.corners) .== space.(env.corners))
        @test all(space.(env′.edges) .== space.(env.edges))
    end
end

@testset "SI dense unit cells" begin
    flavors = (SimultaneousCTMRG, SequentialCTMRG)
    unitcells = ((1, 1), (2, 3))
    for (flavor_index, flavor) in enumerate(flavors),
            (unitcell_index, unitcell) in enumerate(unitcells)
        Random.seed!(0x51ce + 10 * flavor_index + unitcell_index)
        psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(1); unitcell)
        env = CTMRGEnv(psi, ComplexSpace(2))
        common = (; maxiter = 2, miniter = 2, tol = 0.0)
        env_full, = leading_boundary(
            env,
            psi,
            flavor(; projector_alg = :FullInfiniteProjector, verbosity = 0, common...),
        )
        env_si, = leading_boundary(
            env, psi, si_ctmrg(flavor; subspace = ℂ^7, common...)
        )

        @test abs(norm(psi, env_si)) ≈ abs(norm(psi, env_full)) rtol = 1.0e-7
        corner_distance = maximum(
            splat(PEPSKit._singular_value_distance),
            zip(map(svd_vals, env_si.corners), map(svd_vals, env_full.corners)),
        )
        edge_distance = maximum(
            splat(PEPSKit._singular_value_distance),
            zip(map(svd_vals, env_si.edges), map(svd_vals, env_full.edges)),
        )
        @test corner_distance < 1.0e-7
        @test edge_distance < 1.0e-7

        H = heisenberg_XYZ(InfiniteSquare(unitcell...))
        @test cost_function(psi, env_si, H) ≈ cost_function(psi, env_full, H) rtol = 1.0e-7
    end
end

@testset "SI symmetry and differentiation limitations" begin
    P = U1Space(0 => 1)
    V = U1Space(-1 => 1, 0 => 1, 1 => 1)
    psi = InfinitePEPS(randn, ComplexF64, fill(P, 1, 1), fill(V, 1, 1), fill(V, 1, 1))
    env = CTMRGEnv(psi, V)
    alg = SimultaneousCTMRG(;
        projector_alg = (; alg = :SubspaceIterationProjector, subspace = V)
    )
    @test_throws ArgumentError PEPSKit.check_input(
        leading_boundary, InfiniteSquareNetwork(psi), env, alg
    )

    @test_throws ArgumentError PEPSKit._check_algorithm_combination(
        SimultaneousCTMRG(;
            projector_alg = (; alg = :SubspaceIterationProjector, subspace = ℂ^7)
        ),
        FixedPointGradient(),
    )
end
