using Test
using Random
using MatrixAlgebraKit
using TensorKit
using MPSKit
using PEPSKit, Adapt
using PEPSKit: peps_normalize

# initialize parameters
D = 2
χ = 16
unitcells = [(1, 1), (3, 4)]
projector_algs_asymm = [:HalfInfiniteProjector, :FullInfiniteProjector]
Ts = [Float64, ComplexF64]

# minimal subsets of the combinations tested below which still cover every option value at least once
minimal_combinations_unitcells = [((1, 1), :HalfInfiniteProjector), ((3, 4), :FullInfiniteProjector)]
minimal_combinations_fixedspace = [
    (:SequentialCTMRG, :FullInfiniteProjector), (:SimultaneousCTMRG, :HalfInfiniteProjector),
]

function ctmrg_flavors_unitcells(AT; minimal::Bool = false)
    return @testset "$(unitcell) unit cell with $projector_alg ($AT)" for (unitcell, projector_alg) in
        (minimal ? minimal_combinations_unitcells : Iterators.product(unitcells, projector_algs_asymm))
        # compute environments
        Random.seed!(32350283290358)
        psi = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(D); unitcell))
        env_sequential, = leading_boundary(
            CTMRGEnv(psi, ComplexSpace(χ)), psi; alg = :SequentialCTMRG, projector_alg
        )
        env_simultaneous, = leading_boundary(
            CTMRGEnv(psi, ComplexSpace(χ)), psi; alg = :SimultaneousCTMRG, projector_alg
        )

        # compare norms
        @test abs(norm(psi, env_sequential)) ≈ abs(norm(psi, env_simultaneous)) rtol = 1.0e-6

        # compare singular values
        CS_sequential = map(svd_vals, env_sequential.corners)
        CS_simultaneous = map(svd_vals, env_simultaneous.corners)
        ΔCS = maximum(splat(PEPSKit._singular_value_distance), zip(CS_sequential, CS_simultaneous))
        @test ΔCS < 1.0e-2

        TS_sequential = map(svd_vals, env_sequential.edges)
        TS_simultaneous = map(svd_vals, env_simultaneous.edges)
        ΔTS = maximum(splat(PEPSKit._singular_value_distance), zip(TS_sequential, TS_simultaneous))
        @test ΔTS < 1.0e-2

        # compare Heisenberg energies
        H = adapt(AT, heisenberg_XYZ(InfiniteSquare(unitcell...)))
        E_sequential = cost_function(psi, env_sequential, H)
        E_simultaneous = cost_function(psi, env_simultaneous, H)
        @test E_sequential ≈ E_simultaneous rtol = 1.0e-3
    end
end

function ctmrg_flavors_fixedspace_truncation(AT; minimal::Bool = false)
    # test fixedspace actually fixes space
    return @testset "Fixedspace truncation using $alg and $projector_alg ($AT)" for (alg, projector_alg) in (
            minimal ? minimal_combinations_fixedspace :
                Iterators.product([:SequentialCTMRG, :SimultaneousCTMRG], projector_algs_asymm)
        )
        Ds = ComplexSpace.(fill(2, 3, 3))
        χs = ComplexSpace.([16 17 18; 15 20 21; 14 19 22])
        psi = adapt(AT, InfinitePEPS(Ds, Ds, Ds))
        env = CTMRGEnv(psi, ComplexSpace.(rand(10:20, 3, 3)), ComplexSpace.(rand(10:20, 3, 3)))
        env2, = leading_boundary(
            env, psi; alg, maxiter = 1, trunc = FixedSpaceTruncation(), projector_alg
        )

        # check that the space is fixed
        @test all(space.(env.corners) .== space.(env2.corners))
        @test all(space.(env.edges) .== space.(env2.edges))
    end
end

function ctmrg_flavors_c4v(AT; eigh_alg = :QRIteration, minimal::Bool = false)
    projector_algs_c4v = [
        (:C4vQRProjector, :Householder),
        (:C4vEighProjector, eigh_alg), (:C4vEighProjector, :Lanczos),
    ]
    # minimal: test each projector alg once, alternating between scalar types
    combinations = minimal ? zip(Iterators.cycle(Ts), projector_algs_c4v) :
        Iterators.product(Ts, projector_algs_c4v)
    return @testset "C4v with ($T) - ($projector_alg, $decomp_alg) ($AT)" for (T, (projector_alg, decomp_alg)) in
        combinations

        Random.seed!(29358293829382)
        symm = RotateReflect()
        Vphys = ComplexSpace(2)
        Vpeps = ComplexSpace(D)
        Venv = ComplexSpace(χ)

        peps = adapt(AT, InfinitePEPS(randn, T, Vphys, Vpeps, Vpeps))
        peps = peps_normalize(symmetrize!(peps, symm))

        env₀ = initialize_random_c4v_env(peps, Venv)
        env, = leading_boundary(
            env₀, peps; alg = :C4vCTMRG, projector_alg,
            decomposition_alg = (; alg = decomp_alg)
        )
        @test env isa CTMRGEnv
    end
end
