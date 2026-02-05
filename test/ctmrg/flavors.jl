using Test
using Random
using MatrixAlgebraKit
using TensorKit
using MPSKit
using PEPSKit
using PEPSKit: peps_normalize

# initialize parameters
D = 2
χ = 16
unitcells = [(1, 1), (3, 4)]
projector_algs_asymm = [:halfinfinite, :fullinfinite]
projector_algs_c4v = [
    (:c4v_qr, :qr),
    (:c4v_eigh, :qriteration), (:c4v_eigh, :lanczos),
]
Ts = [Float64, ComplexF64]

@testset "$(unitcell) unit cell with $projector_alg" for (unitcell, projector_alg) in
    Iterators.product(unitcells, projector_algs_asymm)
    # compute environments
    Random.seed!(32350283290358)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(D); unitcell)
    env_sequential, = leading_boundary(
        CTMRGEnv(psi, ComplexSpace(χ)), psi; alg = :sequential, projector_alg
    )
    env_simultaneous, = leading_boundary(
        CTMRGEnv(psi, ComplexSpace(χ)), psi; alg = :simultaneous, projector_alg
    )

    # compare norms
    @test abs(norm(psi, env_sequential)) ≈ abs(norm(psi, env_simultaneous)) rtol = 1.0e-6

    # compare singular values
    CS_sequential = map(svd_vals, env_sequential.corners)
    CS_simultaneous = map(svd_vals, env_simultaneous.corners)
    ΔCS = maximum(zip(CS_sequential, CS_simultaneous)) do (c_lm, c_as)
        smallest = infimum(MPSKit._firstspace(c_lm), MPSKit._firstspace(c_as))
        e_old = isometry(MPSKit._firstspace(c_lm), smallest)
        e_new = isometry(MPSKit._firstspace(c_as), smallest)
        return norm(e_new' * c_as * e_new - e_old' * c_lm * e_old)
    end
    @test ΔCS < 1.0e-2

    TS_sequential = map(svd_vals, env_sequential.edges)
    TS_simultaneous = map(svd_vals, env_simultaneous.edges)
    ΔTS = maximum(zip(TS_sequential, TS_simultaneous)) do (t_lm, t_as)
        MPSKit._firstspace(t_lm) == MPSKit._firstspace(t_as) || return scalartype(t_lm)(Inf)
        return norm(t_as - t_lm)
    end
    @test ΔTS < 1.0e-2

    # compare Heisenberg energies
    H = heisenberg_XYZ(InfiniteSquare(unitcell...))
    E_sequential = cost_function(psi, env_sequential, H)
    E_simultaneous = cost_function(psi, env_simultaneous, H)
    @test E_sequential ≈ E_simultaneous rtol = 1.0e-3
end

# test fixedspace actually fixes space
@testset "Fixedspace truncation using $alg and $projector_alg" for (alg, projector_alg) in
    Iterators.product([:sequential, :simultaneous], projector_algs_asymm)
    Ds = ComplexSpace.(fill(2, 3, 3))
    χs = ComplexSpace.([16 17 18; 15 20 21; 14 19 22])
    psi = InfinitePEPS(Ds, Ds, Ds)
    env = CTMRGEnv(psi, ComplexSpace.(rand(10:20, 3, 3)), ComplexSpace.(rand(10:20, 3, 3)))
    env2, = leading_boundary(
        env, psi; alg, maxiter = 1, trunc = FixedSpaceTruncation(), projector_alg
    )

    # check that the space is fixed
    @test all(space.(env.corners) .== space.(env2.corners))
    @test all(space.(env.edges) .== space.(env2.edges))
end

@testset "C4v with ($T) - ($projector_alg, $decomp_alg)" for (T, (projector_alg, decomp_alg)) in
    Iterators.product(Ts, projector_algs_c4v)

    Random.seed!(29358293829382)
    symm = RotateReflect()
    Vphys = ComplexSpace(2)
    Vpeps = ComplexSpace(D)
    Venv = ComplexSpace(χ)

    peps = InfinitePEPS(randn, T, Vphys, Vpeps, Vpeps)
    peps = peps_normalize(symmetrize!(peps, symm))

    env₀ = initialize_random_c4v_env(peps, Venv)
    env, = leading_boundary(
        env₀, peps; alg = :c4v, projector_alg,
        decomposition_alg = (; fwd_alg = (; alg = decomp_alg))
    )
    @test env isa CTMRGEnv
end
