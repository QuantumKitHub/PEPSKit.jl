using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iteration, gauge_fix, calc_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
maxiter = 400
ctmrg_algs = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs = [HalfInfiniteProjector, FullInfiniteProjector]
χ = 8
atol = 1e-4

@testset "Trivial symmetry ($T) - ($unitcell) - ($ctmrg_alg) - ($projector_alg)" for (
    T, unitcell, ctmrg_alg, projector_alg
) in Iterators.product(
    scalartypes, unitcells, ctmrg_algs, projector_algs
)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(χ)

    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)

    Random.seed!(987654321)  # Seed RNG to make random environment consistent
    env = CTMRGEnv(psi, ctm_space)
    alg = ctmrg_alg(; maxiter, projector_alg)

    env = leading_boundary(env, psi, alg)
    env′, = ctmrg_iteration(psi, env, alg)
    env_fixed, = gauge_fix(env, env′)
    @test calc_elementwise_convergence(env, env_fixed) ≈ 0 atol = atol
end

@testset "Z2 symmetry ($T) - ($unitcell) - ($ctmrg_alg) - ($projector_alg)" for (
    T, unitcell, ctmrg_alg, projector_alg
) in Iterators.product(
    scalartypes, unitcells, ctmrg_algs, projector_algs
)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 1, 1 => 1)
    ctm_space = Z2Space(0 => χ ÷ 2, 1 => χ ÷ 2)

    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)

    Random.seed!(29385293852)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(physical_space, peps_space; unitcell)
    env = CTMRGEnv(psi, ctm_space)
    alg = ctmrg_alg(; maxiter, projector_alg)

    env = leading_boundary(env, psi, alg)
    env′, = ctmrg_iteration(psi, env, alg)
    env_fixed, = gauge_fix(env, env′)
    @test calc_elementwise_convergence(env, env_fixed) ≈ 0 atol = atol
end
