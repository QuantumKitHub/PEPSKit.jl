using Test
using TensorKit
using PEPSKit
using Random

using MPSKitModels: classical_ising

sd = 12345

# toggle symmetry, but same issue for both
symmetries = [Z2Irrep, Trivial]
make_space(::Type{Z2Irrep}, d::Int) = Z2Space(0 => d / 2, 1 => d / 2)
make_space(::Type{Trivial}, d::Int) = ComplexSpace(d)

d = 2
D = 4
χ = 20
tol = 1.0e-4
maxiter = 1000
verbosity = 2
trunc = truncrank(χ)
boundary_alg = (; alg = :SimultaneousCTMRG, tol, verbosity, trunc, maxiter)

@testset "CTMRG environment initialization for critical ising with $S symmetry (#255)" for S in symmetries
    # initialize
    T = classical_ising(S)
    O = T[1]
    n = InfinitePartitionFunction([O O; O O])
    Venv = make_space(S, χ)
    P = space(O, 2)

    # random, doesn't converge
    Random.seed!(sd)
    env0_rand = initialize_ctmrg_environment(n, RandomInitialization(), Venv)
    env_rand, info = leading_boundary(env0_rand, n; boundary_alg...)
    @test_broken info.convergence_error ≤ tol

    # embedded random product state, converges
    Random.seed!(sd)
    env0_prod = initialize_ctmrg_environment(n, ProductStateInitialization())
    env_prod, info = leading_boundary(env0_prod, n; boundary_alg...)
    @test info.convergence_error ≤ tol

    # grown product state, converges
    Random.seed!(sd)
    env0_appl = initialize_ctmrg_environment(n, ApplicationInitialization())
    env_appl, info = leading_boundary(env0_appl, n; boundary_alg...)
    @test info.convergence_error ≤ tol

    # PEPS-specific identity initialization; should throw when used on partition functions
    Random.seed!(sd)
    @test_throws ArgumentError env0_prod_id = initialize_ctmrg_environment(n, IdentityInitialization())
end

@testset "CTMRG environment initialization for PEPS with $S symmetry" for S in symmetries
    # initialize
    P = make_space(S, d)
    Vpeps = make_space(S, D)
    Venv = make_space(S, χ)
    peps = InfinitePEPS(P, Vpeps; unitcell = (2, 2))
    n = InfiniteSquareNetwork(peps)

    # random, converges
    Random.seed!(sd)
    env0_rand = initialize_ctmrg_environment(n, RandomInitialization(), Venv)
    env_rand, info = leading_boundary(env0_rand, n; boundary_alg...)
    @test info.convergence_error ≤ tol

    # embedded random product state, converges
    Random.seed!(sd)
    env0_prod = initialize_ctmrg_environment(n, ProductStateInitialization())
    env_prod, info = leading_boundary(env0_prod, n; boundary_alg...)
    @test info.convergence_error ≤ tol

    # embedded product state as identity from ket to bra, converges
    Random.seed!(sd)
    env0_prod_id = initialize_ctmrg_environment(n, IdentityInitialization())
    env_prod, info = leading_boundary(env0_prod_id, n; boundary_alg...)
    @test info.convergence_error ≤ tol

    # grown product state, converges
    Random.seed!(sd)
    env0_appl = initialize_ctmrg_environment(n, ApplicationInitialization())
    env_appl, info = leading_boundary(env0_appl, n; boundary_alg...)
    @test info.convergence_error ≤ tol
end
