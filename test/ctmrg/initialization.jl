using Test
using TensorKit
using PEPSKit
using Random

using MPSKitModels: classical_ising

sd = 12345

# toggle symmetry, but same issue for both
symmetries = [Z2Irrep, Trivial]

χ = 20
tol = 1.0e-4
maxiter = 1000
verbosity = 2
trscheme = FixedSpaceTruncation()
boundary_alg = (;
    alg = :simultaneous,
    tol,
    verbosity,
    trscheme,
    maxiter,
)

@testset "CTMRG environment initialization for critical ising with $S symmetry (#255)" for S in symmetries
    # initialize
    T = classical_ising(S)
    O = T[1]
    n = InfinitePartitionFunction([O O; O O])
    Venv = S == Z2Irrep ? Z2Space(0 => χ / 2, 1 => χ / 2) : ℂ^χ
    P = space(O, 2)

    # random, doesn't converge
    Random.seed!(sd)
    env0_rand = initialize_environment(n, RandomInitialization(), Venv)
    env_rand, info = leading_boundary(env0_rand, n; boundary_alg...)
    @test info.convergence_error > tol

    # embedded product state, converges
    Random.seed!(sd)
    env0_prod = initialize_environment(n, ProductStateInitialization(), Venv)
    env_prod, info = leading_boundary(env0_prod, n; boundary_alg...)
    @test info.convergence_error ≤ tol

    # grown product state, converges
    Random.seed!(sd)
    env0_appl = initialize_environment(n, ApplicationInitialization(), truncdim(χ))
    env_appl, info = leading_boundary(env0_appl, n; boundary_alg...)
    @test info.convergence_error ≤ tol
end
