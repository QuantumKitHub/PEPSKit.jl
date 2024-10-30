using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
χbond = 2
χenv = 16
boundary_alg = VUMPS(
            ifupdown=false,
            tol=1e-10,
            miniter=4,
            maxiter=10,
            verbosity=2
)
opt_alg = PEPSOptimize(;
    boundary_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-8, verbosity=2),
    gradient_alg=nothing,
    reuse_env=true
)

# initialize states
Random.seed!(91283219347)
H = heisenberg_XYZ(InfiniteSquare())
psi_init = InfinitePEPS(2, χbond; unitcell=(1, 1))
env_init = VUMPSRuntime(psi_init, χenv)

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init; 
                    symmetrization=RotateReflect());
@test result.E ≈ -0.66023 atol = 1e-4
