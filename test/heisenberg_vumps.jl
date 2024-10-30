using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
χbond = 2
χenv = 16

# initialize states
Random.seed!(91283219347)
H = heisenberg_XYZ(InfiniteSquare())
psi_init = InfinitePEPS(2, χbond; unitcell=(1, 1))

# find fixedpoint one-site vumps
boundary_alg = VUMPS(
            ifupdown=false,
            tol=1e-10,
            miniter=0,
            maxiter=10,
            verbosity=2
)
opt_alg = PEPSOptimize(;
    boundary_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-6, verbosity=2),
    gradient_alg=nothing,
    reuse_env=true
)
env_init = VUMPSRuntime(psi_init, χenv, boundary_alg)
result = fixedpoint(psi_init, H, opt_alg, env_init; 
                    symmetrization=RotateReflect() 
                    );
@test result.E ≈ -0.66023 atol = 1e-4

# find fixedpoint two-site vumps
boundary_alg = VUMPS(
            ifupdown=true,
            tol=1e-10,
            miniter=0,
            maxiter=10,
            verbosity=2
)
opt_alg = PEPSOptimize(;
    boundary_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-6, verbosity=2),
    gradient_alg=nothing,
    reuse_env=true
)
env_init = VUMPSRuntime(psi_init, χenv, boundary_alg)
result = fixedpoint(psi_init, H, opt_alg, env_init);
@test result.E ≈ -0.66251 atol = 1e-4