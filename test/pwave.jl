using Test
using Random
using TensorKit
using KrylovKit
using PEPSKit

# Initialize parameters
unitcell = (2, 2)
H = pwave_superconductor(InfiniteSquare(unitcell...))
χbond = 2
χenv = 16
ctm_alg = SimultaneousCTMRG(; maxiter=150)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; maxiter=10, gradtol=1e-3, verbosity=2)
)

# initialize states
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Vspace = Vect[FermionParity](0 => χbond ÷ 2, 1 => χbond ÷ 2)
Envspace = Vect[FermionParity](0 => χenv ÷ 2, 1 => χenv ÷ 2)
Random.seed!(91283219347)
psi_init = InfinitePEPS(Pspace, Vspace, Vspace; unitcell)
env_init = leading_boundary(CTMRGEnv(psi_init, Envspace), psi_init, ctm_alg);

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

# comparison with Gaussian PEPS minimum at D=2 on 1000x1000 square lattice with aPBC
@test result.E / prod(size(psi_init)) ≈ -2.6241 atol = 5e-2
