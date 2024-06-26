using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# Initialize parameters
H = square_lattice_pwave()
χbond = 2
χenv = 16
ctm_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, fixedspace=true, verbosity=1
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=3, verbosity=2),
    reuse_env=true,
    verbosity=2,
)

# initialize states
Random.seed!(96678827397)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Vspace = Vect[FermionParity](0 => χbond ÷ 2, 1 => χbond ÷ 2)
Envspace = Vect[FermionParity](0 => χenv ÷ 2, 1 => χenv ÷ 2)
psi_init = InfinitePEPS(Pspace, Vspace, Vspace; unitcell=(2, 2))
env_init = leading_boundary(CTMRGEnv(psi_init; Venv=Envspace), psi_init, ctm_alg);

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

@test result.E / prod(size(psi_init)) ≈ -2.60053 atol = 1e-2 #comparison with Gaussian PEPS minimum at D=2 on 1000x1000 square lattice with aPBC
