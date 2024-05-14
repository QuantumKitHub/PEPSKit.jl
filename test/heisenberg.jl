using Test
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

"""
    square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)

Square lattice Heisenberg model. By default, this implements a single site unit cell via a sublattice rotation.
"""
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    physical_space = ComplexSpace(2)
    T = ComplexF64
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return NLocalOperator{NearestNeighbor}(H / 4)
end

# Initialize parameters
H = square_lattice_heisenberg()
χbond = 2
χenv = 16
verbosity = 2
ctm_alg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity),
    gradient_alg=GMRES(; tol=1e-6, maxiter=100),
    reuse_env=true,
    verbosity,
)

# initialize states
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(psi_init, ctm_alg, CTMRGEnv(psi_init; Venv=ComplexSpace(χenv)))

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

@test result.E ≈ -0.6694421 atol = 1e-2
