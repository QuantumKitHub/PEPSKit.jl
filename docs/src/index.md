# PEPSKit.jl

**Tools for working with projected entangled-pair states**

It contracts, it optimizes, it may break.

## Installation

The package can be installed through the Julia general registry, via the package manager:

```julia-repl
pkg> add PEPSKit
```

## Quickstart

After following the installation process, it should now be possible to load the packages and start simulating.
For example, in order to obtain the groundstate of the 2D Heisenberg model, we can use the following code:

```julia
using TensorKit, PEPSKit, KrylovKit, OptimKit

# constructing the Hamiltonian:
Jx, Jy, Jz = (-1, 1, -1) # sublattice rotation to obtain single-site unit cell
physical_space = ComplexSpace(2)
T = ComplexF64
σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
Heisenberg_hamiltonian = NLocalOperator{NearestNeighbor}(H / 4)

# configuring the parameters
D = 2
chi = 20
ctm_alg = CTMRG(; trscheme = truncdim(chi), tol=1e-20, miniter=4, maxiter=100, verbosity=1)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=100),
    reuse_env=true,
    verbosity=2,
)

# ground state search
state = InfinitePEPS(2, D)
ctm = leading_boundary(CTMRGEnv(state, ComplexSpace(chi)), state, ctm_alg)
result = fixedpoint(state, Heisenberg_hamiltonian, opt_alg, ctm)

@show result.E # -0.6625...
```
