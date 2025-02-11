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
H = heisenberg_XYZ(InfiniteSquare(); Jx=-1, Jy=1, Jz=-1) # sublattice rotation to obtain single-site unit cell

# configuring the parameters
D = 2
chi = 20
ctm_alg = SimultaneousCTMRG(; tol=1e-10, trscheme=truncdim(chi))
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=3),
    gradient_alg=LinSolver(),
    reuse_env=true,
)

# ground state search
state = InfinitePEPS(2, D)
ctm, = leading_boundary(CTMRGEnv(state, ComplexSpace(chi)), state, ctm_alg)
peps, env, E, = fixedpoint(H, state, ctm, opt_alg)

@show E # -0.6625...
```
