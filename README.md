<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/docs/src/assets/logo-dark.svg">
    <img alt="PEPSKit.jl logo" src="https://github.com/QuantumKitHub/PEPSKit.jl/blob/master/docs/src/assets/logo.svg" width="150">
</picture>

# PEPSKit.jl

[![docs][docs-dev-img]][docs-dev-url] ![CI][ci-url] [![codecov][codecov-img]][codecov-url] [![DOI][doi-img]][doi-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://QuantumKitHub.github.io/PEPSKit.jl/dev/

[codecov-img]: https://codecov.io/gh/QuantumKitHub/PEPSKit.jl/graph/badge.svg?token=1OBDY03SUP
[codecov-url]: https://codecov.io/gh/QuantumKitHub/PEPSKit.jl

[ci-url]: https://github.com/QuantumKitHub/PEPSKit.jl/workflows/CI/badge.svg

[doi-url]: https://doi.org/10.5281/zenodo.13938736
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.13938737.svg

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
ctm = leading_boundary(CTMRGEnv(state, ComplexSpace(chi)), state, ctm_alg)
result = fixedpoint(state, H, opt_alg, ctm)

@show result.E # -0.6625...
```

