<img src="./docs/src/assets/logo_readme.svg" width="150">

# PEPSKit.jl

| **Documentation** | **Build Status** | **Coverage** | **Digital Object Identifier** |
|:-----------------:|:----------------:|:------------:|:-----------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] | [![codecov][codecov-img]][codecov-url] | [![DOI][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://QuantumKitHub.github.io/PEPSKit.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://QuantumKitHub.github.io/PEPSKit.jl/dev/

[codecov-img]: https://codecov.io/gh/QuantumKitHub/PEPSKit.jl/graph/badge.svg?token=1OBDY03SUP
[codecov-url]: https://codecov.io/gh/QuantumKitHub/PEPSKit.jl

[ci-img]: https://github.com/QuantumKitHub/PEPSKit.jl/actions/workflows/Tests.yml/badge.svg
[ci-url]: https://github.com/QuantumKitHub/PEPSKit.jl/actions/workflows/Tests.yml

[doi-url]: https://doi.org/10.5281/zenodo.13938736
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.13938737.svg

**Tools for working with projected entangled-pair states**

*It contracts, it optimizes, it evolves.*

## Installation

The package can be installed through the Julia general registry, via the package manager:

```julia-repl
pkg> add PEPSKit
```

## Key features

- Construction and manipulation of infinite projected entangled-pair states (PEPS)
- Contraction of infinite PEPS using the corner transfer matrix renormalization group (CTMRG) and boundary MPS methods
- Native support for symmetric tensors through [TensorKit](https://github.com/Jutho/TensorKit.jl), including fermionic tensors
- PEPS optimization using automatic differentiation (AD) provided through [Zygote](https://fluxml.ai/Zygote.jl/stable/)
- Imaginary time evolution algorithms (including ground state and finite-temperature simulations)
- Support for PEPS with generic unit cells
- Support for classical 2D partition functions and projected entangled-pair operators (PEPOs)
- Extensible system for custom states, operators and algorithms

## Quickstart

After following the installation process, it should now be possible to load the packages and start simulating.
For example, in order to obtain the ground state of the 2D Heisenberg model, we can use the following code:

```julia
using TensorKit, PEPSKit

# construct the Hamiltonian
H = heisenberg_XYZ(InfiniteSquare())

# configure the parameters
D = 2
χ = 20
ctmrg_tol = 1e-10
grad_tol = 1e-4

# initialize a PEPS and CTMRG environment
peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χ)), peps₀; tol=ctmrg_tol)

# ground state search
peps, env, E, = fixedpoint(H, peps₀, env₀; tol=grad_tol, boundary_alg=(; tol=ctmrg_tol))

@show E # -0.6625...
```
