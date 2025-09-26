# PEPSKit.jl

**Tools for working with projected entangled-pair states**

It contracts, it optimizes, it may break.

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
- Imaginary time evolution algorithms
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

For a more in-depth explanation of this simple example, check the [Optimizing the 2D Heisenberg model](@ref examples_heisenberg) tutorial or consult the Manual pages.

## Table of contents

A detailed rundown of PEPSKit's features can be found in the Manual section (not yet complete, more coming soon™), including:

```@contents
Pages = ["man/models.md", "man/multithreading.md", "man/precompilation.md"]
Depth = 1
```

Additionally, we provide a list of commented examples in the [Examples section](@ref e_overview) which showcases most of PEPSKit's capabilities in action.
