```@meta
EditURL = "../../../../examples/bose_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/bose_hubbard)


# Optimizing the $U(1)$-symmetric Bose-Hubbard model

This example demonstrates the simulation of the two-dimensional Bose-Hubbard model. In
particular, the point will be to showcase the use of internal symmetries and finite
particle densities in PEPS ground state searches. As we will see, incorporating symmetries
into the simulation consists of initializing a symmetric Hamiltonian, PEPS state and CTM
environment - made possible through TensorKit.

But first let's seed the RNG and import the required modules:

````julia
using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528935);
````

## Defining the model

We will construct the Bose-Hubbard model Hamiltonian through the
[`bose_hubbard_model`](https://quantumkithub.github.io/MPSKitModels.jl/dev/man/models/#MPSKitModels.bose_hubbard_model),
function from MPSKitModels as reexported by PEPSKit. We'll simulate the model in its
Mott-insulating phase where the ratio $U/t$ is large, since in this phase we expect the
ground state to be well approximated by a PEPS with a manifest global $U(1)$ symmetry.
Furthermore, we'll impose a cutoff at 2 bosons per site, set the chemical potential to zero
and use a simple $1 \times 1$ unit cell:

````julia
t = 1.0
U = 30.0
cutoff = 2
mu = 0.0
lattice = InfiniteSquare(1, 1);
````

Next, we impose an explicit global $U(1)$ symmetry as well as a fixed particle number
density in our simulations. We can do this by setting the `symmetry` argument of the
Hamiltonian constructor to `U1Irrep` and passing one as the particle number density
keyword argument `n`:

````julia
symmetry = U1Irrep
n = 1
H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff, t, U, n);
````

Before we continue, it might be interesting to inspect the corresponding lattice physical
spaces (which is here just a $1 \times 1$ matrix due to the single-site unit cell):

````julia
physical_spaces = physicalspace(H)
````

````
1×1 Matrix{TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}}:
 (0 => 1, 1 => 1, -1 => 1)
````

Note that the physical space contains $U(1)$ charges -1, 0 and +1. Indeed, imposing a
particle number density of +1 corresponds to shifting the physical charges by -1 to
're-center' the physical charges around the desired density. When we do this with a cutoff
of two bosons per site, i.e. starting from $U(1)$ charges 0, 1 and 2 on the physical level,
we indeed get the observed charges.

## Characterizing the virtual spaces

When running PEPS simulations with explicit internal symmetries, specifying the structure of
the virtual spaces of the PEPS and its environment becomes a bit more involved. For the
environment, one could in principle allow the virtual space to be chosen dynamically during
the boundary contraction using CTMRG by using a truncation scheme that allows for this
(e.g. using `alg=:truncrank` or `alg=:trunctol` to truncate to a fixed total bond dimension
or singular value cutoff respectively). For the PEPS virtual space however, the structure
has to be specified before the optimization.

While there are a host of techniques to do this in an informed way (e.g. starting from a
simple update result), here we just specify the virtual space manually. Since we're dealing
with a model at unit filling our physical space only contains integer $U(1)$ irreps.
Therefore, we'll build our PEPS and environment spaces using integer $U(1)$ irreps centered
around the zero charge:

````julia
V_peps = U1Space(0 => 2, 1 => 1, -1 => 1)
V_env = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2);
````

## Finding the ground state

Having defined our Hamiltonian and spaces, it is just a matter of plugging this into the
optimization framework in the usual way to find the ground state. So, we first specify all
algorithms and their tolerances:

````julia
boundary_alg = (; tol = 1.0e-8, alg = :SimultaneousCTMRG, trunc = (; alg = :FixedSpaceTruncation))
gradient_alg = (; tol = 1.0e-6, maxiter = 10, solver_alg = (; alg = :GMRES))
optimizer_alg = (; tol = 1.0e-4, alg = :LBFGS, maxiter = 150, ls_maxiter = 2, ls_maxfg = 2);
````

!!! note
	Taking CTMRG gradients and optimizing symmetric tensors tends to be more problematic
    than with dense tensors. In particular, this means that one frequently needs to tweak
    the `boundary_alg`, `gradient_alg` and `optimizer_alg` settings. There rarely is a
    general-purpose set of settings which will always work, so instead one has to adjust
    the simulation settings for each specific application. For example, it might help to
    switch between the CTMRG flavors `alg=:SimultaneousCTMRG` and `alg=:SequentialCTMRG` to
    improve convergence. Of course the tolerances of the algorithms and their subalgorithms
    also have to be compatible. For more details on the available options, see the
    [`fixedpoint`](@ref) docstring.

Keep in mind that the PEPS is constructed from a unit cell of spaces, so we have to make a
matrix of `V_peps` spaces:

````julia
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = +1.693461429863e+00 +8.390974048721e-02im	err = 1.0000e+00
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514520887044e-11im	err = 3.6943030392e-09	time = 37.67 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity = 3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 9.360531870693e+00, ‖∇f‖ = 1.6944e+01
[ Info: LBFGS: iter    1, Δt  1.00 m: f = 1.243263783214e-01, ‖∇f‖ = 6.2855e+00, α = 1.56e+02, m = 0, nfg = 7
[ Info: LBFGS: iter    2, Δt 22.02 s: f = 6.548777843545e-02, ‖∇f‖ = 7.6025e+00, α = 5.34e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, Δt  3.09 s: f = -4.461379131753e-02, ‖∇f‖ = 1.6140e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt  2.80 s: f = -7.609472691605e-02, ‖∇f‖ = 1.4762e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt 11.01 s: f = -1.249108213575e-01, ‖∇f‖ = 3.1985e+00, α = 5.23e-01, m = 4, nfg = 3
[ Info: LBFGS: iter    6, Δt  4.73 s: f = -1.624764645567e-01, ‖∇f‖ = 1.2573e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt  2.52 s: f = -1.929649368717e-01, ‖∇f‖ = 9.7320e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  4.73 s: f = -2.079922246280e-01, ‖∇f‖ = 7.4754e-01, α = 1.47e-01, m = 7, nfg = 2
[ Info: LBFGS: iter    9, Δt  6.01 s: f = -2.208688416226e-01, ‖∇f‖ = 4.6773e-01, α = 3.12e-01, m = 8, nfg = 2
[ Info: LBFGS: iter   10, Δt  2.10 s: f = -2.284025294095e-01, ‖∇f‖ = 6.7374e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  1.99 s: f = -2.349433738399e-01, ‖∇f‖ = 4.4910e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  3.17 s: f = -2.446392726723e-01, ‖∇f‖ = 3.6587e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  1.87 s: f = -2.506137268599e-01, ‖∇f‖ = 3.1145e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt  1.53 s: f = -2.569782611539e-01, ‖∇f‖ = 2.5645e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  1.61 s: f = -2.640189974559e-01, ‖∇f‖ = 2.9045e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  3.04 s: f = -2.669361881754e-01, ‖∇f‖ = 3.6496e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt  1.62 s: f = -2.692294379003e-01, ‖∇f‖ = 1.1275e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt  1.46 s: f = -2.697415269340e-01, ‖∇f‖ = 9.0510e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt  1.34 s: f = -2.704462248900e-01, ‖∇f‖ = 8.3341e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt  2.83 s: f = -2.710177324987e-01, ‖∇f‖ = 5.9077e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, Δt  1.64 s: f = -2.713287436140e-01, ‖∇f‖ = 8.1709e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, Δt  1.48 s: f = -2.716041340213e-01, ‖∇f‖ = 3.8121e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt  1.55 s: f = -2.717477824224e-01, ‖∇f‖ = 3.9733e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt  2.90 s: f = -2.720364217449e-01, ‖∇f‖ = 4.5134e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt  1.45 s: f = -2.722018371576e-01, ‖∇f‖ = 9.0995e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt  1.22 s: f = -2.724075395453e-01, ‖∇f‖ = 3.4199e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt  1.62 s: f = -2.724774787761e-01, ‖∇f‖ = 1.9122e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt  3.02 s: f = -2.725210111611e-01, ‖∇f‖ = 2.2144e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt  1.61 s: f = -2.726086792351e-01, ‖∇f‖ = 2.7391e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt  1.39 s: f = -2.727251509422e-01, ‖∇f‖ = 2.4263e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt  2.60 s: f = -2.727484993106e-01, ‖∇f‖ = 2.6267e-02, α = 1.99e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   32, Δt  2.82 s: f = -2.727961690429e-01, ‖∇f‖ = 1.3010e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt  1.46 s: f = -2.728248487482e-01, ‖∇f‖ = 1.3368e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt  1.27 s: f = -2.728766589969e-01, ‖∇f‖ = 2.1584e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt  1.41 s: f = -2.729263182570e-01, ‖∇f‖ = 3.4474e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt  2.70 s: f = -2.729793745531e-01, ‖∇f‖ = 2.4629e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt  1.41 s: f = -2.730137433420e-01, ‖∇f‖ = 1.1231e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt  1.06 s: f = -2.730274839336e-01, ‖∇f‖ = 1.0104e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt  1.14 s: f = -2.730397243724e-01, ‖∇f‖ = 1.1262e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt  1.18 s: f = -2.730549146932e-01, ‖∇f‖ = 1.6667e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt  2.79 s: f = -2.730664892653e-01, ‖∇f‖ = 6.6117e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt  1.26 s: f = -2.730700327201e-01, ‖∇f‖ = 4.8442e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt  1.11 s: f = -2.730746811238e-01, ‖∇f‖ = 7.7188e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt  1.15 s: f = -2.730806470256e-01, ‖∇f‖ = 8.2012e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt  2.53 s: f = -2.730933394572e-01, ‖∇f‖ = 1.4327e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt  1.36 s: f = -2.731047972676e-01, ‖∇f‖ = 1.2100e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt  1.28 s: f = -2.731148278169e-01, ‖∇f‖ = 6.3224e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt  1.13 s: f = -2.731213320582e-01, ‖∇f‖ = 5.3534e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt  3.93 s: f = -2.731254405045e-01, ‖∇f‖ = 1.2948e-02, α = 3.19e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   50, Δt  1.24 s: f = -2.731326604930e-01, ‖∇f‖ = 1.0515e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt  1.20 s: f = -2.731499345461e-01, ‖∇f‖ = 7.0551e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt  1.26 s: f = -2.731563762101e-01, ‖∇f‖ = 8.3120e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt  2.52 s: f = -2.731611892874e-01, ‖∇f‖ = 3.4373e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt  1.27 s: f = -2.731631831905e-01, ‖∇f‖ = 2.9119e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt  1.11 s: f = -2.731657032659e-01, ‖∇f‖ = 5.1838e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, Δt  1.37 s: f = -2.731682294143e-01, ‖∇f‖ = 4.2175e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt  1.46 s: f = -2.731702062492e-01, ‖∇f‖ = 3.6058e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt  2.76 s: f = -2.731737713129e-01, ‖∇f‖ = 6.6877e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt  1.49 s: f = -2.731756813198e-01, ‖∇f‖ = 8.6637e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt  1.37 s: f = -2.731783468120e-01, ‖∇f‖ = 3.7331e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt  1.43 s: f = -2.731804331092e-01, ‖∇f‖ = 2.8581e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt  2.85 s: f = -2.731825116666e-01, ‖∇f‖ = 3.6462e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt  1.70 s: f = -2.731868013537e-01, ‖∇f‖ = 5.1067e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt  2.82 s: f = -2.731893835871e-01, ‖∇f‖ = 5.6991e-03, α = 4.59e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   65, Δt  2.72 s: f = -2.731917403384e-01, ‖∇f‖ = 2.6086e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt  1.37 s: f = -2.731930890544e-01, ‖∇f‖ = 2.6696e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt  1.22 s: f = -2.731943299385e-01, ‖∇f‖ = 5.3466e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt  1.14 s: f = -2.731961553724e-01, ‖∇f‖ = 4.6066e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt  1.23 s: f = -2.732006803401e-01, ‖∇f‖ = 4.9900e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt  3.90 s: f = -2.732022291290e-01, ‖∇f‖ = 2.7374e-03, α = 4.63e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   71, Δt  1.11 s: f = -2.732031225346e-01, ‖∇f‖ = 2.1229e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt  1.17 s: f = -2.732044876859e-01, ‖∇f‖ = 3.3103e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt  2.54 s: f = -2.732058172459e-01, ‖∇f‖ = 5.0518e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt  1.33 s: f = -2.732075577134e-01, ‖∇f‖ = 3.0706e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt  1.18 s: f = -2.732105521476e-01, ‖∇f‖ = 2.6272e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt  1.31 s: f = -2.732115612585e-01, ‖∇f‖ = 2.8835e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt  1.25 s: f = -2.732132760768e-01, ‖∇f‖ = 2.5376e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt  2.47 s: f = -2.732150752016e-01, ‖∇f‖ = 3.4192e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt  1.33 s: f = -2.732160040413e-01, ‖∇f‖ = 7.7713e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, Δt  1.09 s: f = -2.732182609394e-01, ‖∇f‖ = 2.5164e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, Δt  1.13 s: f = -2.732190881684e-01, ‖∇f‖ = 2.3323e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, Δt  1.18 s: f = -2.732204123387e-01, ‖∇f‖ = 3.6082e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, Δt  2.45 s: f = -2.732222943866e-01, ‖∇f‖ = 4.1290e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, Δt  1.13 s: f = -2.732240731889e-01, ‖∇f‖ = 4.8688e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, Δt  1.15 s: f = -2.732250101813e-01, ‖∇f‖ = 2.7764e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, Δt  1.22 s: f = -2.732254921166e-01, ‖∇f‖ = 1.1538e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, Δt  2.69 s: f = -2.732256991662e-01, ‖∇f‖ = 1.4513e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, Δt  1.64 s: f = -2.732261404359e-01, ‖∇f‖ = 2.1648e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, Δt  1.41 s: f = -2.732267048618e-01, ‖∇f‖ = 3.3902e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, Δt  1.34 s: f = -2.732274323417e-01, ‖∇f‖ = 2.0604e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, Δt  1.41 s: f = -2.732282234574e-01, ‖∇f‖ = 2.0812e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, Δt  3.02 s: f = -2.732290360555e-01, ‖∇f‖ = 3.4645e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, Δt  1.49 s: f = -2.732301395008e-01, ‖∇f‖ = 4.1781e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, Δt  1.36 s: f = -2.732325993418e-01, ‖∇f‖ = 4.4416e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, Δt  1.49 s: f = -2.732340791849e-01, ‖∇f‖ = 7.1316e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, Δt  2.83 s: f = -2.732364241883e-01, ‖∇f‖ = 2.8036e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, Δt  1.56 s: f = -2.732370704978e-01, ‖∇f‖ = 1.3030e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, Δt  1.36 s: f = -2.732373186335e-01, ‖∇f‖ = 1.3910e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, Δt  1.41 s: f = -2.732376341405e-01, ‖∇f‖ = 1.8924e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, Δt  1.49 s: f = -2.732379877064e-01, ‖∇f‖ = 1.5303e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, Δt  2.97 s: f = -2.732382832257e-01, ‖∇f‖ = 1.1434e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, Δt  1.50 s: f = -2.732386061229e-01, ‖∇f‖ = 1.6328e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, Δt  1.44 s: f = -2.732389876149e-01, ‖∇f‖ = 1.9787e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, Δt  1.52 s: f = -2.732391227889e-01, ‖∇f‖ = 3.6378e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, Δt  1.53 s: f = -2.732396553193e-01, ‖∇f‖ = 1.1043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, Δt  2.95 s: f = -2.732398393026e-01, ‖∇f‖ = 1.0449e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, Δt  1.59 s: f = -2.732401014658e-01, ‖∇f‖ = 1.8022e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, Δt  1.35 s: f = -2.732404781378e-01, ‖∇f‖ = 2.2928e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, Δt  1.38 s: f = -2.732411760202e-01, ‖∇f‖ = 3.0905e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, Δt  3.88 s: f = -2.732416862589e-01, ‖∇f‖ = 1.8335e-03, α = 4.82e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  111, Δt  1.29 s: f = -2.732422082604e-01, ‖∇f‖ = 1.1584e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, Δt  1.11 s: f = -2.732427030710e-01, ‖∇f‖ = 1.8986e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, Δt  1.18 s: f = -2.732430266276e-01, ‖∇f‖ = 2.5947e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, Δt  2.44 s: f = -2.732434755923e-01, ‖∇f‖ = 2.3715e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, Δt  2.43 s: f = -2.732439915476e-01, ‖∇f‖ = 3.3800e-03, α = 5.10e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  116, Δt  1.10 s: f = -2.732446415159e-01, ‖∇f‖ = 1.0650e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, Δt  1.19 s: f = -2.732449637931e-01, ‖∇f‖ = 1.3755e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, Δt  2.42 s: f = -2.732456539893e-01, ‖∇f‖ = 2.4580e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, Δt  1.16 s: f = -2.732461388986e-01, ‖∇f‖ = 3.2084e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, Δt  1.07 s: f = -2.732466182744e-01, ‖∇f‖ = 1.3340e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, Δt  1.11 s: f = -2.732468054089e-01, ‖∇f‖ = 8.1872e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, Δt  2.50 s: f = -2.732469654397e-01, ‖∇f‖ = 1.4153e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, Δt  1.10 s: f = -2.732471591077e-01, ‖∇f‖ = 2.0357e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, Δt  1.12 s: f = -2.732476924969e-01, ‖∇f‖ = 2.9180e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, Δt  1.13 s: f = -2.732484527652e-01, ‖∇f‖ = 3.3604e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  126, Δt  1.28 s: f = -2.732486600040e-01, ‖∇f‖ = 5.2460e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, Δt  2.78 s: f = -2.732498333751e-01, ‖∇f‖ = 1.6849e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, Δt  1.40 s: f = -2.732502276809e-01, ‖∇f‖ = 9.8903e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, Δt  1.39 s: f = -2.732505399228e-01, ‖∇f‖ = 1.5368e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, Δt  1.39 s: f = -2.732508976184e-01, ‖∇f‖ = 2.2007e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, Δt  2.57 s: f = -2.732513239723e-01, ‖∇f‖ = 2.5983e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, Δt  1.38 s: f = -2.732517149815e-01, ‖∇f‖ = 1.4043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, Δt  1.12 s: f = -2.732519923295e-01, ‖∇f‖ = 8.2225e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, Δt  1.48 s: f = -2.732520835844e-01, ‖∇f‖ = 1.0367e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, Δt  2.88 s: f = -2.732522894832e-01, ‖∇f‖ = 1.3641e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, Δt  1.62 s: f = -2.732528229798e-01, ‖∇f‖ = 1.8242e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  137, Δt  2.75 s: f = -2.732530138347e-01, ‖∇f‖ = 2.6137e-03, α = 3.56e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  138, Δt  1.48 s: f = -2.732534193232e-01, ‖∇f‖ = 1.5676e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, Δt  2.84 s: f = -2.732537381142e-01, ‖∇f‖ = 1.3040e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, Δt  1.42 s: f = -2.732539953913e-01, ‖∇f‖ = 1.7209e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, Δt  1.14 s: f = -2.732542657646e-01, ‖∇f‖ = 1.7601e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, Δt  1.13 s: f = -2.732547537059e-01, ‖∇f‖ = 4.6367e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, Δt  2.57 s: f = -2.732555633970e-01, ‖∇f‖ = 1.3725e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, Δt  1.36 s: f = -2.732559306854e-01, ‖∇f‖ = 1.2581e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, Δt  1.29 s: f = -2.732565900521e-01, ‖∇f‖ = 2.1710e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, Δt  1.19 s: f = -2.732572155799e-01, ‖∇f‖ = 2.3025e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, Δt  3.82 s: f = -2.732576251058e-01, ‖∇f‖ = 2.1052e-03, α = 4.42e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  148, Δt  1.31 s: f = -2.732582065608e-01, ‖∇f‖ = 9.2327e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, Δt  1.06 s: f = -2.732583546173e-01, ‖∇f‖ = 1.7228e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 26.95 m: f = -2.732585263374e-01, ‖∇f‖ = 1.0519e-03
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/lbfgs.jl:199
E = -0.2732585263373664

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -9.646220406299748e-5

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

