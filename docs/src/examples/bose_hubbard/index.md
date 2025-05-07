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
 Rep[TensorKitSectors.U₁](0=>1, 1=>1, -1=>1)
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
(e.g. using `alg=:truncdim` or `alg=:truncbelow` to truncate to a fixed total bond dimension
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
boundary_alg = (; tol=1e-8, alg=:simultaneous, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, maxiter=10, alg=:eigsolver, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, maxiter=150, ls_maxiter=2, ls_maxfg=2);
````

!!! note
	Taking CTMRG gradients and optimizing symmetric tensors tends to be more problematic
    than with dense tensors. In particular, this means that one frequently needs to tweak
    the `boundary_alg`, `gradient_alg` and `optimizer_alg` settings. There rarely is a
    general-purpose set of settings which will always work, so instead one has to adjust
    the simulation settings for each specific application. For example, it might help to
    switch between the CTMRG flavors `alg=:simultaneous` and `alg=:sequential` to
    improve convergence. The evaluation of the CTMRG gradient can be instable, so there it
    is advised to try the different `iterscheme=:diffgauge` and `iterscheme=:fixed` schemes
    as well as different `alg` keywords. Of course the tolerances of the algorithms and
    their subalgorithms also have to be compatible. For more details on the available
    options, see the [`fixedpoint`](@ref) docstring.

Keep in mind that the PEPS is constructed from a unit cell of spaces, so we have to make a
matrix of `V_peps` spaces:

````julia
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = +1.693461429863e+00 +8.390974048721e-02im	err = 1.0000e+00
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514255254857e-11im	err = 3.6943032119e-09	time = 8.27 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 9.360531870693, ‖∇f‖ = 1.6957e+01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time  862.65 s: f = 0.112865330403, ‖∇f‖ = 5.9876e+00, α = 1.56e+02, m = 0, nfg = 7
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time  875.83 s: f = 0.031016651818, ‖∇f‖ = 4.7981e+00, α = 5.55e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  878.38 s: f = -0.073286659944, ‖∇f‖ = 1.4991e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  881.06 s: f = -0.113074511097, ‖∇f‖ = 1.4104e+00, α = 1.00e+00, m = 3, nfg = 1
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 4.53e-02, dϕ = -5.09e-01, ϕ - ϕ₀ = -2.42e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    5, time  888.55 s: f = -0.137293934550, ‖∇f‖ = 1.3317e+00, α = 4.53e-02, m = 4, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 4.19e-02, dϕ = -3.58e-01, ϕ - ϕ₀ = -1.56e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    6, time  896.03 s: f = -0.152882613133, ‖∇f‖ = 1.2515e+00, α = 4.19e-02, m = 5, nfg = 3
[ Info: LBFGS: iter    7, time  906.18 s: f = -0.167778524643, ‖∇f‖ = 3.0370e+00, α = 3.97e-01, m = 6, nfg = 4
[ Info: LBFGS: iter    8, time  908.59 s: f = -0.200610144885, ‖∇f‖ = 8.4562e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  911.30 s: f = -0.214869049363, ‖∇f‖ = 5.6088e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  913.43 s: f = -0.222910672089, ‖∇f‖ = 9.8015e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  915.49 s: f = -0.230707833300, ‖∇f‖ = 4.2302e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  917.75 s: f = -0.238105633372, ‖∇f‖ = 2.5801e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  919.64 s: f = -0.247331854867, ‖∇f‖ = 3.2459e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  920.70 s: f = -0.253845651144, ‖∇f‖ = 2.4014e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  922.02 s: f = -0.261289607017, ‖∇f‖ = 3.3777e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  922.92 s: f = -0.267178486858, ‖∇f‖ = 2.0556e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  923.72 s: f = -0.269417408686, ‖∇f‖ = 1.4442e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  924.84 s: f = -0.270255942689, ‖∇f‖ = 7.8602e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  925.66 s: f = -0.270672366603, ‖∇f‖ = 6.3259e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  926.46 s: f = -0.271220543802, ‖∇f‖ = 8.8755e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  927.59 s: f = -0.271543527453, ‖∇f‖ = 4.5233e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  928.38 s: f = -0.271650756946, ‖∇f‖ = 3.4057e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  929.16 s: f = -0.271894826592, ‖∇f‖ = 3.1507e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  930.27 s: f = -0.272056350007, ‖∇f‖ = 3.7796e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  931.08 s: f = -0.272233548851, ‖∇f‖ = 2.8370e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  931.89 s: f = -0.272383908712, ‖∇f‖ = 2.2903e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  933.01 s: f = -0.272455167221, ‖∇f‖ = 4.0448e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  933.79 s: f = -0.272553540301, ‖∇f‖ = 2.1431e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  934.56 s: f = -0.272695812729, ‖∇f‖ = 2.3543e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  935.67 s: f = -0.272771651851, ‖∇f‖ = 2.0076e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  936.45 s: f = -0.272799750606, ‖∇f‖ = 4.7874e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  937.23 s: f = -0.272878619780, ‖∇f‖ = 1.7612e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  938.34 s: f = -0.272929741282, ‖∇f‖ = 1.5455e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  939.14 s: f = -0.273001510559, ‖∇f‖ = 2.3641e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  939.94 s: f = -0.273050577801, ‖∇f‖ = 1.6253e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  941.06 s: f = -0.273061460544, ‖∇f‖ = 2.5195e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  941.84 s: f = -0.273079875270, ‖∇f‖ = 7.2299e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  942.63 s: f = -0.273083995266, ‖∇f‖ = 6.2322e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  943.75 s: f = -0.273091490830, ‖∇f‖ = 7.8002e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  944.54 s: f = -0.273097801979, ‖∇f‖ = 9.2495e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  945.33 s: f = -0.273103085883, ‖∇f‖ = 5.9273e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  946.45 s: f = -0.273106428950, ‖∇f‖ = 5.6938e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  947.24 s: f = -0.273111266609, ‖∇f‖ = 9.0725e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  948.06 s: f = -0.273121071175, ‖∇f‖ = 1.2386e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  949.20 s: f = -0.273137645098, ‖∇f‖ = 1.2942e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  950.82 s: f = -0.273141136808, ‖∇f‖ = 1.4282e-02, α = 1.35e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   47, time  951.96 s: f = -0.273154777599, ‖∇f‖ = 7.0317e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  952.79 s: f = -0.273161148021, ‖∇f‖ = 3.4952e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  953.57 s: f = -0.273164175099, ‖∇f‖ = 4.7738e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  954.68 s: f = -0.273166242670, ‖∇f‖ = 5.4242e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  955.48 s: f = -0.273168481006, ‖∇f‖ = 3.6862e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  956.27 s: f = -0.273172228386, ‖∇f‖ = 4.8463e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  957.40 s: f = -0.273174650673, ‖∇f‖ = 6.4469e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  958.18 s: f = -0.273178852481, ‖∇f‖ = 7.4990e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  958.98 s: f = -0.273186651745, ‖∇f‖ = 8.2158e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  960.10 s: f = -0.273190544009, ‖∇f‖ = 8.6751e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  960.89 s: f = -0.273194527074, ‖∇f‖ = 2.7365e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  961.68 s: f = -0.273195816168, ‖∇f‖ = 2.9114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  962.81 s: f = -0.273197625913, ‖∇f‖ = 3.0896e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  963.60 s: f = -0.273198605154, ‖∇f‖ = 1.0394e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  964.39 s: f = -0.273202461926, ‖∇f‖ = 3.0652e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  965.50 s: f = -0.273203714519, ‖∇f‖ = 2.0133e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  966.28 s: f = -0.273204828019, ‖∇f‖ = 2.5951e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  968.19 s: f = -0.273205465822, ‖∇f‖ = 4.1444e-03, α = 4.87e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   65, time  969.01 s: f = -0.273206458356, ‖∇f‖ = 2.9919e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  969.82 s: f = -0.273208249293, ‖∇f‖ = 1.6948e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  970.96 s: f = -0.273208839969, ‖∇f‖ = 3.0193e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  971.74 s: f = -0.273209433106, ‖∇f‖ = 1.8534e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  972.52 s: f = -0.273210019598, ‖∇f‖ = 1.7898e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  973.63 s: f = -0.273211075315, ‖∇f‖ = 2.7930e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  974.42 s: f = -0.273212703695, ‖∇f‖ = 3.6612e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  975.22 s: f = -0.273214163920, ‖∇f‖ = 6.1973e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  976.34 s: f = -0.273216147362, ‖∇f‖ = 2.7120e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  977.11 s: f = -0.273217199407, ‖∇f‖ = 2.2842e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  977.91 s: f = -0.273218358117, ‖∇f‖ = 3.0566e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  979.04 s: f = -0.273219638996, ‖∇f‖ = 3.8811e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time  979.83 s: f = -0.273221240011, ‖∇f‖ = 4.4440e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  980.62 s: f = -0.273222619191, ‖∇f‖ = 2.8356e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  981.74 s: f = -0.273223777532, ‖∇f‖ = 2.2842e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time  982.53 s: f = -0.273224631407, ‖∇f‖ = 2.6013e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time  983.33 s: f = -0.273225621866, ‖∇f‖ = 2.7625e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time  984.46 s: f = -0.273226222472, ‖∇f‖ = 2.5785e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time  985.23 s: f = -0.273226603039, ‖∇f‖ = 1.2203e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time  986.03 s: f = -0.273226890308, ‖∇f‖ = 1.1848e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time  987.16 s: f = -0.273227256564, ‖∇f‖ = 1.8281e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time  987.96 s: f = -0.273227922952, ‖∇f‖ = 2.0191e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time  989.91 s: f = -0.273228177117, ‖∇f‖ = 2.7409e-03, α = 3.30e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   88, time  990.72 s: f = -0.273228564518, ‖∇f‖ = 1.5762e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time  991.52 s: f = -0.273228984011, ‖∇f‖ = 1.3435e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time  992.64 s: f = -0.273229366257, ‖∇f‖ = 1.9460e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time  993.46 s: f = -0.273230207912, ‖∇f‖ = 2.9080e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time  994.25 s: f = -0.273231202568, ‖∇f‖ = 3.4541e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time  995.36 s: f = -0.273232085510, ‖∇f‖ = 1.8388e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time  996.15 s: f = -0.273232631167, ‖∇f‖ = 1.1594e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time  996.93 s: f = -0.273232992148, ‖∇f‖ = 1.7787e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time  998.04 s: f = -0.273233316686, ‖∇f‖ = 1.5449e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time  998.85 s: f = -0.273233676058, ‖∇f‖ = 1.8482e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time  999.62 s: f = -0.273233957076, ‖∇f‖ = 1.5452e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time 1000.72 s: f = -0.273234130308, ‖∇f‖ = 1.2047e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time 1001.53 s: f = -0.273234491906, ‖∇f‖ = 1.3723e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, time 1002.32 s: f = -0.273234862722, ‖∇f‖ = 2.2468e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time 1003.43 s: f = -0.273235395986, ‖∇f‖ = 1.8201e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time 1004.23 s: f = -0.273235884374, ‖∇f‖ = 1.7284e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time 1005.03 s: f = -0.273236273215, ‖∇f‖ = 1.3732e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time 1006.16 s: f = -0.273236575790, ‖∇f‖ = 1.4651e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time 1006.95 s: f = -0.273236948216, ‖∇f‖ = 1.8153e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time 1007.75 s: f = -0.273237435467, ‖∇f‖ = 2.6401e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, time 1008.87 s: f = -0.273237969174, ‖∇f‖ = 1.3197e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time 1009.67 s: f = -0.273238333588, ‖∇f‖ = 1.0300e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time 1010.46 s: f = -0.273238606412, ‖∇f‖ = 1.3512e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time 1011.58 s: f = -0.273238763494, ‖∇f‖ = 2.0868e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time 1012.38 s: f = -0.273238992097, ‖∇f‖ = 1.0034e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time 1013.17 s: f = -0.273239242010, ‖∇f‖ = 1.0445e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time 1014.29 s: f = -0.273239539038, ‖∇f‖ = 1.4904e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time 1015.09 s: f = -0.273239986741, ‖∇f‖ = 1.4425e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time 1017.02 s: f = -0.273240116483, ‖∇f‖ = 1.9754e-03, α = 2.24e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  117, time 1017.83 s: f = -0.273240380183, ‖∇f‖ = 9.8175e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time 1018.62 s: f = -0.273240575426, ‖∇f‖ = 8.6345e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time 1019.74 s: f = -0.273240832012, ‖∇f‖ = 1.4287e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, time 1020.53 s: f = -0.273241210750, ‖∇f‖ = 1.8219e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, time 1022.47 s: f = -0.273241482573, ‖∇f‖ = 2.4856e-03, α = 5.47e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  122, time 1023.28 s: f = -0.273241934058, ‖∇f‖ = 1.5033e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, time 1024.10 s: f = -0.273242166393, ‖∇f‖ = 1.2112e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, time 1025.25 s: f = -0.273242322900, ‖∇f‖ = 1.0134e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, time 1026.06 s: f = -0.273242508033, ‖∇f‖ = 1.0745e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  126, time 1026.87 s: f = -0.273242887106, ‖∇f‖ = 2.1256e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, time 1028.00 s: f = -0.273243341793, ‖∇f‖ = 1.6582e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, time 1028.81 s: f = -0.273243624575, ‖∇f‖ = 1.0434e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, time 1029.63 s: f = -0.273243926598, ‖∇f‖ = 1.0170e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, time 1030.76 s: f = -0.273244130570, ‖∇f‖ = 1.7663e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, time 1031.56 s: f = -0.273244384615, ‖∇f‖ = 1.4424e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, time 1032.37 s: f = -0.273244721419, ‖∇f‖ = 1.5925e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, time 1033.50 s: f = -0.273244883706, ‖∇f‖ = 1.0143e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, time 1034.30 s: f = -0.273244978610, ‖∇f‖ = 9.4449e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, time 1035.10 s: f = -0.273245503204, ‖∇f‖ = 1.2344e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, time 1036.22 s: f = -0.273245739413, ‖∇f‖ = 2.7009e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  137, time 1037.03 s: f = -0.273246059451, ‖∇f‖ = 1.4108e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, time 1037.84 s: f = -0.273246259501, ‖∇f‖ = 6.7238e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, time 1038.97 s: f = -0.273246408570, ‖∇f‖ = 9.2687e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, time 1039.79 s: f = -0.273246576606, ‖∇f‖ = 1.2150e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, time 1040.60 s: f = -0.273246729060, ‖∇f‖ = 1.3157e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, time 1041.73 s: f = -0.273246873413, ‖∇f‖ = 7.6123e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, time 1042.52 s: f = -0.273247027465, ‖∇f‖ = 7.9430e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, time 1043.32 s: f = -0.273247180548, ‖∇f‖ = 1.0894e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, time 1044.45 s: f = -0.273247387426, ‖∇f‖ = 2.2553e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, time 1045.24 s: f = -0.273247645050, ‖∇f‖ = 1.1468e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, time 1046.05 s: f = -0.273247846912, ‖∇f‖ = 9.1650e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, time 1047.17 s: f = -0.273248001067, ‖∇f‖ = 8.8945e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, time 1048.03 s: f = -0.273248315567, ‖∇f‖ = 1.2684e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 1050.00 s: f = -0.273248467359, ‖∇f‖ = 1.1430e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.2732484673593871

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -0.00013326986676584008

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

