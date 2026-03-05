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
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, trunc = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, maxiter = 10, alg = :eigsolver, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, maxiter = 150, ls_maxiter = 2, ls_maxfg = 2);
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
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.515735205612e-11im	err = 3.6943029805e-09	time = 7.12 sec

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
┌ Warning: Fixed-point gradient computation using Arnoldi failed:
│ 	auxiliary component should be finite but was -7.675459394744023e-9 + 0.0im
│ 	possibly the Jacobian does not have a unique eigenvalue 1
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/optimization/fixed_point_differentiation.jl:497
[ Info: Falling back to linear solver for fixed-point gradient computation.
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 5.265374663940242e-9 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 5.820766091346741e-11)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 1.5798722040141302e-9 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 2.1364030544646084e-9)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 7.293393124916005e-10 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.5734258340671659e-9)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 1.0233883965879778e-9 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 4.3655745685100555e-11)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 4.822766471246583e-10 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.3096723705530167e-10)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 3.424672430431249e-10 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 4.729372449219227e-11)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 2.7090554645808807e-10 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.5643308870494366e-10)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 2.626054619445045e-10 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 4.320099833421409e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangent linear problem returns unexpected result: error = 1.551993161126192e-11 vs tol = 1.0e-12
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:299
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 3.1725733151688473e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 7.263523116307624e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.3784529073745944e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 2.7569058147491887e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 2.19824158875781e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.4779288903810084e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.4066525722000733e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: Fixed-point gradient computation using Arnoldi failed:
│ 	auxiliary component should be finite but was -2.9567896256434878e-9 + 0.0im
│ 	possibly the Jacobian does not have a unique eigenvalue 1
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/optimization/fixed_point_differentiation.jl:497
[ Info: Falling back to linear solver for fixed-point gradient computation.
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 5.3717030823463574e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 6.139089236967266e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.1368683772161603e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 2.0094148567295633e-11)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.5916157281026244e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
┌ Warning: `eigsolve` cotangents sensitive to gauge choice: (|Δgauge| = 1.5845103007450234e-12)
└ @ KrylovKitChainRulesCoreExt ~/.julia/packages/KrylovKit/ZcdRg/ext/KrylovKitChainRulesCoreExt/eigsolve.jl:212
[ Info: LBFGS: iter    1, Δt  3.69 m: f = 1.243260265733e-01, ‖∇f‖ = 6.2855e+00, α = 1.56e+02, m = 0, nfg = 7
[ Info: LBFGS: iter    2, Δt 24.14 s: f = 6.540464570417e-02, ‖∇f‖ = 7.5894e+00, α = 5.34e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, Δt  1.46 s: f = -4.474083024431e-02, ‖∇f‖ = 1.6126e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt  1.46 s: f = -7.620383117375e-02, ‖∇f‖ = 1.4755e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt  5.30 s: f = -1.235688818436e-01, ‖∇f‖ = 3.2490e+00, α = 5.23e-01, m = 4, nfg = 3
[ Info: LBFGS: iter    6, Δt  1.64 s: f = -1.619496132224e-01, ‖∇f‖ = 1.2602e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt  1.48 s: f = -1.925928573609e-01, ‖∇f‖ = 9.7802e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  3.02 s: f = -2.076673801923e-01, ‖∇f‖ = 7.5446e-01, α = 1.45e-01, m = 7, nfg = 2
[ Info: LBFGS: iter    9, Δt  2.73 s: f = -2.206428218408e-01, ‖∇f‖ = 4.7295e-01, α = 3.05e-01, m = 8, nfg = 2
[ Info: LBFGS: iter   10, Δt  1.46 s: f = -2.281911819848e-01, ‖∇f‖ = 6.9226e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  1.18 s: f = -2.346626994273e-01, ‖∇f‖ = 4.4525e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  1.23 s: f = -2.442699596566e-01, ‖∇f‖ = 3.6315e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  1.27 s: f = -2.503580279242e-01, ‖∇f‖ = 2.8363e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 895.2 ms: f = -2.570141088033e-01, ‖∇f‖ = 2.5490e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 945.5 ms: f = -2.638770275739e-01, ‖∇f‖ = 3.2847e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 882.8 ms: f = -2.677886281359e-01, ‖∇f‖ = 2.6195e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt  1.04 s: f = -2.692196095650e-01, ‖∇f‖ = 1.0850e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 767.1 ms: f = -2.698032409871e-01, ‖∇f‖ = 9.0920e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt 775.4 ms: f = -2.705488379404e-01, ‖∇f‖ = 7.7031e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt 806.9 ms: f = -2.711089638519e-01, ‖∇f‖ = 5.2491e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, Δt  1.04 s: f = -2.714072269671e-01, ‖∇f‖ = 7.6039e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, Δt 703.6 ms: f = -2.716509819808e-01, ‖∇f‖ = 3.9459e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt 801.5 ms: f = -2.718116455865e-01, ‖∇f‖ = 4.2551e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt 834.1 ms: f = -2.720754359498e-01, ‖∇f‖ = 4.8587e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt  1.09 s: f = -2.723130721138e-01, ‖∇f‖ = 4.7700e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt 665.4 ms: f = -2.724574297064e-01, ‖∇f‖ = 3.4658e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt 762.2 ms: f = -2.725342173431e-01, ‖∇f‖ = 2.0955e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt 843.5 ms: f = -2.725893092543e-01, ‖∇f‖ = 2.4369e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt 828.4 ms: f = -2.726831013274e-01, ‖∇f‖ = 3.1016e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt  1.02 s: f = -2.727104448434e-01, ‖∇f‖ = 4.2233e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt 740.8 ms: f = -2.727640267357e-01, ‖∇f‖ = 1.3811e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt 782.9 ms: f = -2.727826539244e-01, ‖∇f‖ = 1.1632e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt 829.5 ms: f = -2.728090272976e-01, ‖∇f‖ = 1.6703e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt 826.9 ms: f = -2.728603066902e-01, ‖∇f‖ = 2.1066e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt  1.01 s: f = -2.729235900272e-01, ‖∇f‖ = 4.4090e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt 772.5 ms: f = -2.730019247535e-01, ‖∇f‖ = 1.6455e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt 842.9 ms: f = -2.730286814844e-01, ‖∇f‖ = 8.2515e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt  1.07 s: f = -2.730442050194e-01, ‖∇f‖ = 8.5619e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt 685.0 ms: f = -2.730553781406e-01, ‖∇f‖ = 8.7872e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt 732.4 ms: f = -2.730669968210e-01, ‖∇f‖ = 9.3366e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt 826.4 ms: f = -2.730745696985e-01, ‖∇f‖ = 7.8343e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt  1.01 s: f = -2.730795878742e-01, ‖∇f‖ = 6.6005e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt 735.4 ms: f = -2.730854202249e-01, ‖∇f‖ = 6.1388e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt 794.6 ms: f = -2.730961426488e-01, ‖∇f‖ = 9.3577e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt 845.8 ms: f = -2.731086851474e-01, ‖∇f‖ = 1.1157e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt 852.6 ms: f = -2.731188389879e-01, ‖∇f‖ = 6.9895e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt  1.01 s: f = -2.731268414302e-01, ‖∇f‖ = 6.8254e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt 736.3 ms: f = -2.731321728371e-01, ‖∇f‖ = 1.4723e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt 818.4 ms: f = -2.731409219123e-01, ‖∇f‖ = 6.8465e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt 804.9 ms: f = -2.731513436269e-01, ‖∇f‖ = 6.7719e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt 880.1 ms: f = -2.731572421266e-01, ‖∇f‖ = 7.1045e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt  1.62 s: f = -2.731603559271e-01, ‖∇f‖ = 9.4122e-03, α = 4.51e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   53, Δt 798.1 ms: f = -2.731642387647e-01, ‖∇f‖ = 3.6971e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt 800.8 ms: f = -2.731658552954e-01, ‖∇f‖ = 2.9782e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt  1.01 s: f = -2.731676168597e-01, ‖∇f‖ = 3.5594e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, Δt 745.4 ms: f = -2.731699845181e-01, ‖∇f‖ = 3.6931e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt 799.4 ms: f = -2.731736603146e-01, ‖∇f‖ = 8.3114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt 813.8 ms: f = -2.731789832089e-01, ‖∇f‖ = 4.1291e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt 831.2 ms: f = -2.731824564175e-01, ‖∇f‖ = 3.8335e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt  1.05 s: f = -2.731844800291e-01, ‖∇f‖ = 7.2619e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 659.2 ms: f = -2.731863798158e-01, ‖∇f‖ = 3.5730e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt 832.8 ms: f = -2.731873944059e-01, ‖∇f‖ = 2.7923e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt 831.2 ms: f = -2.731914664343e-01, ‖∇f‖ = 4.5101e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt  1.00 s: f = -2.731941482754e-01, ‖∇f‖ = 5.9766e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt 776.3 ms: f = -2.731964718885e-01, ‖∇f‖ = 4.1327e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt 814.6 ms: f = -2.731978019575e-01, ‖∇f‖ = 2.4562e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt 822.2 ms: f = -2.731988575035e-01, ‖∇f‖ = 3.3385e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt  1.01 s: f = -2.732002369828e-01, ‖∇f‖ = 5.2553e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 722.4 ms: f = -2.732026002277e-01, ‖∇f‖ = 6.6141e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt 802.8 ms: f = -2.732040988254e-01, ‖∇f‖ = 7.8696e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt 788.3 ms: f = -2.732064681917e-01, ‖∇f‖ = 2.9546e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt 819.0 ms: f = -2.732073111362e-01, ‖∇f‖ = 1.8645e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt 988.8 ms: f = -2.732081000983e-01, ‖∇f‖ = 2.9502e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 722.9 ms: f = -2.732090584455e-01, ‖∇f‖ = 3.9355e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt 811.4 ms: f = -2.732109629750e-01, ‖∇f‖ = 4.9359e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt  1.72 s: f = -2.732118777907e-01, ‖∇f‖ = 4.2565e-03, α = 3.06e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   77, Δt 756.7 ms: f = -2.732134562653e-01, ‖∇f‖ = 2.4342e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 786.0 ms: f = -2.732148472002e-01, ‖∇f‖ = 2.4432e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt 803.0 ms: f = -2.732162109056e-01, ‖∇f‖ = 3.3614e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, Δt  1.03 s: f = -2.732176222957e-01, ‖∇f‖ = 4.6175e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, Δt 718.1 ms: f = -2.732194828573e-01, ‖∇f‖ = 2.9626e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, Δt 829.5 ms: f = -2.732215731864e-01, ‖∇f‖ = 2.7844e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, Δt  1.06 s: f = -2.732226825451e-01, ‖∇f‖ = 5.5136e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, Δt 679.8 ms: f = -2.732237971780e-01, ‖∇f‖ = 4.0768e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, Δt 703.4 ms: f = -2.732250328864e-01, ‖∇f‖ = 2.2603e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, Δt 791.2 ms: f = -2.732255973793e-01, ‖∇f‖ = 1.5346e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, Δt 803.4 ms: f = -2.732260304594e-01, ‖∇f‖ = 2.0003e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, Δt 963.8 ms: f = -2.732265860219e-01, ‖∇f‖ = 2.6084e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, Δt 760.8 ms: f = -2.732272437461e-01, ‖∇f‖ = 3.8745e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, Δt 760.2 ms: f = -2.732279347862e-01, ‖∇f‖ = 2.1167e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, Δt 805.8 ms: f = -2.732283048214e-01, ‖∇f‖ = 1.2744e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, Δt 829.9 ms: f = -2.732286072434e-01, ‖∇f‖ = 1.7204e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, Δt 988.6 ms: f = -2.732290051320e-01, ‖∇f‖ = 2.3761e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, Δt 757.8 ms: f = -2.732301704429e-01, ‖∇f‖ = 3.3591e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, Δt  1.60 s: f = -2.732308345395e-01, ‖∇f‖ = 5.1322e-03, α = 4.20e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   96, Δt  1.02 s: f = -2.732319356337e-01, ‖∇f‖ = 3.0210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, Δt 763.0 ms: f = -2.732327935083e-01, ‖∇f‖ = 1.1955e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, Δt 831.3 ms: f = -2.732330540990e-01, ‖∇f‖ = 1.6478e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, Δt  1.06 s: f = -2.732335280782e-01, ‖∇f‖ = 2.4169e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, Δt 705.7 ms: f = -2.732343692378e-01, ‖∇f‖ = 4.0871e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, Δt 747.3 ms: f = -2.732354883056e-01, ‖∇f‖ = 2.3987e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, Δt 824.0 ms: f = -2.732363290424e-01, ‖∇f‖ = 1.5636e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, Δt 880.2 ms: f = -2.732365969936e-01, ‖∇f‖ = 2.9324e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, Δt 919.0 ms: f = -2.732369332222e-01, ‖∇f‖ = 1.5259e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, Δt 746.5 ms: f = -2.732371489479e-01, ‖∇f‖ = 1.1928e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, Δt 840.7 ms: f = -2.732376045347e-01, ‖∇f‖ = 1.4928e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, Δt 814.5 ms: f = -2.732380108239e-01, ‖∇f‖ = 1.7220e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, Δt  1.02 s: f = -2.732386077664e-01, ‖∇f‖ = 3.0258e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, Δt 754.1 ms: f = -2.732391970133e-01, ‖∇f‖ = 2.2474e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, Δt 785.0 ms: f = -2.732396073063e-01, ‖∇f‖ = 1.0122e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, Δt 850.8 ms: f = -2.732398859842e-01, ‖∇f‖ = 1.0870e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, Δt 795.8 ms: f = -2.732401560035e-01, ‖∇f‖ = 1.3637e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, Δt  1.03 s: f = -2.732406361089e-01, ‖∇f‖ = 2.4090e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, Δt 764.8 ms: f = -2.732409842326e-01, ‖∇f‖ = 1.6469e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, Δt 807.4 ms: f = -2.732411948013e-01, ‖∇f‖ = 1.0501e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, Δt 802.3 ms: f = -2.732414707437e-01, ‖∇f‖ = 1.0593e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, Δt 816.6 ms: f = -2.732419552943e-01, ‖∇f‖ = 1.8672e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, Δt  1.01 s: f = -2.732428086775e-01, ‖∇f‖ = 2.1763e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, Δt  1.53 s: f = -2.732432161724e-01, ‖∇f‖ = 3.4942e-03, α = 3.20e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  120, Δt 822.7 ms: f = -2.732438697840e-01, ‖∇f‖ = 2.0682e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, Δt  1.03 s: f = -2.732444077519e-01, ‖∇f‖ = 1.1870e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, Δt 721.3 ms: f = -2.732446753641e-01, ‖∇f‖ = 1.5223e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, Δt 791.7 ms: f = -2.732453800526e-01, ‖∇f‖ = 1.9394e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, Δt 816.9 ms: f = -2.732457282343e-01, ‖∇f‖ = 5.7829e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, Δt 840.2 ms: f = -2.732466016515e-01, ‖∇f‖ = 2.3409e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  126, Δt  1.03 s: f = -2.732469585045e-01, ‖∇f‖ = 1.0965e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, Δt 787.8 ms: f = -2.732471502257e-01, ‖∇f‖ = 1.1892e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, Δt 856.8 ms: f = -2.732473532839e-01, ‖∇f‖ = 1.3152e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, Δt  1.04 s: f = -2.732478151159e-01, ‖∇f‖ = 1.4153e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, Δt  1.52 s: f = -2.732480548696e-01, ‖∇f‖ = 1.6400e-03, α = 5.14e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  131, Δt 820.1 ms: f = -2.732483594208e-01, ‖∇f‖ = 9.6044e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, Δt 807.4 ms: f = -2.732485568311e-01, ‖∇f‖ = 1.4584e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, Δt  1.08 s: f = -2.732487619600e-01, ‖∇f‖ = 1.2654e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, Δt 723.1 ms: f = -2.732490369573e-01, ‖∇f‖ = 1.2303e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, Δt 794.8 ms: f = -2.732495393625e-01, ‖∇f‖ = 1.3217e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, Δt  1.77 s: f = -2.732497763246e-01, ‖∇f‖ = 2.1691e-03, α = 3.21e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  137, Δt 723.2 ms: f = -2.732502123636e-01, ‖∇f‖ = 1.7632e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, Δt 762.4 ms: f = -2.732508566749e-01, ‖∇f‖ = 2.0896e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, Δt 810.8 ms: f = -2.732516254658e-01, ‖∇f‖ = 3.9507e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, Δt 775.8 ms: f = -2.732524120528e-01, ‖∇f‖ = 2.3533e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, Δt  1.08 s: f = -2.732528630712e-01, ‖∇f‖ = 1.5257e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, Δt 736.8 ms: f = -2.732529374302e-01, ‖∇f‖ = 2.2820e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, Δt 787.1 ms: f = -2.732530810307e-01, ‖∇f‖ = 8.8629e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, Δt 856.0 ms: f = -2.732531672803e-01, ‖∇f‖ = 7.1131e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, Δt  1.03 s: f = -2.732534205744e-01, ‖∇f‖ = 1.4157e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, Δt 744.0 ms: f = -2.732536577089e-01, ‖∇f‖ = 1.5354e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, Δt 833.3 ms: f = -2.732540654116e-01, ‖∇f‖ = 2.1733e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, Δt 817.3 ms: f = -2.732547211115e-01, ‖∇f‖ = 1.1725e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, Δt 821.5 ms: f = -2.732551254210e-01, ‖∇f‖ = 2.1288e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 14.83 m: f = -2.732555353209e-01, ‖∇f‖ = 1.5243e-03
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/lbfgs.jl:199
E = -0.2732555353208551

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -0.00010740688722198748

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

