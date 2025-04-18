```@meta
EditURL = "../../../../examples/3.bose_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//3.bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//3.bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//3.bose_hubbard)


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
physical_spaces = H.lattice
````

````
1×1 Matrix{GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}}:
 Rep[U₁](0=>1, 1=>1, -1=>1)
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
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514027570125e-11im	err = 3.6943032119e-09	time = 0.60 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 9.360531870693, ‖∇f‖ = 1.6944e+01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time   68.36 s: f = 0.124397324377, ‖∇f‖ = 6.2876e+00, α = 1.56e+02, m = 0, nfg = 7
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time   89.37 s: f = 0.065740243544, ‖∇f‖ = 8.6301e+00, α = 5.34e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time   91.49 s: f = -0.035484016742, ‖∇f‖ = 1.7043e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time   93.49 s: f = -0.068142497162, ‖∇f‖ = 1.5153e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  101.18 s: f = -0.161915006602, ‖∇f‖ = 1.4797e+00, α = 5.52e-01, m = 4, nfg = 3
[ Info: LBFGS: iter    6, time  103.30 s: f = -0.192198031506, ‖∇f‖ = 9.0883e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  105.34 s: f = -0.205025430186, ‖∇f‖ = 1.4077e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  107.32 s: f = -0.221962222995, ‖∇f‖ = 5.1030e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  109.92 s: f = -0.228516635895, ‖∇f‖ = 4.0683e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  111.73 s: f = -0.238976021230, ‖∇f‖ = 2.5966e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  113.50 s: f = -0.245116064461, ‖∇f‖ = 3.7637e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  115.21 s: f = -0.252909139554, ‖∇f‖ = 3.4356e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  116.74 s: f = -0.260342387029, ‖∇f‖ = 2.9482e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  118.22 s: f = -0.265537731651, ‖∇f‖ = 2.5638e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  120.30 s: f = -0.268085955662, ‖∇f‖ = 1.4302e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  121.73 s: f = -0.269213801539, ‖∇f‖ = 9.4430e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  123.01 s: f = -0.270165070558, ‖∇f‖ = 7.6548e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  124.32 s: f = -0.270700725371, ‖∇f‖ = 8.1691e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  125.63 s: f = -0.271027470591, ‖∇f‖ = 4.3095e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  126.92 s: f = -0.271239343525, ‖∇f‖ = 4.0156e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  128.90 s: f = -0.271530179046, ‖∇f‖ = 5.9545e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  130.23 s: f = -0.271855643129, ‖∇f‖ = 5.4653e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  131.51 s: f = -0.272062163616, ‖∇f‖ = 3.1192e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  132.82 s: f = -0.272177598947, ‖∇f‖ = 3.2733e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  134.11 s: f = -0.272296901359, ‖∇f‖ = 3.2681e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  135.40 s: f = -0.272496091725, ‖∇f‖ = 3.3863e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  137.34 s: f = -0.272637326787, ‖∇f‖ = 2.0637e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  138.62 s: f = -0.272670103336, ‖∇f‖ = 2.1720e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  139.91 s: f = -0.272699141285, ‖∇f‖ = 1.5696e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  141.22 s: f = -0.272746811570, ‖∇f‖ = 2.3734e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  142.51 s: f = -0.272818112786, ‖∇f‖ = 3.1753e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  143.81 s: f = -0.272906643893, ‖∇f‖ = 2.5985e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  145.13 s: f = -0.272952554121, ‖∇f‖ = 4.4589e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  147.07 s: f = -0.273005267007, ‖∇f‖ = 9.9076e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  148.35 s: f = -0.273013070971, ‖∇f‖ = 8.2167e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  149.64 s: f = -0.273020751438, ‖∇f‖ = 9.8987e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  150.95 s: f = -0.273028647747, ‖∇f‖ = 1.5048e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  152.25 s: f = -0.273039550238, ‖∇f‖ = 7.4328e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  153.55 s: f = -0.273049748943, ‖∇f‖ = 8.1203e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  154.84 s: f = -0.273057542987, ‖∇f‖ = 1.0556e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  156.78 s: f = -0.273070228574, ‖∇f‖ = 1.2840e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  158.06 s: f = -0.273082077719, ‖∇f‖ = 1.1193e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  159.34 s: f = -0.273092996552, ‖∇f‖ = 8.1332e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  160.64 s: f = -0.273103912308, ‖∇f‖ = 1.2611e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  161.94 s: f = -0.273116383103, ‖∇f‖ = 1.4901e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  163.23 s: f = -0.273124864427, ‖∇f‖ = 2.0568e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  164.53 s: f = -0.273139111302, ‖∇f‖ = 7.0923e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  166.46 s: f = -0.273143969600, ‖∇f‖ = 4.7032e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  167.72 s: f = -0.273146563392, ‖∇f‖ = 4.8587e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  169.01 s: f = -0.273149681682, ‖∇f‖ = 4.4858e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  170.32 s: f = -0.273153535758, ‖∇f‖ = 6.1631e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  171.63 s: f = -0.273156987689, ‖∇f‖ = 4.5642e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  172.93 s: f = -0.273159364143, ‖∇f‖ = 3.9352e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  174.22 s: f = -0.273162378681, ‖∇f‖ = 5.2448e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  176.18 s: f = -0.273164685489, ‖∇f‖ = 7.9447e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  177.46 s: f = -0.273167770852, ‖∇f‖ = 4.5867e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  178.77 s: f = -0.273171004811, ‖∇f‖ = 4.0036e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  180.07 s: f = -0.273172734685, ‖∇f‖ = 5.1587e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  181.38 s: f = -0.273175277923, ‖∇f‖ = 5.2136e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  182.69 s: f = -0.273178436887, ‖∇f‖ = 4.5043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  184.02 s: f = -0.273181599401, ‖∇f‖ = 4.1926e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  186.03 s: f = -0.273184127264, ‖∇f‖ = 3.5435e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  187.31 s: f = -0.273186580584, ‖∇f‖ = 3.2519e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  188.61 s: f = -0.273189436738, ‖∇f‖ = 6.9448e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  189.90 s: f = -0.273192692504, ‖∇f‖ = 4.6825e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  191.20 s: f = -0.273196715211, ‖∇f‖ = 4.7511e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  192.50 s: f = -0.273198196827, ‖∇f‖ = 5.4294e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  193.79 s: f = -0.273199435966, ‖∇f‖ = 2.8645e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  195.71 s: f = -0.273200799932, ‖∇f‖ = 2.4252e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  196.99 s: f = -0.273202184331, ‖∇f‖ = 3.0503e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  198.30 s: f = -0.273203318657, ‖∇f‖ = 7.2710e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  199.59 s: f = -0.273204976120, ‖∇f‖ = 2.8153e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  200.88 s: f = -0.273205613161, ‖∇f‖ = 2.0563e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  202.15 s: f = -0.273206541752, ‖∇f‖ = 2.6401e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  203.45 s: f = -0.273207825994, ‖∇f‖ = 3.3947e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  205.42 s: f = -0.273210784096, ‖∇f‖ = 5.5166e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time  206.71 s: f = -0.273212703606, ‖∇f‖ = 5.9417e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  207.99 s: f = -0.273214840394, ‖∇f‖ = 2.5444e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  209.28 s: f = -0.273216186725, ‖∇f‖ = 2.5210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time  210.57 s: f = -0.273217132240, ‖∇f‖ = 2.9834e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time  211.88 s: f = -0.273217860371, ‖∇f‖ = 8.6736e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time  213.17 s: f = -0.273219978798, ‖∇f‖ = 2.8197e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time  215.10 s: f = -0.273220887198, ‖∇f‖ = 1.4861e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time  216.41 s: f = -0.273221346093, ‖∇f‖ = 1.9427e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time  217.68 s: f = -0.273221865528, ‖∇f‖ = 1.7524e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time  218.95 s: f = -0.273222181741, ‖∇f‖ = 5.3066e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time  220.24 s: f = -0.273223246562, ‖∇f‖ = 1.9685e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time  221.54 s: f = -0.273223705610, ‖∇f‖ = 2.3045e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time  223.47 s: f = -0.273224556828, ‖∇f‖ = 2.5083e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time  224.80 s: f = -0.273225646272, ‖∇f‖ = 4.2989e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time  226.09 s: f = -0.273227256373, ‖∇f‖ = 2.7322e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time  227.40 s: f = -0.273229828316, ‖∇f‖ = 2.2841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time  228.86 s: f = -0.273231704181, ‖∇f‖ = 2.4489e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time  232.19 s: f = -0.273232179586, ‖∇f‖ = 2.5292e-03, α = 3.61e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   95, time  233.47 s: f = -0.273232651417, ‖∇f‖ = 1.6879e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time  234.79 s: f = -0.273233247066, ‖∇f‖ = 1.8610e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time  236.09 s: f = -0.273233724645, ‖∇f‖ = 2.3320e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time  237.39 s: f = -0.273234496326, ‖∇f‖ = 2.1070e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time  238.70 s: f = -0.273234942243, ‖∇f‖ = 4.3229e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time  239.98 s: f = -0.273235872899, ‖∇f‖ = 1.0227e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, time  241.92 s: f = -0.273236083478, ‖∇f‖ = 8.8617e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time  243.21 s: f = -0.273236502150, ‖∇f‖ = 1.2113e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time  245.84 s: f = -0.273236852006, ‖∇f‖ = 2.2250e-03, α = 5.40e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  104, time  247.13 s: f = -0.273237309095, ‖∇f‖ = 1.5651e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time  248.47 s: f = -0.273237826712, ‖∇f‖ = 1.7300e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time  249.76 s: f = -0.273238151589, ‖∇f‖ = 1.4739e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time  251.70 s: f = -0.273238452194, ‖∇f‖ = 1.3132e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, time  253.01 s: f = -0.273239083045, ‖∇f‖ = 3.6339e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time  254.30 s: f = -0.273239925488, ‖∇f‖ = 1.6856e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time  255.60 s: f = -0.273240465522, ‖∇f‖ = 1.3613e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time  256.90 s: f = -0.273240934835, ‖∇f‖ = 2.3524e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time  258.22 s: f = -0.273241253954, ‖∇f‖ = 2.3895e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time  259.55 s: f = -0.273241593627, ‖∇f‖ = 1.3301e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time  261.55 s: f = -0.273242108811, ‖∇f‖ = 1.2847e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time  262.86 s: f = -0.273242449543, ‖∇f‖ = 1.3579e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time  265.50 s: f = -0.273242675076, ‖∇f‖ = 2.3470e-03, α = 3.57e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  117, time  266.81 s: f = -0.273243104527, ‖∇f‖ = 1.2979e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time  268.13 s: f = -0.273243436953, ‖∇f‖ = 1.2233e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time  269.48 s: f = -0.273243761256, ‖∇f‖ = 1.6044e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, time  271.45 s: f = -0.273244062582, ‖∇f‖ = 1.7999e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, time  272.74 s: f = -0.273244428575, ‖∇f‖ = 1.3779e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, time  274.04 s: f = -0.273244941087, ‖∇f‖ = 1.6881e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, time  275.34 s: f = -0.273245490453, ‖∇f‖ = 1.8823e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, time  276.64 s: f = -0.273246014128, ‖∇f‖ = 1.3348e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, time  277.96 s: f = -0.273246415220, ‖∇f‖ = 1.6797e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  126, time  279.27 s: f = -0.273246886558, ‖∇f‖ = 1.9101e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, time  281.26 s: f = -0.273247578331, ‖∇f‖ = 2.6783e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, time  282.56 s: f = -0.273248596780, ‖∇f‖ = 4.2726e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, time  283.87 s: f = -0.273249248580, ‖∇f‖ = 3.4266e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, time  285.18 s: f = -0.273249798264, ‖∇f‖ = 1.3846e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, time  286.50 s: f = -0.273250091726, ‖∇f‖ = 1.3050e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, time  287.81 s: f = -0.273250298144, ‖∇f‖ = 1.7451e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, time  289.13 s: f = -0.273250674484, ‖∇f‖ = 1.8764e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, time  291.11 s: f = -0.273250849981, ‖∇f‖ = 2.5432e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, time  292.38 s: f = -0.273251094386, ‖∇f‖ = 1.0169e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, time  293.67 s: f = -0.273251250424, ‖∇f‖ = 1.1018e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  137, time  294.99 s: f = -0.273251472616, ‖∇f‖ = 1.7175e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, time  296.32 s: f = -0.273251802341, ‖∇f‖ = 1.9190e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, time  298.97 s: f = -0.273251940669, ‖∇f‖ = 2.5923e-03, α = 2.54e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  140, time  300.94 s: f = -0.273252255136, ‖∇f‖ = 1.6162e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, time  302.31 s: f = -0.273252593676, ‖∇f‖ = 9.5803e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, time  303.66 s: f = -0.273252841099, ‖∇f‖ = 1.5506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, time  305.01 s: f = -0.273253128531, ‖∇f‖ = 2.2830e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, time  306.32 s: f = -0.273253470875, ‖∇f‖ = 1.3732e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, time  307.66 s: f = -0.273253787952, ‖∇f‖ = 1.4764e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, time  308.97 s: f = -0.273254080621, ‖∇f‖ = 1.9509e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, time  310.97 s: f = -0.273254306475, ‖∇f‖ = 1.9010e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, time  312.30 s: f = -0.273255277035, ‖∇f‖ = 2.1723e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, time  313.65 s: f = -0.273255976775, ‖∇f‖ = 2.3105e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 314.98 s: f = -0.273256484758, ‖∇f‖ = 1.2498e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.27325648475803166

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -0.00010393272081822697

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

