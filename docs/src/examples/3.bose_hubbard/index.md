```@meta
EditURL = "../../../../examples/3.bose_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//3.bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//3.bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//3.bose_hubbard)

````julia
using Markdown
````

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
boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, maxiter=10, alg=:eigsolver, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=200, ls_maxiter=2, ls_maxfg=2);
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
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514027570125e-11im	err = 3.6943032119e-09	time = 6.77 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg)
@show E;
````

````
[ Info: CTMRG init:	obj = +6.242353178969e-02	err = 1.0000e+00
[ Info: CTMRG conv 4:	obj = +6.242353178968e-02	err = 5.3242429630e-11	time = 0.13 sec
[ Info: LBFGS: initializing with f = 9.360531870693, ‖∇f‖ = 1.6944e+01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: CTMRG init:	obj = +6.240949771847e-02	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +6.241448365552e-02	err = 9.9109245574e-09	time = 0.17 sec
[ Info: CTMRG init:	obj = +6.237870901373e-02	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.250909523927e-02	err = 3.7008104828e-09	time = 0.16 sec
[ Info: CTMRG init:	obj = +6.282962969748e-02 +4.542778772990e-13im	err = 1.0000e+00
[ Info: CTMRG conv 25:	obj = +6.631935894614e-02	err = 6.4811816657e-09	time = 0.26 sec
[ Info: CTMRG init:	obj = +7.333609757066e-02 +7.025628888643e-13im	err = 1.0000e+00
┌ Warning: CTMRG cancel 100:	obj = +1.222846396868e-01	err = 9.4598744433e-06	time = 1.05 sec
└ @ PEPSKit ~/.julia/packages/PEPSKit/P7ER3/src/algorithms/ctmrg/ctmrg.jl:129
[ Info: CTMRG init:	obj = +6.248815924073e-02	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +6.254210320788e-02	err = 5.8592754706e-09	time = 0.14 sec
[ Info: CTMRG init:	obj = +6.245238566875e-02	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +6.246617228050e-02	err = 3.9601332591e-09	time = 0.15 sec
[ Info: CTMRG init:	obj = +7.503564551435e-02	err = 1.0000e+00
┌ Warning: CTMRG cancel 100:	obj = +9.980119603028e-02	err = 1.7005988492e-06	time = 1.06 sec
└ @ PEPSKit ~/.julia/packages/PEPSKit/P7ER3/src/algorithms/ctmrg/ctmrg.jl:129
[ Info: LBFGS: iter    1, time  800.83 s: f = 0.124397324377, ‖∇f‖ = 6.2876e+00, α = 1.56e+02, m = 0, nfg = 7
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: CTMRG init:	obj = +1.753936502020e-01	err = 1.0000e+00
[ Info: CTMRG conv 55:	obj = +1.733894725524e-01	err = 8.1365124077e-09	time = 0.62 sec
[ Info: CTMRG init:	obj = +1.402751961848e-01	err = 1.0000e+00
┌ Warning: CTMRG cancel 100:	obj = +1.402868717464e-01	err = 1.1002578927e-05	time = 1.05 sec
└ @ PEPSKit ~/.julia/packages/PEPSKit/P7ER3/src/algorithms/ctmrg/ctmrg.jl:129
[ Info: LBFGS: iter    2, time  819.10 s: f = 0.065740243544, ‖∇f‖ = 8.6301e+00, α = 5.34e-01, m = 1, nfg = 2
[ Info: CTMRG init:	obj = +1.864268466450e-01	err = 1.0000e+00
[ Info: CTMRG conv 25:	obj = +1.900559755493e-01	err = 4.6098787403e-09	time = 0.26 sec
[ Info: LBFGS: iter    3, time  820.69 s: f = -0.035484016742, ‖∇f‖ = 1.7043e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: CTMRG init:	obj = +1.843525493223e-01	err = 1.0000e+00
[ Info: CTMRG conv 19:	obj = +1.843555438547e-01	err = 6.0174972186e-09	time = 0.17 sec
[ Info: LBFGS: iter    4, time  822.01 s: f = -0.068142497162, ‖∇f‖ = 1.5153e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: CTMRG init:	obj = +1.205660214209e-01	err = 1.0000e+00
[ Info: CTMRG conv 37:	obj = +1.230027024040e-01	err = 8.1345079594e-09	time = 0.42 sec
[ Info: CTMRG init:	obj = +1.777797347746e-01	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +1.777857216166e-01	err = 9.5736429674e-09	time = 0.19 sec
[ Info: CTMRG init:	obj = +1.490804268869e-01	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +1.494241241324e-01	err = 6.2342436828e-09	time = 0.28 sec
[ Info: LBFGS: iter    5, time  826.91 s: f = -0.161915006602, ‖∇f‖ = 1.4797e+00, α = 5.52e-01, m = 4, nfg = 3
[ Info: CTMRG init:	obj = +1.584441491329e-01	err = 1.0000e+00
[ Info: CTMRG conv 19:	obj = +1.585829116178e-01	err = 7.3150666982e-09	time = 0.20 sec
[ Info: LBFGS: iter    6, time  828.30 s: f = -0.192198031506, ‖∇f‖ = 9.0883e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: CTMRG init:	obj = +1.555986931082e-01	err = 1.0000e+00
[ Info: CTMRG conv 21:	obj = +1.559453587913e-01	err = 4.7549366621e-09	time = 0.21 sec
[ Info: LBFGS: iter    7, time  829.59 s: f = -0.205025430186, ‖∇f‖ = 1.4077e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: CTMRG init:	obj = +1.642012255794e-01	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +1.642477281365e-01	err = 8.4090087635e-09	time = 0.20 sec
[ Info: LBFGS: iter    8, time  830.89 s: f = -0.221962222995, ‖∇f‖ = 5.1030e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: CTMRG init:	obj = +1.699305880848e-01	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +1.699638761285e-01	err = 5.2636089744e-09	time = 0.17 sec
[ Info: LBFGS: iter    9, time  832.36 s: f = -0.228516635895, ‖∇f‖ = 4.0683e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: CTMRG init:	obj = +1.850735815256e-01	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +1.851356199709e-01	err = 5.5067543930e-09	time = 0.14 sec
[ Info: LBFGS: iter   10, time  833.39 s: f = -0.238976021230, ‖∇f‖ = 2.5966e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: CTMRG init:	obj = +2.027068304089e-01	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +2.027852287122e-01	err = 2.7944641112e-09	time = 0.14 sec
[ Info: LBFGS: iter   11, time  834.44 s: f = -0.245116064461, ‖∇f‖ = 3.7637e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: CTMRG init:	obj = +2.320245389893e-01	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +2.321898571322e-01	err = 8.2007194775e-09	time = 0.11 sec
[ Info: LBFGS: iter   12, time  835.39 s: f = -0.252909139554, ‖∇f‖ = 3.4356e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: CTMRG init:	obj = +2.717013307419e-01	err = 1.0000e+00
[ Info: CTMRG conv 9:	obj = +2.720038319914e-01	err = 3.1708110696e-09	time = 0.09 sec
[ Info: LBFGS: iter   13, time  836.50 s: f = -0.260342387029, ‖∇f‖ = 2.9482e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: CTMRG init:	obj = +3.142116244338e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.146350793253e-01	err = 5.8852994747e-09	time = 0.07 sec
[ Info: LBFGS: iter   14, time  837.38 s: f = -0.265537731651, ‖∇f‖ = 2.5638e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: CTMRG init:	obj = +3.372181485668e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.374449695056e-01	err = 3.9399420036e-09	time = 0.07 sec
[ Info: LBFGS: iter   15, time  838.21 s: f = -0.268085955662, ‖∇f‖ = 1.4302e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: CTMRG init:	obj = +3.511358819550e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.512358490282e-01	err = 2.8080096982e-09	time = 0.08 sec
[ Info: LBFGS: iter   16, time  839.01 s: f = -0.269213801539, ‖∇f‖ = 9.4430e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: CTMRG init:	obj = +3.607127041006e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.607676677594e-01	err = 1.9984047107e-09	time = 0.08 sec
[ Info: LBFGS: iter   17, time  839.95 s: f = -0.270165070558, ‖∇f‖ = 7.6548e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +3.669365823186e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.669833353692e-01	err = 2.9609976008e-09	time = 0.07 sec
[ Info: LBFGS: iter   18, time  840.67 s: f = -0.270700725371, ‖∇f‖ = 8.1691e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: CTMRG init:	obj = +3.662821626355e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.662785214245e-01	err = 9.5463302733e-10	time = 0.08 sec
[ Info: LBFGS: iter   19, time  841.33 s: f = -0.271027470591, ‖∇f‖ = 4.3095e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: CTMRG init:	obj = +3.652544451924e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +3.652538502298e-01	err = 7.8623197014e-09	time = 0.08 sec
[ Info: LBFGS: iter   20, time  842.03 s: f = -0.271239343525, ‖∇f‖ = 4.0156e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: CTMRG init:	obj = +3.675947033282e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.675962363363e-01	err = 1.5798610291e-09	time = 0.09 sec
[ Info: LBFGS: iter   21, time  843.04 s: f = -0.271530179046, ‖∇f‖ = 5.9545e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.737427181841e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.737631616818e-01	err = 3.3856069217e-09	time = 0.06 sec
[ Info: LBFGS: iter   22, time  843.75 s: f = -0.271855643129, ‖∇f‖ = 5.4653e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.793177955257e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.793309874498e-01	err = 2.3841155460e-09	time = 0.10 sec
[ Info: LBFGS: iter   23, time  844.46 s: f = -0.272062163616, ‖∇f‖ = 3.1192e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.855276802897e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +3.855464418552e-01	err = 2.7835477038e-09	time = 0.09 sec
[ Info: LBFGS: iter   24, time  845.20 s: f = -0.272177598947, ‖∇f‖ = 3.2733e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.884945750065e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +3.884960623902e-01	err = 8.6715117074e-09	time = 0.08 sec
[ Info: LBFGS: iter   25, time  846.20 s: f = -0.272296901359, ‖∇f‖ = 3.2681e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.930849227646e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +3.930864681387e-01	err = 7.2244702258e-09	time = 0.05 sec
[ Info: LBFGS: iter   26, time  846.87 s: f = -0.272496091725, ‖∇f‖ = 3.3863e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.973131595676e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +3.973144781440e-01	err = 8.3712881010e-09	time = 0.08 sec
[ Info: LBFGS: iter   27, time  847.57 s: f = -0.272637326787, ‖∇f‖ = 2.0637e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.986958901805e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +3.986969801782e-01	err = 6.6737409627e-09	time = 0.09 sec
[ Info: LBFGS: iter   28, time  848.29 s: f = -0.272670103336, ‖∇f‖ = 2.1720e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.996067064404e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +3.996067526264e-01	err = 1.5899308014e-09	time = 0.07 sec
[ Info: LBFGS: iter   29, time  849.27 s: f = -0.272699141285, ‖∇f‖ = 1.5696e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.018837916115e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.018834369289e-01	err = 3.4335860175e-09	time = 0.05 sec
[ Info: LBFGS: iter   30, time  849.94 s: f = -0.272746811570, ‖∇f‖ = 2.3734e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.071011385726e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.070990570134e-01	err = 6.5552329812e-09	time = 0.05 sec
[ Info: LBFGS: iter   31, time  850.59 s: f = -0.272818112786, ‖∇f‖ = 3.1753e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.145240753004e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.145178808901e-01	err = 6.8052663533e-09	time = 0.08 sec
[ Info: LBFGS: iter   32, time  851.32 s: f = -0.272906643893, ‖∇f‖ = 2.5985e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.242551245171e-01	err = 1.0000e+00
[ Info: CTMRG conv 8:	obj = +4.242386562514e-01	err = 9.8539759844e-10	time = 0.08 sec
[ Info: LBFGS: iter   33, time  852.05 s: f = -0.272952554121, ‖∇f‖ = 4.4589e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.260999681855e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.260997173475e-01	err = 2.1123831393e-09	time = 0.38 sec
[ Info: LBFGS: iter   34, time  853.10 s: f = -0.273005267007, ‖∇f‖ = 9.9076e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.263395922801e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.263395940764e-01	err = 1.4406662993e-09	time = 0.09 sec
[ Info: LBFGS: iter   35, time  853.77 s: f = -0.273013070971, ‖∇f‖ = 8.2167e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.267340571890e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.267340095510e-01	err = 1.8605278064e-09	time = 0.07 sec
[ Info: LBFGS: iter   36, time  854.49 s: f = -0.273020751438, ‖∇f‖ = 9.8987e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.270278252693e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.270279916970e-01	err = 3.1989087010e-09	time = 0.07 sec
[ Info: LBFGS: iter   37, time  855.21 s: f = -0.273028647747, ‖∇f‖ = 1.5048e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.275457230483e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.275457274368e-01	err = 1.9224715076e-09	time = 0.07 sec
[ Info: LBFGS: iter   38, time  856.18 s: f = -0.273039550238, ‖∇f‖ = 7.4328e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.283093133458e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.283093311740e-01	err = 1.7195411799e-09	time = 0.06 sec
[ Info: LBFGS: iter   39, time  856.88 s: f = -0.273049748943, ‖∇f‖ = 8.1203e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.293602909349e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.293603320901e-01	err = 1.3448015618e-09	time = 0.08 sec
[ Info: LBFGS: iter   40, time  857.57 s: f = -0.273057542987, ‖∇f‖ = 1.0556e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.315340639040e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.315343813704e-01	err = 2.7317716643e-09	time = 0.08 sec
[ Info: LBFGS: iter   41, time  858.34 s: f = -0.273070228574, ‖∇f‖ = 1.2840e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.356033735656e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.356056142543e-01	err = 6.4345317970e-09	time = 0.07 sec
[ Info: LBFGS: iter   42, time  859.34 s: f = -0.273082077719, ‖∇f‖ = 1.1193e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.361087228120e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.361086699054e-01	err = 1.1521036128e-09	time = 0.06 sec
[ Info: LBFGS: iter   43, time  860.05 s: f = -0.273092996552, ‖∇f‖ = 8.1332e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.369765582804e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.369766520549e-01	err = 2.6684951030e-09	time = 0.08 sec
[ Info: LBFGS: iter   44, time  860.78 s: f = -0.273103912308, ‖∇f‖ = 1.2611e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.387686835133e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.387695437311e-01	err = 4.9139230079e-09	time = 0.08 sec
[ Info: LBFGS: iter   45, time  861.52 s: f = -0.273116383103, ‖∇f‖ = 1.4901e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.426689967922e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.426741768214e-01	err = 9.6216823102e-09	time = 0.09 sec
[ Info: LBFGS: iter   46, time  862.31 s: f = -0.273124864427, ‖∇f‖ = 2.0568e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.435228513815e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.435230477001e-01	err = 1.3938375485e-09	time = 0.32 sec
[ Info: LBFGS: iter   47, time  863.27 s: f = -0.273139111302, ‖∇f‖ = 7.0923e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.442359332390e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.442360624121e-01	err = 8.2102807253e-09	time = 0.05 sec
[ Info: LBFGS: iter   48, time  863.93 s: f = -0.273143969600, ‖∇f‖ = 4.7032e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.450671762927e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.450673528731e-01	err = 9.4782302337e-09	time = 0.06 sec
[ Info: LBFGS: iter   49, time  864.68 s: f = -0.273146563392, ‖∇f‖ = 4.8587e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.461659228579e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.461662107224e-01	err = 1.4651344945e-09	time = 0.08 sec
[ Info: LBFGS: iter   50, time  865.66 s: f = -0.273149681682, ‖∇f‖ = 4.4858e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.478197469584e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.478201251117e-01	err = 1.4263848945e-09	time = 0.06 sec
[ Info: LBFGS: iter   51, time  866.36 s: f = -0.273153535758, ‖∇f‖ = 6.1631e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.499444375418e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.499449995341e-01	err = 1.7222819789e-09	time = 0.08 sec
[ Info: LBFGS: iter   52, time  867.07 s: f = -0.273156987689, ‖∇f‖ = 4.5642e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.504547041087e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.504547220327e-01	err = 7.7826823112e-09	time = 0.07 sec
[ Info: LBFGS: iter   53, time  867.79 s: f = -0.273159364143, ‖∇f‖ = 3.9352e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.511021911515e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.511022771394e-01	err = 7.4953726522e-10	time = 0.09 sec
[ Info: LBFGS: iter   54, time  868.57 s: f = -0.273162378681, ‖∇f‖ = 5.2448e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.520921936956e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.520921967728e-01	err = 5.3039284811e-10	time = 0.31 sec
[ Info: LBFGS: iter   55, time  869.53 s: f = -0.273164685489, ‖∇f‖ = 7.9447e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.530357875303e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.530358010870e-01	err = 8.7429488510e-09	time = 0.05 sec
[ Info: LBFGS: iter   56, time  870.18 s: f = -0.273167770852, ‖∇f‖ = 4.5867e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.545882887291e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.545884315469e-01	err = 1.6054261815e-09	time = 0.09 sec
[ Info: LBFGS: iter   57, time  870.92 s: f = -0.273171004811, ‖∇f‖ = 4.0036e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.556787931045e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.556789973237e-01	err = 1.4934621828e-09	time = 0.09 sec
[ Info: LBFGS: iter   58, time  871.92 s: f = -0.273172734685, ‖∇f‖ = 5.1587e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.572978492667e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.572983075304e-01	err = 2.1649745710e-09	time = 0.06 sec
[ Info: LBFGS: iter   59, time  872.61 s: f = -0.273175277923, ‖∇f‖ = 5.2136e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.592789057454e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.592795996463e-01	err = 2.4538482534e-09	time = 0.08 sec
[ Info: LBFGS: iter   60, time  873.36 s: f = -0.273178436887, ‖∇f‖ = 4.5043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.610203529605e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.610209113341e-01	err = 1.9923210396e-09	time = 0.07 sec
[ Info: LBFGS: iter   61, time  874.08 s: f = -0.273181599401, ‖∇f‖ = 4.1926e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.616536977817e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.616537604132e-01	err = 6.4830178495e-09	time = 0.07 sec
[ Info: LBFGS: iter   62, time  875.13 s: f = -0.273184127264, ‖∇f‖ = 3.5435e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.615991308044e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.615992366511e-01	err = 5.0404638781e-10	time = 0.06 sec
[ Info: LBFGS: iter   63, time  875.81 s: f = -0.273186580584, ‖∇f‖ = 3.2519e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.622036014559e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.622036743292e-01	err = 8.0180819642e-10	time = 0.06 sec
[ Info: LBFGS: iter   64, time  876.50 s: f = -0.273189436738, ‖∇f‖ = 6.9448e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.625455998697e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.625456196865e-01	err = 8.9687844711e-09	time = 0.07 sec
[ Info: LBFGS: iter   65, time  877.22 s: f = -0.273192692504, ‖∇f‖ = 4.6825e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.645067676136e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.645069109392e-01	err = 1.9375918118e-09	time = 0.09 sec
[ Info: LBFGS: iter   66, time  877.96 s: f = -0.273196715211, ‖∇f‖ = 4.7511e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.657748055462e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.657748482350e-01	err = 1.2394947561e-09	time = 0.37 sec
[ Info: LBFGS: iter   67, time  878.91 s: f = -0.273198196827, ‖∇f‖ = 5.4294e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.665064064468e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.665065369184e-01	err = 6.4068919881e-09	time = 0.07 sec
[ Info: LBFGS: iter   68, time  879.57 s: f = -0.273199435966, ‖∇f‖ = 2.8645e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.674195111640e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.674196890444e-01	err = 7.6716513054e-09	time = 0.06 sec
[ Info: LBFGS: iter   69, time  880.24 s: f = -0.273200799932, ‖∇f‖ = 2.4252e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.683160818296e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.683162051195e-01	err = 7.4930079200e-09	time = 0.08 sec
[ Info: LBFGS: iter   70, time  880.94 s: f = -0.273202184331, ‖∇f‖ = 3.0503e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.690638581817e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.690637699300e-01	err = 9.6440009281e-10	time = 0.07 sec
[ Info: LBFGS: iter   71, time  881.90 s: f = -0.273203318657, ‖∇f‖ = 7.2710e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.694434155331e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.694433572079e-01	err = 6.3638742883e-09	time = 0.05 sec
[ Info: LBFGS: iter   72, time  882.59 s: f = -0.273204976120, ‖∇f‖ = 2.8153e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.690025332347e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.690026028015e-01	err = 4.5405781107e-09	time = 0.07 sec
[ Info: LBFGS: iter   73, time  883.29 s: f = -0.273205613161, ‖∇f‖ = 2.0563e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.683640409461e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.683642070895e-01	err = 7.5729673107e-09	time = 0.06 sec
[ Info: LBFGS: iter   74, time  883.98 s: f = -0.273206541752, ‖∇f‖ = 2.6401e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.680812736007e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.680812878329e-01	err = 5.8296871295e-09	time = 0.06 sec
[ Info: LBFGS: iter   75, time  884.93 s: f = -0.273207825994, ‖∇f‖ = 3.3947e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.683658632583e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.683654640805e-01	err = 9.6410675213e-10	time = 0.06 sec
[ Info: LBFGS: iter   76, time  885.62 s: f = -0.273210784096, ‖∇f‖ = 5.5166e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.695632207222e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.695628536799e-01	err = 1.4863989272e-09	time = 0.07 sec
[ Info: LBFGS: iter   77, time  886.29 s: f = -0.273212703606, ‖∇f‖ = 5.9417e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.704381714894e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.704383900529e-01	err = 8.0609213295e-09	time = 0.06 sec
[ Info: LBFGS: iter   78, time  886.98 s: f = -0.273214840394, ‖∇f‖ = 2.5444e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.716629768178e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.716633806818e-01	err = 9.3022446537e-09	time = 0.06 sec
[ Info: LBFGS: iter   79, time  887.93 s: f = -0.273216186725, ‖∇f‖ = 2.5210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.726362728301e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.726365646547e-01	err = 7.0379215850e-09	time = 0.05 sec
[ Info: LBFGS: iter   80, time  888.61 s: f = -0.273217132240, ‖∇f‖ = 2.9834e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.746270632342e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.746275390474e-01	err = 1.2462578943e-09	time = 0.07 sec
[ Info: LBFGS: iter   81, time  889.28 s: f = -0.273217860371, ‖∇f‖ = 8.6736e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.742606682887e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.742606864218e-01	err = 3.5846387007e-09	time = 0.06 sec
[ Info: LBFGS: iter   82, time  889.96 s: f = -0.273219978798, ‖∇f‖ = 2.8197e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.739168218451e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.739168525084e-01	err = 6.3378884097e-09	time = 0.06 sec
[ Info: LBFGS: iter   83, time  890.68 s: f = -0.273220887198, ‖∇f‖ = 1.4861e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.736949927381e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.736950134585e-01	err = 5.3980945320e-09	time = 0.05 sec
[ Info: LBFGS: iter   84, time  891.56 s: f = -0.273221346093, ‖∇f‖ = 1.9427e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.737094141910e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.737094093508e-01	err = 4.1427533982e-09	time = 0.05 sec
[ Info: LBFGS: iter   85, time  892.28 s: f = -0.273221865528, ‖∇f‖ = 1.7524e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.739498377882e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.739497241759e-01	err = 8.3094272741e-09	time = 0.06 sec
[ Info: LBFGS: iter   86, time  892.99 s: f = -0.273222181741, ‖∇f‖ = 5.3066e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.742275640120e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.742275861455e-01	err = 2.1070029154e-09	time = 0.07 sec
[ Info: LBFGS: iter   87, time  893.98 s: f = -0.273223246562, ‖∇f‖ = 1.9685e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.745615531112e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.745615970216e-01	err = 2.5255250793e-09	time = 0.05 sec
[ Info: LBFGS: iter   88, time  894.67 s: f = -0.273223705610, ‖∇f‖ = 2.3045e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.753529638533e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.753531654518e-01	err = 5.7678184307e-09	time = 0.07 sec
[ Info: LBFGS: iter   89, time  895.38 s: f = -0.273224556828, ‖∇f‖ = 2.5083e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.765772035582e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.765776837597e-01	err = 1.0124171631e-09	time = 0.08 sec
[ Info: LBFGS: iter   90, time  896.12 s: f = -0.273225646272, ‖∇f‖ = 4.2989e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.777304198473e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.777307237114e-01	err = 1.0180024105e-09	time = 0.08 sec
[ Info: LBFGS: iter   91, time  897.15 s: f = -0.273227256373, ‖∇f‖ = 2.7322e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.783240501999e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.783240673497e-01	err = 7.1416725880e-10	time = 0.06 sec
[ Info: LBFGS: iter   92, time  897.82 s: f = -0.273229828316, ‖∇f‖ = 2.2841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.791322881259e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.791323312929e-01	err = 1.0389736817e-09	time = 0.09 sec
[ Info: LBFGS: iter   93, time  898.53 s: f = -0.273231704181, ‖∇f‖ = 2.4489e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.786596533113e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.786601089445e-01	err = 5.9830244732e-10	time = 0.09 sec
[ Info: CTMRG init:	obj = +4.789662294135e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.789662889092e-01	err = 6.4853481779e-09	time = 0.07 sec
[ Info: LBFGS: iter   94, time  900.03 s: f = -0.273232179586, ‖∇f‖ = 2.5292e-03, α = 3.61e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +4.788074933549e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.788075351219e-01	err = 2.2631300753e-09	time = 0.07 sec
[ Info: LBFGS: iter   95, time  901.02 s: f = -0.273232651417, ‖∇f‖ = 1.6879e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.786080247497e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.786081498864e-01	err = 3.2395930512e-09	time = 0.05 sec
[ Info: LBFGS: iter   96, time  901.75 s: f = -0.273233247066, ‖∇f‖ = 1.8610e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.786842676431e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.786842578665e-01	err = 1.1277031841e-09	time = 0.06 sec
[ Info: LBFGS: iter   97, time  902.47 s: f = -0.273233724645, ‖∇f‖ = 2.3320e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.792106070878e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.792104764342e-01	err = 4.9994793879e-09	time = 0.06 sec
[ Info: LBFGS: iter   98, time  903.15 s: f = -0.273234496326, ‖∇f‖ = 2.1070e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.792807644559e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.792804289320e-01	err = 5.7368410925e-10	time = 0.07 sec
[ Info: LBFGS: iter   99, time  903.85 s: f = -0.273234942243, ‖∇f‖ = 4.3229e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.797988803658e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.797989152104e-01	err = 4.5762436281e-09	time = 0.07 sec
[ Info: LBFGS: iter  100, time  904.79 s: f = -0.273235872899, ‖∇f‖ = 1.0227e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.797019933537e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.797019858679e-01	err = 9.2179000290e-10	time = 0.10 sec
[ Info: LBFGS: iter  101, time  905.50 s: f = -0.273236083478, ‖∇f‖ = 8.8617e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.793891521133e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.793891740523e-01	err = 5.6337023713e-09	time = 0.06 sec
[ Info: LBFGS: iter  102, time  906.21 s: f = -0.273236502150, ‖∇f‖ = 1.2113e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.789744279460e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.789745607107e-01	err = 3.7058752456e-10	time = 0.09 sec
[ Info: CTMRG init:	obj = +4.791677254254e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.791677641063e-01	err = 7.3689893544e-09	time = 0.07 sec
[ Info: LBFGS: iter  103, time  907.70 s: f = -0.273236852006, ‖∇f‖ = 2.2250e-03, α = 5.40e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +4.789276815767e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.789277231124e-01	err = 7.5306084214e-09	time = 0.07 sec
[ Info: LBFGS: iter  104, time  908.67 s: f = -0.273237309095, ‖∇f‖ = 1.5651e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.788843602719e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.788843637881e-01	err = 2.6405239504e-10	time = 0.11 sec
[ Info: LBFGS: iter  105, time  909.39 s: f = -0.273237826712, ‖∇f‖ = 1.7300e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.789798585945e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.789798340955e-01	err = 4.5254772597e-09	time = 0.08 sec
[ Info: LBFGS: iter  106, time  910.12 s: f = -0.273238151589, ‖∇f‖ = 1.4739e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.792964889500e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.792965031016e-01	err = 3.0861610133e-09	time = 0.07 sec
[ Info: LBFGS: iter  107, time  910.85 s: f = -0.273238452194, ‖∇f‖ = 1.3132e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.797585448501e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.797583969194e-01	err = 5.5138899038e-10	time = 0.09 sec
[ Info: LBFGS: iter  108, time  911.83 s: f = -0.273239083045, ‖∇f‖ = 3.6339e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.801830137612e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.801830200283e-01	err = 4.9330311249e-09	time = 0.05 sec
[ Info: LBFGS: iter  109, time  912.50 s: f = -0.273239925488, ‖∇f‖ = 1.6856e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.802634867190e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.802634639939e-01	err = 5.1863757945e-09	time = 0.08 sec
[ Info: LBFGS: iter  110, time  913.22 s: f = -0.273240465522, ‖∇f‖ = 1.3613e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.802760262834e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.802760076695e-01	err = 4.9382603595e-09	time = 0.07 sec
[ Info: LBFGS: iter  111, time  913.97 s: f = -0.273240934835, ‖∇f‖ = 2.3524e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.802939956932e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.802939410658e-01	err = 8.9304889036e-09	time = 0.06 sec
[ Info: LBFGS: iter  112, time  914.64 s: f = -0.273241253954, ‖∇f‖ = 2.3895e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.802973441785e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.802973435724e-01	err = 2.4216364455e-09	time = 0.07 sec
[ Info: LBFGS: iter  113, time  915.58 s: f = -0.273241593627, ‖∇f‖ = 1.3301e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.804014797027e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.804014788612e-01	err = 1.8737547985e-09	time = 0.09 sec
[ Info: LBFGS: iter  114, time  916.26 s: f = -0.273242108811, ‖∇f‖ = 1.2847e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805466102721e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.805466013995e-01	err = 2.4328755928e-09	time = 0.06 sec
[ Info: LBFGS: iter  115, time  916.92 s: f = -0.273242449543, ‖∇f‖ = 1.3579e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.803090646228e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.803090968139e-01	err = 5.2955340458e-10	time = 0.08 sec
[ Info: CTMRG init:	obj = +4.804639686484e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.804639727315e-01	err = 5.4757355659e-09	time = 0.06 sec
[ Info: LBFGS: iter  116, time  918.28 s: f = -0.273242675076, ‖∇f‖ = 2.3470e-03, α = 3.57e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +4.805754019387e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.805753859421e-01	err = 7.4466348840e-09	time = 0.07 sec
[ Info: LBFGS: iter  117, time  919.21 s: f = -0.273243104527, ‖∇f‖ = 1.2979e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805735505394e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.805735313982e-01	err = 4.2148915555e-10	time = 0.11 sec
[ Info: LBFGS: iter  118, time  919.93 s: f = -0.273243436953, ‖∇f‖ = 1.2233e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.804710505169e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.804710342447e-01	err = 6.4482932714e-10	time = 0.08 sec
[ Info: LBFGS: iter  119, time  920.72 s: f = -0.273243761256, ‖∇f‖ = 1.6044e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805019941032e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.805019970087e-01	err = 5.7035908358e-09	time = 0.06 sec
[ Info: LBFGS: iter  120, time  921.44 s: f = -0.273244062582, ‖∇f‖ = 1.7999e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805364210841e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.805364298716e-01	err = 5.4422519760e-09	time = 0.07 sec
[ Info: LBFGS: iter  121, time  922.37 s: f = -0.273244428575, ‖∇f‖ = 1.3779e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.807559862614e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.807560378139e-01	err = 3.7379046929e-09	time = 0.05 sec
[ Info: LBFGS: iter  122, time  923.04 s: f = -0.273244941087, ‖∇f‖ = 1.6881e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.810722414880e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.810722899338e-01	err = 7.1509377999e-09	time = 0.06 sec
[ Info: LBFGS: iter  123, time  923.70 s: f = -0.273245490453, ‖∇f‖ = 1.8823e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.814299105464e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.814299327851e-01	err = 7.1850148357e-09	time = 0.06 sec
[ Info: LBFGS: iter  124, time  924.47 s: f = -0.273246014128, ‖∇f‖ = 1.3348e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.816345761873e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.816345765078e-01	err = 5.1886510131e-09	time = 0.08 sec
[ Info: LBFGS: iter  125, time  925.47 s: f = -0.273246415220, ‖∇f‖ = 1.6797e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.817457892273e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.817457676277e-01	err = 7.9942843277e-09	time = 0.05 sec
[ Info: LBFGS: iter  126, time  926.18 s: f = -0.273246886558, ‖∇f‖ = 1.9101e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.816486297860e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.816486220771e-01	err = 5.8526396306e-10	time = 0.07 sec
[ Info: LBFGS: iter  127, time  926.91 s: f = -0.273247578331, ‖∇f‖ = 2.6783e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.813734078208e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.813733741400e-01	err = 2.1140612736e-09	time = 0.08 sec
[ Info: LBFGS: iter  128, time  927.65 s: f = -0.273248596780, ‖∇f‖ = 4.2726e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.809326459337e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.809326329788e-01	err = 3.3469639206e-09	time = 0.07 sec
[ Info: LBFGS: iter  129, time  928.44 s: f = -0.273249248580, ‖∇f‖ = 3.4266e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.809941074596e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.809940893817e-01	err = 9.1754566530e-09	time = 0.29 sec
[ Info: LBFGS: iter  130, time  929.37 s: f = -0.273249798264, ‖∇f‖ = 1.3846e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.810122860248e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.810123085036e-01	err = 7.9488237170e-10	time = 0.09 sec
[ Info: LBFGS: iter  131, time  930.09 s: f = -0.273250091726, ‖∇f‖ = 1.3050e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.810105154042e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.810105224177e-01	err = 7.4451601923e-10	time = 0.09 sec
[ Info: LBFGS: iter  132, time  930.86 s: f = -0.273250298144, ‖∇f‖ = 1.7451e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.809813774018e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.809814217119e-01	err = 1.7143550686e-09	time = 0.09 sec
[ Info: LBFGS: iter  133, time  931.61 s: f = -0.273250674484, ‖∇f‖ = 1.8764e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.809589590340e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.809590192078e-01	err = 1.8391504148e-09	time = 0.08 sec
[ Info: LBFGS: iter  134, time  932.61 s: f = -0.273250849981, ‖∇f‖ = 2.5432e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.808955403058e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.808955449714e-01	err = 2.2951085914e-09	time = 0.05 sec
[ Info: LBFGS: iter  135, time  933.31 s: f = -0.273251094386, ‖∇f‖ = 1.0169e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.808431471175e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.808431483697e-01	err = 6.0541470008e-09	time = 0.07 sec
[ Info: LBFGS: iter  136, time  934.01 s: f = -0.273251250424, ‖∇f‖ = 1.1018e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.807421686570e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.807421786855e-01	err = 3.8423948125e-10	time = 0.08 sec
[ Info: LBFGS: iter  137, time  934.75 s: f = -0.273251472616, ‖∇f‖ = 1.7175e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805501691108e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.805501937811e-01	err = 1.2032452024e-09	time = 0.09 sec
[ Info: LBFGS: iter  138, time  935.51 s: f = -0.273251802341, ‖∇f‖ = 1.9190e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.804289072221e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.804289531354e-01	err = 1.2177451389e-09	time = 0.08 sec
[ Info: CTMRG init:	obj = +4.805208730891e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.805208760207e-01	err = 3.1793481864e-10	time = 0.11 sec
[ Info: LBFGS: iter  139, time  937.24 s: f = -0.273251940669, ‖∇f‖ = 2.5923e-03, α = 2.54e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +4.803946272888e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.803946461498e-01	err = 1.2157672579e-09	time = 0.07 sec
[ Info: LBFGS: iter  140, time  937.97 s: f = -0.273252255136, ‖∇f‖ = 1.6162e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.803241381625e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.803241574618e-01	err = 1.1802934031e-09	time = 0.09 sec
[ Info: LBFGS: iter  141, time  938.74 s: f = -0.273252593676, ‖∇f‖ = 9.5803e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.803546161524e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.803546361914e-01	err = 8.7972431541e-10	time = 0.08 sec
[ Info: LBFGS: iter  142, time  939.48 s: f = -0.273252841099, ‖∇f‖ = 1.5506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.804182585835e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.804182806506e-01	err = 8.3811595962e-10	time = 0.08 sec
[ Info: LBFGS: iter  143, time  940.47 s: f = -0.273253128531, ‖∇f‖ = 2.2830e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805261995241e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.805262097495e-01	err = 4.7700114831e-09	time = 0.10 sec
[ Info: LBFGS: iter  144, time  941.19 s: f = -0.273253470875, ‖∇f‖ = 1.3732e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805616712670e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.805616822842e-01	err = 1.2044520669e-09	time = 0.07 sec
[ Info: LBFGS: iter  145, time  941.97 s: f = -0.273253787952, ‖∇f‖ = 1.4764e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.806272354494e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.806272279261e-01	err = 1.9225106931e-09	time = 0.06 sec
[ Info: LBFGS: iter  146, time  942.69 s: f = -0.273254080621, ‖∇f‖ = 1.9509e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.805717973510e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.805717949586e-01	err = 4.3409144020e-10	time = 0.08 sec
[ Info: LBFGS: iter  147, time  943.42 s: f = -0.273254306475, ‖∇f‖ = 1.9010e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.801673305193e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.801673335911e-01	err = 3.7575359203e-09	time = 0.36 sec
[ Info: LBFGS: iter  148, time  944.41 s: f = -0.273255277035, ‖∇f‖ = 2.1723e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.800445144046e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.800445745424e-01	err = 3.0499085824e-09	time = 0.08 sec
[ Info: LBFGS: iter  149, time  945.12 s: f = -0.273255976775, ‖∇f‖ = 2.3105e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.800359660106e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.800360016141e-01	err = 1.1855876964e-09	time = 0.08 sec
[ Info: LBFGS: iter  150, time  945.86 s: f = -0.273256484758, ‖∇f‖ = 1.2498e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.800064707408e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.800066140113e-01	err = 2.0245498571e-09	time = 0.08 sec
[ Info: LBFGS: iter  151, time  946.63 s: f = -0.273256936415, ‖∇f‖ = 1.2435e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.799563181331e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.799563788753e-01	err = 1.1719463343e-09	time = 0.07 sec
[ Info: LBFGS: iter  152, time  947.63 s: f = -0.273257194268, ‖∇f‖ = 1.5879e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.798930407799e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.798931268224e-01	err = 1.1396026674e-09	time = 0.06 sec
[ Info: LBFGS: iter  153, time  948.38 s: f = -0.273257492644, ‖∇f‖ = 1.2019e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.797802516360e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.797802819870e-01	err = 8.0552494762e-10	time = 0.07 sec
[ Info: LBFGS: iter  154, time  949.11 s: f = -0.273257706633, ‖∇f‖ = 1.2598e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.797768701253e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.797768714427e-01	err = 3.3382578990e-09	time = 0.07 sec
[ Info: LBFGS: iter  155, time  949.86 s: f = -0.273257863534, ‖∇f‖ = 1.2053e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.796659993069e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.796660368515e-01	err = 9.1008880966e-10	time = 0.07 sec
[ Info: LBFGS: iter  156, time  950.61 s: f = -0.273258179903, ‖∇f‖ = 1.2266e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.795411638633e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.795411850459e-01	err = 1.2512833434e-09	time = 0.08 sec
[ Info: LBFGS: iter  157, time  951.59 s: f = -0.273258423796, ‖∇f‖ = 1.9907e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.795802685592e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.795802885861e-01	err = 8.2245725331e-09	time = 0.10 sec
[ Info: LBFGS: iter  158, time  952.30 s: f = -0.273258732899, ‖∇f‖ = 1.0784e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.792877111859e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.792879760255e-01	err = 4.0009118433e-09	time = 0.07 sec
[ Info: LBFGS: iter  159, time  953.04 s: f = -0.273259186735, ‖∇f‖ = 1.6223e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.791975875972e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.791976321664e-01	err = 1.6716860412e-09	time = 0.07 sec
[ Info: LBFGS: iter  160, time  953.76 s: f = -0.273259592363, ‖∇f‖ = 1.6598e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.789064571829e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.789065644045e-01	err = 3.5932050759e-09	time = 0.09 sec
[ Info: LBFGS: iter  161, time  954.53 s: f = -0.273259817595, ‖∇f‖ = 3.7859e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.786811932101e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.786812583197e-01	err = 2.0205940886e-09	time = 0.09 sec
[ Info: LBFGS: iter  162, time  955.51 s: f = -0.273260282927, ‖∇f‖ = 9.1253e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.788559393609e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.788560073523e-01	err = 2.0974758634e-09	time = 0.10 sec
[ Info: LBFGS: iter  163, time  956.22 s: f = -0.273260379913, ‖∇f‖ = 7.4528e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.788376643167e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.788376636156e-01	err = 4.3116122365e-09	time = 0.06 sec
[ Info: LBFGS: iter  164, time  956.93 s: f = -0.273260551977, ‖∇f‖ = 6.2372e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.785975525078e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.785975539109e-01	err = 1.4094402103e-09	time = 0.07 sec
[ Info: LBFGS: iter  165, time  957.68 s: f = -0.273260703522, ‖∇f‖ = 2.8247e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.785447283809e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.785447317097e-01	err = 6.4005366060e-09	time = 0.06 sec
[ Info: LBFGS: iter  166, time  958.66 s: f = -0.273260965983, ‖∇f‖ = 7.7435e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.785616857991e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.785616855805e-01	err = 2.4400266361e-09	time = 0.05 sec
[ Info: LBFGS: iter  167, time  959.34 s: f = -0.273261029226, ‖∇f‖ = 6.0943e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.784706282454e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.784706362702e-01	err = 1.0967731972e-09	time = 0.08 sec
[ Info: LBFGS: iter  168, time  960.06 s: f = -0.273261217327, ‖∇f‖ = 6.3523e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.782868347547e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.782868052432e-01	err = 1.6148867721e-09	time = 0.09 sec
[ Info: LBFGS: iter  169, time  960.83 s: f = -0.273261413376, ‖∇f‖ = 2.2349e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.780984755793e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.780985167087e-01	err = 2.2970561742e-09	time = 0.07 sec
[ Info: LBFGS: iter  170, time  961.59 s: f = -0.273261694992, ‖∇f‖ = 8.1484e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.780266153849e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.780266154518e-01	err = 8.1891024560e-09	time = 0.07 sec
[ Info: LBFGS: iter  171, time  962.57 s: f = -0.273261835395, ‖∇f‖ = 7.1536e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.779268653137e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.779268609535e-01	err = 9.9986450086e-10	time = 0.10 sec
[ Info: LBFGS: iter  172, time  963.29 s: f = -0.273262028596, ‖∇f‖ = 1.0380e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.777985422147e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.777985306143e-01	err = 8.7770193229e-10	time = 0.07 sec
[ Info: LBFGS: iter  173, time  964.03 s: f = -0.273262181630, ‖∇f‖ = 1.1178e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.777370606901e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.777370633742e-01	err = 1.4143631977e-09	time = 0.08 sec
[ Info: LBFGS: iter  174, time  964.79 s: f = -0.273262422962, ‖∇f‖ = 1.0462e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.775641028873e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.775641014404e-01	err = 3.1561784283e-09	time = 0.07 sec
[ Info: LBFGS: iter  175, time  965.50 s: f = -0.273262837415, ‖∇f‖ = 1.1822e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.774748517265e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.774748352105e-01	err = 2.4478019594e-09	time = 0.07 sec
[ Info: LBFGS: iter  176, time  966.45 s: f = -0.273263042769, ‖∇f‖ = 2.2133e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.773219350857e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.773219260487e-01	err = 1.6189317245e-09	time = 0.11 sec
[ Info: LBFGS: iter  177, time  967.17 s: f = -0.273263279588, ‖∇f‖ = 8.7624e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.772854043848e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.772854020301e-01	err = 2.9822673611e-09	time = 0.07 sec
[ Info: LBFGS: iter  178, time  967.90 s: f = -0.273263449265, ‖∇f‖ = 1.0564e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.771758028625e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.771757836120e-01	err = 8.8142743730e-10	time = 0.09 sec
[ Info: LBFGS: iter  179, time  968.65 s: f = -0.273263776392, ‖∇f‖ = 1.6950e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.771302116662e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.771301874760e-01	err = 5.2541798786e-09	time = 0.06 sec
[ Info: LBFGS: iter  180, time  969.40 s: f = -0.273264059850, ‖∇f‖ = 1.5481e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.767996014124e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.767995959424e-01	err = 5.5848027818e-09	time = 0.07 sec
[ Info: CTMRG init:	obj = +4.770208398690e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.770208392566e-01	err = 1.6681094467e-09	time = 0.10 sec
[ Info: LBFGS: iter  181, time  971.10 s: f = -0.273264187559, ‖∇f‖ = 1.2988e-03, α = 3.40e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +4.770440823632e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.770440797053e-01	err = 1.5739787613e-09	time = 0.06 sec
[ Info: LBFGS: iter  182, time  971.83 s: f = -0.273264310040, ‖∇f‖ = 5.0027e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.770117929246e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.770117950770e-01	err = 8.9575249608e-10	time = 0.07 sec
[ Info: LBFGS: iter  183, time  972.55 s: f = -0.273264350660, ‖∇f‖ = 5.8334e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769971605641e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.769971621318e-01	err = 7.7748930713e-10	time = 0.07 sec
[ Info: LBFGS: iter  184, time  973.26 s: f = -0.273264392156, ‖∇f‖ = 7.4958e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769083171394e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.769083261527e-01	err = 2.2158423859e-09	time = 0.07 sec
[ Info: LBFGS: iter  185, time  974.24 s: f = -0.273264466960, ‖∇f‖ = 1.5690e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.768678353976e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.768678382372e-01	err = 1.4134441577e-09	time = 0.11 sec
[ Info: LBFGS: iter  186, time  974.91 s: f = -0.273264575915, ‖∇f‖ = 8.7235e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.767879001421e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.767878991691e-01	err = 1.7814314057e-09	time = 0.09 sec
[ Info: LBFGS: iter  187, time  975.63 s: f = -0.273264679414, ‖∇f‖ = 5.2723e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.767421973633e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.767421945382e-01	err = 7.9236797418e-10	time = 0.07 sec
[ Info: LBFGS: iter  188, time  976.36 s: f = -0.273264731290, ‖∇f‖ = 6.4665e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.766872205654e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.766872174361e-01	err = 1.0396357887e-09	time = 0.07 sec
[ Info: LBFGS: iter  189, time  977.08 s: f = -0.273264767679, ‖∇f‖ = 9.9049e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.766967641060e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.766967637394e-01	err = 2.0981305723e-09	time = 0.36 sec
[ Info: LBFGS: iter  190, time  978.05 s: f = -0.273264818438, ‖∇f‖ = 5.8232e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.767219705006e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.767219689583e-01	err = 2.1271834938e-09	time = 0.08 sec
[ Info: LBFGS: iter  191, time  978.75 s: f = -0.273264883219, ‖∇f‖ = 5.0729e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.767527763688e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.767527741140e-01	err = 9.6804643183e-10	time = 0.06 sec
[ Info: LBFGS: iter  192, time  979.56 s: f = -0.273264932574, ‖∇f‖ = 7.3332e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.768092611055e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.768092546795e-01	err = 1.9216532006e-09	time = 0.06 sec
[ Info: LBFGS: iter  193, time  980.29 s: f = -0.273265008002, ‖∇f‖ = 8.3120e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769703094195e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.769702847581e-01	err = 2.0268032020e-09	time = 0.07 sec
[ Info: LBFGS: iter  194, time  981.35 s: f = -0.273265046576, ‖∇f‖ = 1.6803e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769516426028e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.769516416392e-01	err = 7.6116111015e-09	time = 0.05 sec
[ Info: LBFGS: iter  195, time  982.07 s: f = -0.273265157726, ‖∇f‖ = 5.2448e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769446882418e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.769446879824e-01	err = 2.9030694112e-09	time = 0.07 sec
[ Info: LBFGS: iter  196, time  982.81 s: f = -0.273265202826, ‖∇f‖ = 5.7354e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769545446496e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.769545416856e-01	err = 7.3607404224e-09	time = 0.06 sec
[ Info: LBFGS: iter  197, time  983.55 s: f = -0.273265259520, ‖∇f‖ = 8.1885e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769849318412e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.769849242592e-01	err = 6.6225776188e-10	time = 0.07 sec
[ Info: LBFGS: iter  198, time  984.30 s: f = -0.273265334929, ‖∇f‖ = 8.0328e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.769764186629e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.769763223155e-01	err = 2.8607955001e-09	time = 0.07 sec
[ Info: CTMRG init:	obj = +4.769817945901e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +4.769817694585e-01	err = 1.3320949671e-09	time = 0.12 sec
[ Info: LBFGS: iter  199, time  986.02 s: f = -0.273265394805, ‖∇f‖ = 1.1434e-03, α = 5.11e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +4.770224767611e-01	err = 1.0000e+00
[ Info: CTMRG conv 6:	obj = +4.770224657374e-01	err = 6.8019993036e-09	time = 0.06 sec
┌ Warning: LBFGS: not converged to requested tol after 200 iterations and time 986.74 s: f = -0.273265481969, ‖∇f‖ = 5.2333e-04
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.2732654819685406

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -7.101026186047375e-5

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

