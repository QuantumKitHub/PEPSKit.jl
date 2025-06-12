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
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514786519231e-11im	err = 3.6943030570e-09	time = 4.71 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 9.360531870693, ‖∇f‖ = 1.6954e+01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time  447.37 s: f = 0.114269686001, ‖∇f‖ = 6.0686e+00, α = 1.56e+02, m = 0, nfg = 7
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time  466.08 s: f = 0.059480937972, ‖∇f‖ = 7.2206e+00, α = 5.27e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  467.79 s: f = -0.046499451984, ‖∇f‖ = 1.6329e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  469.43 s: f = -0.079703746601, ‖∇f‖ = 1.4901e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  474.93 s: f = -0.125317853655, ‖∇f‖ = 3.2630e+00, α = 5.23e-01, m = 4, nfg = 3
[ Info: LBFGS: iter    6, time  476.69 s: f = -0.163554919741, ‖∇f‖ = 1.2781e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  478.35 s: f = -0.193532735759, ‖∇f‖ = 9.6932e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  481.56 s: f = -0.208656321991, ‖∇f‖ = 7.0028e-01, α = 1.68e-01, m = 7, nfg = 2
[ Info: LBFGS: iter    9, time  484.64 s: f = -0.220718433428, ‖∇f‖ = 4.3381e-01, α = 3.95e-01, m = 8, nfg = 2
[ Info: LBFGS: iter   10, time  486.23 s: f = -0.227817345394, ‖∇f‖ = 5.8993e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  487.03 s: f = -0.235906487248, ‖∇f‖ = 5.2265e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  487.81 s: f = -0.245544719368, ‖∇f‖ = 3.6462e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  488.49 s: f = -0.251717239679, ‖∇f‖ = 3.3074e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  489.19 s: f = -0.256869389571, ‖∇f‖ = 2.9129e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  489.81 s: f = -0.265345632904, ‖∇f‖ = 2.3580e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  490.31 s: f = -0.267397828052, ‖∇f‖ = 3.0098e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  490.80 s: f = -0.268894232375, ‖∇f‖ = 1.1725e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  491.31 s: f = -0.269501536254, ‖∇f‖ = 8.8162e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  491.83 s: f = -0.270154405655, ‖∇f‖ = 7.1880e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  492.33 s: f = -0.270612692416, ‖∇f‖ = 6.5906e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  492.87 s: f = -0.270978612593, ‖∇f‖ = 6.8050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  493.36 s: f = -0.271251104300, ‖∇f‖ = 4.7832e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  493.86 s: f = -0.271592065719, ‖∇f‖ = 5.2245e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  494.36 s: f = -0.271907473273, ‖∇f‖ = 4.7783e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  494.85 s: f = -0.272188790454, ‖∇f‖ = 6.1727e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  495.36 s: f = -0.272341714824, ‖∇f‖ = 2.8588e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  495.90 s: f = -0.272416984884, ‖∇f‖ = 2.4404e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  496.38 s: f = -0.272488139998, ‖∇f‖ = 2.8167e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  496.88 s: f = -0.272607173868, ‖∇f‖ = 4.0551e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  497.38 s: f = -0.272669541571, ‖∇f‖ = 2.8338e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  497.86 s: f = -0.272710735494, ‖∇f‖ = 1.3171e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  498.35 s: f = -0.272737399013, ‖∇f‖ = 1.5064e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  498.89 s: f = -0.272785529117, ‖∇f‖ = 2.2114e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  499.36 s: f = -0.272869319752, ‖∇f‖ = 2.7454e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  499.86 s: f = -0.272917745978, ‖∇f‖ = 4.3200e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  500.38 s: f = -0.272982782834, ‖∇f‖ = 1.3998e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  500.86 s: f = -0.273001975520, ‖∇f‖ = 9.8876e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  501.35 s: f = -0.273014701113, ‖∇f‖ = 1.2336e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  501.89 s: f = -0.273032513067, ‖∇f‖ = 1.6628e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  502.36 s: f = -0.273047957526, ‖∇f‖ = 1.1548e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  502.87 s: f = -0.273056319292, ‖∇f‖ = 6.3307e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  503.35 s: f = -0.273062571619, ‖∇f‖ = 6.8219e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  503.85 s: f = -0.273067065002, ‖∇f‖ = 8.8511e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  504.36 s: f = -0.273077210183, ‖∇f‖ = 9.9163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  505.37 s: f = -0.273086893690, ‖∇f‖ = 1.8575e-02, α = 5.12e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   46, time  505.87 s: f = -0.273103078890, ‖∇f‖ = 8.5717e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  506.37 s: f = -0.273110799084, ‖∇f‖ = 5.8581e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  506.86 s: f = -0.273120104625, ‖∇f‖ = 8.0410e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  507.35 s: f = -0.273131281022, ‖∇f‖ = 1.1878e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  507.89 s: f = -0.273143808327, ‖∇f‖ = 9.4250e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  508.37 s: f = -0.273153887747, ‖∇f‖ = 7.2151e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  508.87 s: f = -0.273158885694, ‖∇f‖ = 6.7196e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  509.37 s: f = -0.273161233666, ‖∇f‖ = 4.1608e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  509.87 s: f = -0.273163225657, ‖∇f‖ = 4.0591e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  510.35 s: f = -0.273166294377, ‖∇f‖ = 4.9791e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  510.87 s: f = -0.273169366104, ‖∇f‖ = 4.4715e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  511.36 s: f = -0.273172354404, ‖∇f‖ = 6.3680e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  511.85 s: f = -0.273175363987, ‖∇f‖ = 3.9913e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  512.36 s: f = -0.273177279849, ‖∇f‖ = 3.9256e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  512.85 s: f = -0.273182789727, ‖∇f‖ = 6.6564e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  513.87 s: f = -0.273184791112, ‖∇f‖ = 5.6286e-03, α = 5.40e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   62, time  514.35 s: f = -0.273186538799, ‖∇f‖ = 2.7233e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  514.84 s: f = -0.273187761490, ‖∇f‖ = 2.8881e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  515.33 s: f = -0.273189383694, ‖∇f‖ = 3.5844e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  515.85 s: f = -0.273193894986, ‖∇f‖ = 8.4540e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  516.34 s: f = -0.273197827972, ‖∇f‖ = 6.5506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  516.87 s: f = -0.273200889354, ‖∇f‖ = 3.7353e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  517.33 s: f = -0.273203154672, ‖∇f‖ = 3.5496e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  517.82 s: f = -0.273203930333, ‖∇f‖ = 3.6200e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  518.31 s: f = -0.273204836234, ‖∇f‖ = 3.2203e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  518.82 s: f = -0.273208060322, ‖∇f‖ = 3.2220e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  519.86 s: f = -0.273208659882, ‖∇f‖ = 2.6304e-03, α = 3.30e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   73, time  520.36 s: f = -0.273209094356, ‖∇f‖ = 2.4028e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  520.86 s: f = -0.273212024054, ‖∇f‖ = 3.6763e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  521.36 s: f = -0.273214505104, ‖∇f‖ = 4.4241e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  522.35 s: f = -0.273216120203, ‖∇f‖ = 4.6060e-03, α = 4.46e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   77, time  522.88 s: f = -0.273217778997, ‖∇f‖ = 2.2945e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  523.36 s: f = -0.273218861805, ‖∇f‖ = 2.0798e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  523.87 s: f = -0.273220320459, ‖∇f‖ = 2.6195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time  524.36 s: f = -0.273220966454, ‖∇f‖ = 6.6742e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time  524.87 s: f = -0.273223114382, ‖∇f‖ = 2.3270e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time  525.37 s: f = -0.273223753168, ‖∇f‖ = 1.4786e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time  525.91 s: f = -0.273224307696, ‖∇f‖ = 1.6485e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time  526.40 s: f = -0.273224600439, ‖∇f‖ = 5.4182e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time  526.90 s: f = -0.273225488603, ‖∇f‖ = 2.6481e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time  527.39 s: f = -0.273226550195, ‖∇f‖ = 1.3533e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time  527.89 s: f = -0.273227343499, ‖∇f‖ = 1.9551e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time  528.39 s: f = -0.273228077778, ‖∇f‖ = 2.4565e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time  528.94 s: f = -0.273228825482, ‖∇f‖ = 1.9176e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time  529.41 s: f = -0.273229465307, ‖∇f‖ = 1.6975e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time  529.91 s: f = -0.273230476822, ‖∇f‖ = 2.4229e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time  530.91 s: f = -0.273230923961, ‖∇f‖ = 3.4496e-03, α = 5.28e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   93, time  531.41 s: f = -0.273231613690, ‖∇f‖ = 2.4096e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time  531.94 s: f = -0.273233077541, ‖∇f‖ = 2.2882e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time  532.42 s: f = -0.273234143388, ‖∇f‖ = 3.6772e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time  532.93 s: f = -0.273235315095, ‖∇f‖ = 2.7705e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time  533.42 s: f = -0.273236133845, ‖∇f‖ = 2.0233e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time  533.93 s: f = -0.273236758946, ‖∇f‖ = 1.8428e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time  534.43 s: f = -0.273237309116, ‖∇f‖ = 2.3707e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time  534.98 s: f = -0.273237974552, ‖∇f‖ = 3.0858e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, time  535.45 s: f = -0.273238197348, ‖∇f‖ = 2.5086e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time  535.95 s: f = -0.273238449282, ‖∇f‖ = 8.1789e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time  536.44 s: f = -0.273238537002, ‖∇f‖ = 7.6412e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time  536.95 s: f = -0.273238800364, ‖∇f‖ = 1.6776e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time  537.45 s: f = -0.273239205513, ‖∇f‖ = 2.6281e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time  538.00 s: f = -0.273239986999, ‖∇f‖ = 3.1798e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time  538.99 s: f = -0.273240310042, ‖∇f‖ = 4.0499e-03, α = 3.69e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  108, time  539.50 s: f = -0.273241100661, ‖∇f‖ = 2.2454e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time  540.02 s: f = -0.273241570020, ‖∇f‖ = 9.0842e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time  540.49 s: f = -0.273241816398, ‖∇f‖ = 1.4870e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time  541.03 s: f = -0.273242081589, ‖∇f‖ = 2.0094e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time  541.52 s: f = -0.273242493083, ‖∇f‖ = 2.3851e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time  542.00 s: f = -0.273242825761, ‖∇f‖ = 2.2691e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time  542.51 s: f = -0.273243095912, ‖∇f‖ = 1.0385e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time  543.01 s: f = -0.273243261674, ‖∇f‖ = 1.1665e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time  543.49 s: f = -0.273243430429, ‖∇f‖ = 1.5301e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, time  544.04 s: f = -0.273243845538, ‖∇f‖ = 1.9295e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time  544.54 s: f = -0.273243996942, ‖∇f‖ = 3.7175e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time  545.03 s: f = -0.273244541844, ‖∇f‖ = 1.3305e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, time  545.54 s: f = -0.273244794317, ‖∇f‖ = 1.0951e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, time  546.03 s: f = -0.273245079103, ‖∇f‖ = 1.7333e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, time  546.52 s: f = -0.273245452668, ‖∇f‖ = 2.1114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, time  547.05 s: f = -0.273245861726, ‖∇f‖ = 2.8753e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, time  547.54 s: f = -0.273246397362, ‖∇f‖ = 1.5979e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, time  548.03 s: f = -0.273246879624, ‖∇f‖ = 1.7962e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  126, time  548.53 s: f = -0.273247409052, ‖∇f‖ = 2.3830e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, time  549.02 s: f = -0.273248235197, ‖∇f‖ = 4.1297e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, time  549.52 s: f = -0.273249227894, ‖∇f‖ = 2.6896e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, time  550.08 s: f = -0.273249955792, ‖∇f‖ = 1.5408e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, time  550.55 s: f = -0.273250284830, ‖∇f‖ = 1.7044e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, time  551.08 s: f = -0.273250448746, ‖∇f‖ = 1.2287e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, time  551.58 s: f = -0.273250656030, ‖∇f‖ = 1.2902e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, time  552.07 s: f = -0.273250913523, ‖∇f‖ = 1.4269e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, time  552.57 s: f = -0.273251303649, ‖∇f‖ = 1.5899e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, time  553.11 s: f = -0.273251928054, ‖∇f‖ = 1.6908e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, time  554.08 s: f = -0.273252243511, ‖∇f‖ = 2.4858e-03, α = 4.43e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  137, time  554.60 s: f = -0.273252759132, ‖∇f‖ = 1.6358e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, time  555.08 s: f = -0.273253191927, ‖∇f‖ = 1.1045e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, time  555.58 s: f = -0.273253357638, ‖∇f‖ = 1.3790e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, time  556.11 s: f = -0.273253585255, ‖∇f‖ = 1.1112e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, time  556.59 s: f = -0.273253964333, ‖∇f‖ = 1.6607e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, time  557.09 s: f = -0.273254279074, ‖∇f‖ = 2.5294e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, time  557.58 s: f = -0.273254536702, ‖∇f‖ = 2.3866e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, time  558.08 s: f = -0.273254785704, ‖∇f‖ = 1.2933e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, time  558.58 s: f = -0.273255070970, ‖∇f‖ = 1.3534e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, time  559.13 s: f = -0.273255346722, ‖∇f‖ = 2.1967e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, time  559.60 s: f = -0.273255986144, ‖∇f‖ = 3.3170e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, time  560.12 s: f = -0.273256730620, ‖∇f‖ = 3.2763e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, time  561.11 s: f = -0.273256927747, ‖∇f‖ = 3.1783e-03, α = 2.33e-01, m = 20, nfg = 2
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 561.60 s: f = -0.273257402157, ‖∇f‖ = 1.4919e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.2732574021567276

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -0.00010057578914641676

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

