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
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514065517581e-11im	err = 3.6943032303e-09	time = 10.11 sec

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
[ Info: LBFGS: iter    1, time  701.65 s: f = 0.114269686001, ‖∇f‖ = 6.0686e+00, α = 1.56e+02, m = 0, nfg = 7
[ Info: LBFGS: iter    2, time  728.49 s: f = 0.059480938109, ‖∇f‖ = 7.2206e+00, α = 5.27e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  731.02 s: f = -0.046499452154, ‖∇f‖ = 1.6329e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  733.33 s: f = -0.079703746761, ‖∇f‖ = 1.4901e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  741.06 s: f = -0.125317852116, ‖∇f‖ = 3.2630e+00, α = 5.23e-01, m = 4, nfg = 3
[ Info: LBFGS: iter    6, time  743.44 s: f = -0.163554919136, ‖∇f‖ = 1.2781e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  745.87 s: f = -0.193532735205, ‖∇f‖ = 9.6932e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  750.42 s: f = -0.208656321305, ‖∇f‖ = 7.0028e-01, α = 1.68e-01, m = 7, nfg = 2
[ Info: LBFGS: iter    9, time  754.81 s: f = -0.220718433131, ‖∇f‖ = 4.3381e-01, α = 3.95e-01, m = 8, nfg = 2
[ Info: LBFGS: iter   10, time  756.97 s: f = -0.227817345451, ‖∇f‖ = 5.8993e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  758.22 s: f = -0.235906486614, ‖∇f‖ = 5.2265e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  759.22 s: f = -0.245544719146, ‖∇f‖ = 3.6462e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  760.15 s: f = -0.251717239104, ‖∇f‖ = 3.3074e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  761.46 s: f = -0.256869388517, ‖∇f‖ = 2.9129e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  762.25 s: f = -0.265345632618, ‖∇f‖ = 2.3580e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  762.92 s: f = -0.267397827065, ‖∇f‖ = 3.0098e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  763.80 s: f = -0.268894232393, ‖∇f‖ = 1.1725e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  764.47 s: f = -0.269501536204, ‖∇f‖ = 8.8162e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  765.20 s: f = -0.270154405723, ‖∇f‖ = 7.1880e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  766.33 s: f = -0.270612692535, ‖∇f‖ = 6.5906e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  767.10 s: f = -0.270978612739, ‖∇f‖ = 6.8050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  767.78 s: f = -0.271251104415, ‖∇f‖ = 4.7832e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  768.66 s: f = -0.271592065726, ‖∇f‖ = 5.2245e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  769.33 s: f = -0.271907473229, ‖∇f‖ = 4.7783e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  769.98 s: f = -0.272188790523, ‖∇f‖ = 6.1727e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  770.91 s: f = -0.272341714943, ‖∇f‖ = 2.8588e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  771.62 s: f = -0.272416985007, ‖∇f‖ = 2.4404e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  772.51 s: f = -0.272488140129, ‖∇f‖ = 2.8167e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  773.65 s: f = -0.272607173824, ‖∇f‖ = 4.0551e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  774.38 s: f = -0.272669542134, ‖∇f‖ = 2.8338e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  775.06 s: f = -0.272710735517, ‖∇f‖ = 1.3171e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  776.00 s: f = -0.272737399252, ‖∇f‖ = 1.5064e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  776.65 s: f = -0.272785529240, ‖∇f‖ = 2.2115e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  777.35 s: f = -0.272869320169, ‖∇f‖ = 2.7454e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  778.48 s: f = -0.272917746483, ‖∇f‖ = 4.3200e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  779.25 s: f = -0.272982782962, ‖∇f‖ = 1.3998e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  779.88 s: f = -0.273001975620, ‖∇f‖ = 9.8876e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  780.76 s: f = -0.273014701177, ‖∇f‖ = 1.2336e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  781.44 s: f = -0.273032513150, ‖∇f‖ = 1.6628e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  782.05 s: f = -0.273047957581, ‖∇f‖ = 1.1548e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  782.92 s: f = -0.273056319318, ‖∇f‖ = 6.3307e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  783.58 s: f = -0.273062571625, ‖∇f‖ = 6.8219e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  784.32 s: f = -0.273067065009, ‖∇f‖ = 8.8510e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  785.40 s: f = -0.273077210298, ‖∇f‖ = 9.9163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  786.71 s: f = -0.273086893657, ‖∇f‖ = 1.8575e-02, α = 5.12e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   46, time  787.59 s: f = -0.273103078820, ‖∇f‖ = 8.5718e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  788.24 s: f = -0.273110799146, ‖∇f‖ = 5.8581e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  788.86 s: f = -0.273120104690, ‖∇f‖ = 8.0410e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  789.90 s: f = -0.273131281119, ‖∇f‖ = 1.1878e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  790.67 s: f = -0.273143808372, ‖∇f‖ = 9.4250e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  791.37 s: f = -0.273153887746, ‖∇f‖ = 7.2150e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  792.24 s: f = -0.273158885697, ‖∇f‖ = 6.7195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  792.89 s: f = -0.273161233672, ‖∇f‖ = 4.1608e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  793.51 s: f = -0.273163225685, ‖∇f‖ = 4.0591e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  794.38 s: f = -0.273166294476, ‖∇f‖ = 4.9791e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  795.00 s: f = -0.273169366216, ‖∇f‖ = 4.4714e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  795.65 s: f = -0.273172354203, ‖∇f‖ = 6.3685e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  796.67 s: f = -0.273175363804, ‖∇f‖ = 3.9908e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  797.42 s: f = -0.273177279555, ‖∇f‖ = 3.9256e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  798.14 s: f = -0.273182789739, ‖∇f‖ = 6.6561e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  799.68 s: f = -0.273184790994, ‖∇f‖ = 5.6288e-03, α = 5.40e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   62, time  800.38 s: f = -0.273186538837, ‖∇f‖ = 2.7232e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  801.29 s: f = -0.273187761546, ‖∇f‖ = 2.8879e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  802.02 s: f = -0.273189383925, ‖∇f‖ = 3.5846e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  802.81 s: f = -0.273193896451, ‖∇f‖ = 8.4511e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  803.91 s: f = -0.273197826539, ‖∇f‖ = 6.5550e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  804.62 s: f = -0.273200889061, ‖∇f‖ = 3.7358e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  805.29 s: f = -0.273203155178, ‖∇f‖ = 3.5407e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  806.21 s: f = -0.273203929980, ‖∇f‖ = 3.6210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  806.87 s: f = -0.273204838261, ‖∇f‖ = 3.2225e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  807.58 s: f = -0.273208065775, ‖∇f‖ = 3.2082e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  809.27 s: f = -0.273208664645, ‖∇f‖ = 2.6363e-03, α = 3.31e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   73, time  809.86 s: f = -0.273209100891, ‖∇f‖ = 2.4085e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  810.72 s: f = -0.273212054238, ‖∇f‖ = 3.6766e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  811.36 s: f = -0.273214539258, ‖∇f‖ = 4.4246e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  812.95 s: f = -0.273216162063, ‖∇f‖ = 4.5554e-03, α = 4.50e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   77, time  813.63 s: f = -0.273217804060, ‖∇f‖ = 2.2786e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  814.23 s: f = -0.273218883332, ‖∇f‖ = 2.0806e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  815.09 s: f = -0.273220350202, ‖∇f‖ = 2.6261e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time  815.76 s: f = -0.273220936627, ‖∇f‖ = 6.7759e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time  816.37 s: f = -0.273223126358, ‖∇f‖ = 2.3172e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time  817.34 s: f = -0.273223758693, ‖∇f‖ = 1.4729e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time  818.06 s: f = -0.273224312355, ‖∇f‖ = 1.6490e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time  818.66 s: f = -0.273224595270, ‖∇f‖ = 5.4563e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time  819.52 s: f = -0.273225489234, ‖∇f‖ = 2.6430e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time  820.21 s: f = -0.273226533430, ‖∇f‖ = 1.3412e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time  820.82 s: f = -0.273227338612, ‖∇f‖ = 1.9615e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time  821.71 s: f = -0.273228071289, ‖∇f‖ = 2.3685e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time  822.46 s: f = -0.273228784006, ‖∇f‖ = 2.1043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time  823.10 s: f = -0.273229430175, ‖∇f‖ = 1.6761e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time  823.97 s: f = -0.273230395114, ‖∇f‖ = 2.3630e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time  824.61 s: f = -0.273230913013, ‖∇f‖ = 4.6973e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time  825.22 s: f = -0.273231822369, ‖∇f‖ = 2.4314e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time  826.15 s: f = -0.273233026044, ‖∇f‖ = 2.0545e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time  826.88 s: f = -0.273234060512, ‖∇f‖ = 3.0017e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time  827.53 s: f = -0.273235400747, ‖∇f‖ = 3.5897e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time  828.40 s: f = -0.273236050489, ‖∇f‖ = 3.5955e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time  829.05 s: f = -0.273236795492, ‖∇f‖ = 1.2968e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time  829.66 s: f = -0.273237167136, ‖∇f‖ = 1.6986e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time  830.53 s: f = -0.273237666635, ‖∇f‖ = 2.2956e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, time  832.10 s: f = -0.273237902441, ‖∇f‖ = 2.7825e-03, α = 4.67e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  102, time  832.99 s: f = -0.273238278596, ‖∇f‖ = 1.4666e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time  833.63 s: f = -0.273238526414, ‖∇f‖ = 8.6123e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time  834.30 s: f = -0.273238754345, ‖∇f‖ = 1.6678e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time  835.20 s: f = -0.273239114660, ‖∇f‖ = 2.6137e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time  835.98 s: f = -0.273239823032, ‖∇f‖ = 3.4309e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time  836.70 s: f = -0.273240368425, ‖∇f‖ = 4.2900e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, time  837.60 s: f = -0.273241164149, ‖∇f‖ = 1.7960e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time  838.27 s: f = -0.273241477217, ‖∇f‖ = 1.0362e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time  838.99 s: f = -0.273241714533, ‖∇f‖ = 1.6183e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time  839.90 s: f = -0.273241995140, ‖∇f‖ = 2.2016e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time  840.64 s: f = -0.273242534073, ‖∇f‖ = 2.3919e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time  842.25 s: f = -0.273242711278, ‖∇f‖ = 2.5310e-03, α = 2.34e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  114, time  842.90 s: f = -0.273243067693, ‖∇f‖ = 1.2098e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time  843.58 s: f = -0.273243262667, ‖∇f‖ = 8.8410e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time  844.49 s: f = -0.273243396694, ‖∇f‖ = 1.3054e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, time  845.17 s: f = -0.273243632472, ‖∇f‖ = 1.6606e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time  845.92 s: f = -0.273243978972, ‖∇f‖ = 2.4494e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time  846.80 s: f = -0.273244469815, ‖∇f‖ = 1.7476e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, time  847.44 s: f = -0.273244908487, ‖∇f‖ = 1.1114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, time  848.15 s: f = -0.273245191811, ‖∇f‖ = 1.8355e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, time  849.04 s: f = -0.273245438183, ‖∇f‖ = 1.7466e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, time  849.78 s: f = -0.273245813350, ‖∇f‖ = 1.8849e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, time  850.49 s: f = -0.273246990667, ‖∇f‖ = 2.4067e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, time  852.08 s: f = -0.273247342251, ‖∇f‖ = 2.8162e-03, α = 4.42e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  126, time  852.72 s: f = -0.273247837859, ‖∇f‖ = 1.6699e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, time  853.62 s: f = -0.273248269606, ‖∇f‖ = 1.5051e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, time  854.38 s: f = -0.273248928543, ‖∇f‖ = 2.4379e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, time  855.08 s: f = -0.273249639369, ‖∇f‖ = 2.6766e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, time  855.99 s: f = -0.273250417480, ‖∇f‖ = 3.1154e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, time  856.68 s: f = -0.273250924677, ‖∇f‖ = 1.7813e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, time  857.35 s: f = -0.273251180562, ‖∇f‖ = 1.2277e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, time  858.27 s: f = -0.273251373009, ‖∇f‖ = 1.2344e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, time  859.00 s: f = -0.273251572043, ‖∇f‖ = 1.5999e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, time  859.67 s: f = -0.273251882821, ‖∇f‖ = 2.1224e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, time  860.53 s: f = -0.273252172658, ‖∇f‖ = 1.7119e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  137, time  861.16 s: f = -0.273252409193, ‖∇f‖ = 1.1616e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, time  861.80 s: f = -0.273252645647, ‖∇f‖ = 1.3121e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, time  862.69 s: f = -0.273252871996, ‖∇f‖ = 1.3829e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, time  863.40 s: f = -0.273253359134, ‖∇f‖ = 1.8830e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, time  864.06 s: f = -0.273253963261, ‖∇f‖ = 2.5909e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, time  864.94 s: f = -0.273254681754, ‖∇f‖ = 1.9402e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, time  865.57 s: f = -0.273255691367, ‖∇f‖ = 3.1128e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, time  866.18 s: f = -0.273256163902, ‖∇f‖ = 2.2255e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, time  867.02 s: f = -0.273256631020, ‖∇f‖ = 1.3304e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, time  867.73 s: f = -0.273257004955, ‖∇f‖ = 1.5051e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, time  868.41 s: f = -0.273257432452, ‖∇f‖ = 1.4023e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, time  869.94 s: f = -0.273257620772, ‖∇f‖ = 2.4790e-03, α = 1.96e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  149, time  870.56 s: f = -0.273257961671, ‖∇f‖ = 1.0700e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 871.43 s: f = -0.273258098818, ‖∇f‖ = 7.5289e-04
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.27325809881788027

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -9.802657701195407e-5

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

