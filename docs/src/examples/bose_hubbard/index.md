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
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, trscheme = (; alg = :fixedspace))
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
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.514482939589e-11im	err = 3.6943031884e-09	time = 8.81 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity = 3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 9.360531870693, ‖∇f‖ = 1.6954e+01
[ Info: LBFGS: iter    1, time  861.65 s: f = 0.114269686001, ‖∇f‖ = 6.0686e+00, α = 1.56e+02, m = 0, nfg = 7
[ Info: LBFGS: iter    2, time  902.28 s: f = 0.059480938116, ‖∇f‖ = 7.2206e+00, α = 5.27e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  906.36 s: f = -0.046499452190, ‖∇f‖ = 1.6329e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  910.22 s: f = -0.079703746795, ‖∇f‖ = 1.4901e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  923.97 s: f = -0.125317851795, ‖∇f‖ = 3.2630e+00, α = 5.23e-01, m = 4, nfg = 3
[ Info: LBFGS: iter    6, time  928.10 s: f = -0.163554919012, ‖∇f‖ = 1.2781e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  932.62 s: f = -0.193532735097, ‖∇f‖ = 9.6932e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  940.14 s: f = -0.208656321174, ‖∇f‖ = 7.0028e-01, α = 1.68e-01, m = 7, nfg = 2
[ Info: LBFGS: iter    9, time  947.93 s: f = -0.220718433074, ‖∇f‖ = 4.3381e-01, α = 3.95e-01, m = 8, nfg = 2
[ Info: LBFGS: iter   10, time  951.64 s: f = -0.227817345457, ‖∇f‖ = 5.8993e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  953.60 s: f = -0.235906486487, ‖∇f‖ = 5.2265e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  955.46 s: f = -0.245544719100, ‖∇f‖ = 3.6462e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  957.65 s: f = -0.251717238994, ‖∇f‖ = 3.3074e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  959.26 s: f = -0.256869388318, ‖∇f‖ = 2.9129e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  960.71 s: f = -0.265345632562, ‖∇f‖ = 2.3580e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  961.94 s: f = -0.267397826877, ‖∇f‖ = 3.0098e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  963.60 s: f = -0.268894232398, ‖∇f‖ = 1.1725e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  964.83 s: f = -0.269501536195, ‖∇f‖ = 8.8162e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  966.07 s: f = -0.270154405735, ‖∇f‖ = 7.1880e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  967.32 s: f = -0.270612692556, ‖∇f‖ = 6.5906e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  969.04 s: f = -0.270978612765, ‖∇f‖ = 6.8050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  970.27 s: f = -0.271251104436, ‖∇f‖ = 4.7832e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  971.47 s: f = -0.271592065728, ‖∇f‖ = 5.2245e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  972.68 s: f = -0.271907473221, ‖∇f‖ = 4.7783e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  974.39 s: f = -0.272188790538, ‖∇f‖ = 6.1727e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  975.65 s: f = -0.272341714966, ‖∇f‖ = 2.8588e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  976.85 s: f = -0.272416985030, ‖∇f‖ = 2.4404e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  978.07 s: f = -0.272488140154, ‖∇f‖ = 2.8167e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  979.74 s: f = -0.272607173817, ‖∇f‖ = 4.0551e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  980.96 s: f = -0.272669542235, ‖∇f‖ = 2.8338e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  982.14 s: f = -0.272710735522, ‖∇f‖ = 1.3171e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  983.36 s: f = -0.272737399295, ‖∇f‖ = 1.5064e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  985.07 s: f = -0.272785529262, ‖∇f‖ = 2.2115e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  986.30 s: f = -0.272869320244, ‖∇f‖ = 2.7454e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  987.51 s: f = -0.272917746572, ‖∇f‖ = 4.3200e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  988.74 s: f = -0.272982782986, ‖∇f‖ = 1.3998e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  990.43 s: f = -0.273001975638, ‖∇f‖ = 9.8876e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  991.62 s: f = -0.273014701189, ‖∇f‖ = 1.2336e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  992.83 s: f = -0.273032513165, ‖∇f‖ = 1.6628e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  994.05 s: f = -0.273047957590, ‖∇f‖ = 1.1548e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  995.73 s: f = -0.273056319323, ‖∇f‖ = 6.3307e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  996.93 s: f = -0.273062571626, ‖∇f‖ = 6.8219e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  998.12 s: f = -0.273067065010, ‖∇f‖ = 8.8510e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  999.33 s: f = -0.273077210319, ‖∇f‖ = 9.9163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1002.27 s: f = -0.273086893652, ‖∇f‖ = 1.8575e-02, α = 5.12e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   46, time 1003.48 s: f = -0.273103078808, ‖∇f‖ = 8.5718e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1004.69 s: f = -0.273110799156, ‖∇f‖ = 5.8581e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1006.37 s: f = -0.273120104700, ‖∇f‖ = 8.0410e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1007.59 s: f = -0.273131281135, ‖∇f‖ = 1.1878e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1008.80 s: f = -0.273143808380, ‖∇f‖ = 9.4250e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1010.01 s: f = -0.273153887746, ‖∇f‖ = 7.2150e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1011.68 s: f = -0.273158885697, ‖∇f‖ = 6.7195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1012.87 s: f = -0.273161233673, ‖∇f‖ = 4.1608e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1014.06 s: f = -0.273163225690, ‖∇f‖ = 4.0591e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1015.25 s: f = -0.273166294493, ‖∇f‖ = 4.9790e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1016.91 s: f = -0.273169366234, ‖∇f‖ = 4.4714e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1018.11 s: f = -0.273172354171, ‖∇f‖ = 6.3685e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1019.31 s: f = -0.273175363774, ‖∇f‖ = 3.9908e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1020.49 s: f = -0.273177279507, ‖∇f‖ = 3.9256e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1022.16 s: f = -0.273182789742, ‖∇f‖ = 6.6560e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1024.61 s: f = -0.273184790975, ‖∇f‖ = 5.6288e-03, α = 5.40e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   62, time 1025.80 s: f = -0.273186538843, ‖∇f‖ = 2.7232e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1027.47 s: f = -0.273187761555, ‖∇f‖ = 2.8878e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1028.69 s: f = -0.273189383961, ‖∇f‖ = 3.5847e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1029.94 s: f = -0.273193896683, ‖∇f‖ = 8.4506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1031.17 s: f = -0.273197826303, ‖∇f‖ = 6.5557e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1032.87 s: f = -0.273200889010, ‖∇f‖ = 3.7359e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1034.09 s: f = -0.273203155255, ‖∇f‖ = 3.5393e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1035.31 s: f = -0.273203929922, ‖∇f‖ = 3.6211e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1036.51 s: f = -0.273204838588, ‖∇f‖ = 3.2228e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1038.20 s: f = -0.273208066645, ‖∇f‖ = 3.2060e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1040.67 s: f = -0.273208665413, ‖∇f‖ = 2.6372e-03, α = 3.31e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   73, time 1041.86 s: f = -0.273209101949, ‖∇f‖ = 2.4094e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1043.53 s: f = -0.273212059074, ‖∇f‖ = 3.6765e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1044.78 s: f = -0.273214544738, ‖∇f‖ = 4.4248e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1047.23 s: f = -0.273216168915, ‖∇f‖ = 4.5470e-03, α = 4.51e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   77, time 1048.92 s: f = -0.273217808146, ‖∇f‖ = 2.2760e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1050.15 s: f = -0.273218886889, ‖∇f‖ = 2.0807e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1051.34 s: f = -0.273220355057, ‖∇f‖ = 2.6274e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1052.56 s: f = -0.273220933232, ‖∇f‖ = 6.7890e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1054.27 s: f = -0.273223128592, ‖∇f‖ = 2.3155e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1055.52 s: f = -0.273223759946, ‖∇f‖ = 1.4721e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 1056.73 s: f = -0.273224313347, ‖∇f‖ = 1.6492e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1057.93 s: f = -0.273224595630, ‖∇f‖ = 5.4590e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time 1059.63 s: f = -0.273225490090, ‖∇f‖ = 2.6423e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time 1060.86 s: f = -0.273226533018, ‖∇f‖ = 1.3404e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time 1062.07 s: f = -0.273227338976, ‖∇f‖ = 1.9614e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time 1063.27 s: f = -0.273228071678, ‖∇f‖ = 2.3639e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time 1064.98 s: f = -0.273228782600, ‖∇f‖ = 2.1160e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time 1066.21 s: f = -0.273229429650, ‖∇f‖ = 1.6753e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time 1067.41 s: f = -0.273230390996, ‖∇f‖ = 2.3609e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time 1068.61 s: f = -0.273230928765, ‖∇f‖ = 4.6107e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time 1070.27 s: f = -0.273231840740, ‖∇f‖ = 2.4182e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time 1071.48 s: f = -0.273233071637, ‖∇f‖ = 2.0758e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time 1072.68 s: f = -0.273234113592, ‖∇f‖ = 3.0066e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time 1073.90 s: f = -0.273235305338, ‖∇f‖ = 4.2795e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time 1075.58 s: f = -0.273236308005, ‖∇f‖ = 1.8402e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time 1076.79 s: f = -0.273236758321, ‖∇f‖ = 1.2908e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time 1077.99 s: f = -0.273237372099, ‖∇f‖ = 2.2772e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time 1079.19 s: f = -0.273237743321, ‖∇f‖ = 2.7806e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, time 1080.86 s: f = -0.273238153812, ‖∇f‖ = 1.7400e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time 1082.08 s: f = -0.273238392828, ‖∇f‖ = 9.7559e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time 1083.29 s: f = -0.273238534969, ‖∇f‖ = 1.3297e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time 1084.49 s: f = -0.273238765149, ‖∇f‖ = 2.0593e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time 1086.13 s: f = -0.273239348217, ‖∇f‖ = 3.1203e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time 1087.35 s: f = -0.273240193588, ‖∇f‖ = 3.3818e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time 1089.76 s: f = -0.273240446453, ‖∇f‖ = 4.2123e-03, α = 2.30e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  108, time 1091.43 s: f = -0.273241296795, ‖∇f‖ = 2.3185e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time 1092.70 s: f = -0.273241818027, ‖∇f‖ = 1.0065e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time 1093.94 s: f = -0.273242159613, ‖∇f‖ = 1.5444e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time 1095.14 s: f = -0.273242469229, ‖∇f‖ = 1.8806e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time 1096.80 s: f = -0.273242706300, ‖∇f‖ = 3.1393e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time 1098.04 s: f = -0.273243075339, ‖∇f‖ = 1.2114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time 1099.27 s: f = -0.273243239034, ‖∇f‖ = 8.4598e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time 1100.48 s: f = -0.273243372014, ‖∇f‖ = 1.2172e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time 1102.19 s: f = -0.273243583308, ‖∇f‖ = 1.5371e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, time 1103.42 s: f = -0.273243938735, ‖∇f‖ = 2.1381e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time 1104.65 s: f = -0.273244363148, ‖∇f‖ = 1.6240e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time 1105.85 s: f = -0.273244738065, ‖∇f‖ = 1.2512e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, time 1107.53 s: f = -0.273245251721, ‖∇f‖ = 2.3616e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, time 1108.80 s: f = -0.273245518354, ‖∇f‖ = 2.4101e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, time 1110.02 s: f = -0.273245777298, ‖∇f‖ = 1.7471e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, time 1111.23 s: f = -0.273246338830, ‖∇f‖ = 1.4633e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, time 1112.91 s: f = -0.273246614070, ‖∇f‖ = 1.8858e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, time 1114.13 s: f = -0.273247568165, ‖∇f‖ = 2.9878e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  126, time 1115.37 s: f = -0.273248615113, ‖∇f‖ = 4.4438e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, time 1116.58 s: f = -0.273249470467, ‖∇f‖ = 2.3517e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, time 1118.29 s: f = -0.273249854779, ‖∇f‖ = 1.0060e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, time 1119.52 s: f = -0.273250016419, ‖∇f‖ = 9.6136e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, time 1120.73 s: f = -0.273250163684, ‖∇f‖ = 1.2541e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, time 1121.94 s: f = -0.273250656329, ‖∇f‖ = 1.9185e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, time 1124.91 s: f = -0.273250904715, ‖∇f‖ = 2.4524e-03, α = 5.53e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  133, time 1126.15 s: f = -0.273251281759, ‖∇f‖ = 1.5466e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, time 1127.36 s: f = -0.273251560398, ‖∇f‖ = 1.0582e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, time 1129.07 s: f = -0.273251766620, ‖∇f‖ = 1.2828e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, time 1130.30 s: f = -0.273252106306, ‖∇f‖ = 1.9167e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  137, time 1131.52 s: f = -0.273252998552, ‖∇f‖ = 3.2747e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, time 1132.73 s: f = -0.273253992568, ‖∇f‖ = 3.0379e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, time 1134.45 s: f = -0.273254876737, ‖∇f‖ = 2.7112e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, time 1136.91 s: f = -0.273255330095, ‖∇f‖ = 2.2918e-03, α = 4.12e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  141, time 1138.12 s: f = -0.273255765188, ‖∇f‖ = 1.6608e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, time 1139.82 s: f = -0.273256046421, ‖∇f‖ = 1.8868e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, time 1141.04 s: f = -0.273256528884, ‖∇f‖ = 2.0170e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, time 1142.25 s: f = -0.273256984988, ‖∇f‖ = 3.1684e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, time 1143.46 s: f = -0.273257432568, ‖∇f‖ = 1.1398e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, time 1145.16 s: f = -0.273257623908, ‖∇f‖ = 1.0447e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, time 1146.40 s: f = -0.273257804170, ‖∇f‖ = 1.2568e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, time 1147.63 s: f = -0.273257965252, ‖∇f‖ = 9.3561e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, time 1148.85 s: f = -0.273258082189, ‖∇f‖ = 1.6544e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 1150.55 s: f = -0.273258235812, ‖∇f‖ = 7.8440e-04
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.273258235811703

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -9.752529125212733e-5

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

