```@meta
EditURL = "../../../../examples/xxz/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/xxz/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/xxz/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/xxz)


# Néel order in the $U(1)$-symmetric XXZ model

Here, we want to look at a special case of the Heisenberg model, where the $x$ and $y$
couplings are equal, called the XXZ model

```math
H_0 = J \big(\sum_{\langle i, j \rangle} S_i^x S_j^x + S_i^y S_j^y + \Delta S_i^z S_j^z \big) .
```

For appropriate $\Delta$, the model enters an antiferromagnetic phase (Néel order) which we
will force by adding staggered magnetic charges to $H_0$. Furthermore, since the XXZ
Hamiltonian obeys a $U(1)$ symmetry, we will make use of that and work with $U(1)$-symmetric
PEPS and CTMRG environments. For simplicity, we will consider spin-$1/2$ operators.

But first, let's make this example deterministic and import the required packages:

````julia
using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528935);
````

## Constructing the model

Let us define the $U(1)$-symmetric XXZ Hamiltonian on a $2 \times 2$ unit cell with the
parameters:

````julia
J = 1.0
Delta = 1.0
spin = 1//2
symmetry = U1Irrep
lattice = InfiniteSquare(2, 2)
H₀ = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin);
````

This ensures that our PEPS ansatz can support the bipartite Néel order. As discussed above,
we encode the Néel order directly in the ansatz by adding staggered auxiliary physical
charges:

````julia
S_aux = [
    U1Irrep(-1//2) U1Irrep(1//2)
    U1Irrep(1//2) U1Irrep(-1//2)
]
H = add_physical_charge(H₀, S_aux);
````

## Specifying the symmetric virtual spaces

Before we create an initial PEPS and CTM environment, we need to think about which
symmetric spaces we need to construct. Since we want to exploit the global $U(1)$ symmetry
of the model, we will use TensorKit's `U1Space`s where we specify dimensions for each
symmetry sector. From the virtual spaces, we will need to construct a unit cell (a matrix)
of spaces which will be supplied to the PEPS constructor. The same is true for the physical
spaces, which can be extracted directly from the Hamiltonian `LocalOperator`:

````julia
V_peps = U1Space(0 => 2, 1 => 1, -1 => 1)
V_env = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)
virtual_spaces = fill(V_peps, size(lattice)...)
physical_spaces = physicalspace(H)
````

````
2×2 Matrix{TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}}:
 Rep[TensorKitSectors.U₁](0=>1, -1=>1)  Rep[TensorKitSectors.U₁](0=>1, 1=>1)
 Rep[TensorKitSectors.U₁](0=>1, 1=>1)   Rep[TensorKitSectors.U₁](0=>1, -1=>1)
````

## Ground state search

From this point onwards it's business as usual: Create an initial PEPS and environment
(using the symmetric spaces), specify the algorithmic parameters and optimize:

````julia
boundary_alg = (; tol=1e-8, alg=:simultaneous, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, maxiter=85, ls_maxiter=3, ls_maxfg=3)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -1.121020187593e+04 -6.991066478499e+03im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +6.369731502336e+03 -8.500546755386e-08im	err = 7.5599921139e-09	time = 2.05 sec

````

Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian. Note that the
optimization might take a while since precompilation of symmetric AD code takes longer and
because symmetric tensors do create a bit of overhead (which does pay off at larger bond
and environment dimensions):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = -0.033045967451, ‖∇f‖ = 3.2952e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -9.90e-03, ϕ - ϕ₀ = -1.53e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  151.05 s: f = -0.185750019542, ‖∇f‖ = 1.8647e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.84e-03, ϕ - ϕ₀ = -3.93e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  198.17 s: f = -0.579230108476, ‖∇f‖ = 5.7732e-01, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  209.90 s: f = -0.613445426304, ‖∇f‖ = 3.3947e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  220.81 s: f = -0.638685295144, ‖∇f‖ = 2.2104e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, time  232.32 s: f = -0.650336962208, ‖∇f‖ = 1.9524e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  243.27 s: f = -0.654880752783, ‖∇f‖ = 7.1842e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  253.29 s: f = -0.656075650331, ‖∇f‖ = 5.2129e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  264.02 s: f = -0.659041890147, ‖∇f‖ = 5.3917e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  274.83 s: f = -0.660552875456, ‖∇f‖ = 9.6848e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  284.84 s: f = -0.662163341463, ‖∇f‖ = 2.9524e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  295.73 s: f = -0.662506513828, ‖∇f‖ = 2.1440e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, time  305.73 s: f = -0.662847746095, ‖∇f‖ = 2.0917e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, time  316.59 s: f = -0.663230218002, ‖∇f‖ = 2.5387e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  323.15 s: f = -0.663678142653, ‖∇f‖ = 2.2924e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  328.96 s: f = -0.664034475269, ‖∇f‖ = 2.1574e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  340.52 s: f = -0.664687988771, ‖∇f‖ = 2.7632e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time  363.13 s: f = -0.664947633065, ‖∇f‖ = 3.2552e-02, α = 4.47e-01, m = 15, nfg = 2
[ Info: LBFGS: iter   18, time  374.71 s: f = -0.665379251393, ‖∇f‖ = 2.5817e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  386.41 s: f = -0.665603907305, ‖∇f‖ = 2.4951e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time  397.29 s: f = -0.665762559605, ‖∇f‖ = 1.5130e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time  408.95 s: f = -0.666003995146, ‖∇f‖ = 1.6565e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time  420.69 s: f = -0.666347181648, ‖∇f‖ = 2.3025e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  432.27 s: f = -0.666599630335, ‖∇f‖ = 2.3462e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  444.52 s: f = -0.666793651525, ‖∇f‖ = 2.1908e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  456.87 s: f = -0.666949601577, ‖∇f‖ = 1.0867e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  469.23 s: f = -0.667058133735, ‖∇f‖ = 1.2475e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  480.73 s: f = -0.667165497296, ‖∇f‖ = 1.4340e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  492.42 s: f = -0.667263554939, ‖∇f‖ = 1.5667e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  504.11 s: f = -0.667357062063, ‖∇f‖ = 8.8494e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  515.48 s: f = -0.667450569240, ‖∇f‖ = 1.1496e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  527.47 s: f = -0.667569394671, ‖∇f‖ = 1.3976e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  539.89 s: f = -0.667657944766, ‖∇f‖ = 2.2321e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  551.52 s: f = -0.667799360459, ‖∇f‖ = 8.7543e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  563.83 s: f = -0.667852887856, ‖∇f‖ = 6.6668e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  576.15 s: f = -0.667926685233, ‖∇f‖ = 1.1902e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  588.14 s: f = -0.667979135052, ‖∇f‖ = 1.6349e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  595.62 s: f = -0.668039789983, ‖∇f‖ = 9.3108e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  608.27 s: f = -0.668087921855, ‖∇f‖ = 5.4669e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  619.58 s: f = -0.668109250700, ‖∇f‖ = 6.6628e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  627.01 s: f = -0.668159589044, ‖∇f‖ = 9.3986e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  634.57 s: f = -0.668230776691, ‖∇f‖ = 1.1534e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  646.27 s: f = -0.668277979949, ‖∇f‖ = 1.0707e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  658.67 s: f = -0.668314491012, ‖∇f‖ = 4.4476e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  670.98 s: f = -0.668333048195, ‖∇f‖ = 5.0062e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  682.77 s: f = -0.668357479998, ‖∇f‖ = 7.1015e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  695.32 s: f = -0.668412792965, ‖∇f‖ = 9.9323e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  719.56 s: f = -0.668439289699, ‖∇f‖ = 1.1349e-02, α = 4.53e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   48, time  732.05 s: f = -0.668482319738, ‖∇f‖ = 6.5163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  744.54 s: f = -0.668507687742, ‖∇f‖ = 3.4077e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  756.98 s: f = -0.668523331148, ‖∇f‖ = 4.4119e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  768.78 s: f = -0.668544159952, ‖∇f‖ = 6.8178e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  781.25 s: f = -0.668572411228, ‖∇f‖ = 8.9225e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  794.44 s: f = -0.668603530102, ‖∇f‖ = 5.4599e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  807.70 s: f = -0.668626672236, ‖∇f‖ = 3.2841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  819.27 s: f = -0.668639698673, ‖∇f‖ = 4.2163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  831.79 s: f = -0.668655954756, ‖∇f‖ = 5.4299e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  844.32 s: f = -0.668673395222, ‖∇f‖ = 5.4970e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  856.85 s: f = -0.668687854178, ‖∇f‖ = 3.7476e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  868.59 s: f = -0.668698487673, ‖∇f‖ = 3.3854e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  881.11 s: f = -0.668705111514, ‖∇f‖ = 3.8959e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  894.30 s: f = -0.668720251413, ‖∇f‖ = 4.7426e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  906.61 s: f = -0.668726232684, ‖∇f‖ = 7.2528e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  919.67 s: f = -0.668739847818, ‖∇f‖ = 2.5815e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  932.86 s: f = -0.668746591101, ‖∇f‖ = 2.1675e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  944.63 s: f = -0.668754620918, ‖∇f‖ = 3.0368e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  957.21 s: f = -0.668767041464, ‖∇f‖ = 3.2464e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  982.43 s: f = -0.668775239596, ‖∇f‖ = 4.1610e-03, α = 4.24e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   68, time  995.66 s: f = -0.668784522482, ‖∇f‖ = 2.0585e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1008.92 s: f = -0.668792875861, ‖∇f‖ = 2.5737e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1020.82 s: f = -0.668799555353, ‖∇f‖ = 3.0991e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1034.03 s: f = -0.668807510786, ‖∇f‖ = 3.9740e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1047.37 s: f = -0.668815529198, ‖∇f‖ = 2.8312e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1059.65 s: f = -0.668820072176, ‖∇f‖ = 1.8277e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1072.71 s: f = -0.668823045663, ‖∇f‖ = 2.1189e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1086.03 s: f = -0.668829224417, ‖∇f‖ = 3.3000e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1098.65 s: f = -0.668834661996, ‖∇f‖ = 3.3841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1112.36 s: f = -0.668839125190, ‖∇f‖ = 1.5494e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1126.16 s: f = -0.668842178047, ‖∇f‖ = 1.8074e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1140.07 s: f = -0.668846182434, ‖∇f‖ = 2.1704e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1152.70 s: f = -0.668849851199, ‖∇f‖ = 4.3895e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1165.98 s: f = -0.668854730352, ‖∇f‖ = 1.9014e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1179.77 s: f = -0.668857445978, ‖∇f‖ = 1.6213e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 1192.89 s: f = -0.668861083091, ‖∇f‖ = 2.1544e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1205.95 s: f = -0.668865914837, ‖∇f‖ = 2.4867e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 1219.27 s: f = -0.668869945971, ‖∇f‖ = 4.5125e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.6688699459708735

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian as
in the [Heisenberg example](@ref examples_heisenberg). In that example, with a non-symmetric
$D=2$ PEPS simulation, we reached a ground-state energy of around $E_\text{D=2} = -0.6625\dots$.
Again comparing against [Sandvik's](@cite sandvik_computational_2011) accurate QMC estimate
``E_{\text{ref}}=−0.6694421``, we see that we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

