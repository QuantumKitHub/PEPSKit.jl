```@meta
EditURL = "../../../../examples/xxz/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//xxz/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//xxz/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//xxz)


# Néel order in the $U(1)$-symmetric XXZ model

Here, we want to look at a special case of the Heisenberg model, where the $x$ and $y$
couplings are equal, called the XXZ model

```math
H_0 = J \big(\sum_{\langle i, j \rangle} S_i^x S_j^x + S_i^y S_j^y + \Delta S_i^z S_j^z \big) .
```

For appropriate $\Delta$, the model enters an antiferromagnetic phase (Néel order) which we
will force by adding staggered magnetic charges to ``H_0``. Furthermore, since the XXZ
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
physical_spaces = H.lattice
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
[ Info: CTMRG conv 26:	obj = +6.369731502336e+03 -8.500546755386e-08im	err = 7.5599921139e-09	time = 8.69 sec

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
[ Info: LBFGS: iter    1, time  808.38 s: f = -0.185750019542, ‖∇f‖ = 1.8647e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.84e-03, ϕ - ϕ₀ = -3.93e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  857.60 s: f = -0.579230108476, ‖∇f‖ = 5.7732e-01, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  868.75 s: f = -0.613445426304, ‖∇f‖ = 3.3947e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  880.15 s: f = -0.638685295144, ‖∇f‖ = 2.2104e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, time  891.56 s: f = -0.650336962208, ‖∇f‖ = 1.9524e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  902.39 s: f = -0.654880752783, ‖∇f‖ = 7.1842e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  912.20 s: f = -0.656075650331, ‖∇f‖ = 5.2129e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  922.70 s: f = -0.659041890147, ‖∇f‖ = 5.3917e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  933.30 s: f = -0.660552875456, ‖∇f‖ = 9.6848e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  943.09 s: f = -0.662163341463, ‖∇f‖ = 2.9524e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  953.68 s: f = -0.662506513828, ‖∇f‖ = 2.1440e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, time  964.33 s: f = -0.662847746095, ‖∇f‖ = 2.0917e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, time  974.26 s: f = -0.663230218002, ‖∇f‖ = 2.5387e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  980.64 s: f = -0.663678142653, ‖∇f‖ = 2.2924e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  987.02 s: f = -0.664034475269, ‖∇f‖ = 2.1574e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  997.52 s: f = -0.664687988771, ‖∇f‖ = 2.7632e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time 1020.76 s: f = -0.664947633065, ‖∇f‖ = 3.2552e-02, α = 4.47e-01, m = 15, nfg = 2
[ Info: LBFGS: iter   18, time 1031.65 s: f = -0.665379251393, ‖∇f‖ = 2.5817e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time 1043.13 s: f = -0.665603907305, ‖∇f‖ = 2.4951e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time 1054.48 s: f = -0.665762559605, ‖∇f‖ = 1.5130e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time 1065.68 s: f = -0.666003995146, ‖∇f‖ = 1.6565e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time 1076.34 s: f = -0.666347181648, ‖∇f‖ = 2.3025e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time 1088.47 s: f = -0.666599630335, ‖∇f‖ = 2.3462e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time 1100.74 s: f = -0.666793651525, ‖∇f‖ = 2.1908e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time 1112.91 s: f = -0.666949601577, ‖∇f‖ = 1.0867e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time 1125.12 s: f = -0.667058133735, ‖∇f‖ = 1.2475e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time 1136.45 s: f = -0.667165497296, ‖∇f‖ = 1.4340e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time 1147.86 s: f = -0.667263554939, ‖∇f‖ = 1.5667e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time 1159.14 s: f = -0.667357062063, ‖∇f‖ = 8.8494e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1170.29 s: f = -0.667450569240, ‖∇f‖ = 1.1496e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1182.11 s: f = -0.667569394671, ‖∇f‖ = 1.3976e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1194.13 s: f = -0.667657944766, ‖∇f‖ = 2.2321e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1205.47 s: f = -0.667799360459, ‖∇f‖ = 8.7543e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1217.47 s: f = -0.667852887856, ‖∇f‖ = 6.6668e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1229.64 s: f = -0.667926685233, ‖∇f‖ = 1.1902e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time 1242.59 s: f = -0.667979135052, ‖∇f‖ = 1.6349e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1249.99 s: f = -0.668039789983, ‖∇f‖ = 9.3108e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1261.95 s: f = -0.668087921855, ‖∇f‖ = 5.4669e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1274.08 s: f = -0.668109250700, ‖∇f‖ = 6.6628e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1281.58 s: f = -0.668159589044, ‖∇f‖ = 9.3986e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1288.46 s: f = -0.668230776691, ‖∇f‖ = 1.1534e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1300.78 s: f = -0.668277979949, ‖∇f‖ = 1.0707e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1313.19 s: f = -0.668314491012, ‖∇f‖ = 4.4476e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1325.48 s: f = -0.668333048195, ‖∇f‖ = 5.0062e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1337.91 s: f = -0.668357479998, ‖∇f‖ = 7.1015e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1349.38 s: f = -0.668412792965, ‖∇f‖ = 9.9323e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1373.92 s: f = -0.668439289699, ‖∇f‖ = 1.1349e-02, α = 4.53e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   48, time 1386.29 s: f = -0.668482319738, ‖∇f‖ = 6.5163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1398.51 s: f = -0.668507687742, ‖∇f‖ = 3.4077e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1409.94 s: f = -0.668523331148, ‖∇f‖ = 4.4119e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1422.07 s: f = -0.668544159952, ‖∇f‖ = 6.8178e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1434.33 s: f = -0.668572411228, ‖∇f‖ = 8.9225e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1447.43 s: f = -0.668603530102, ‖∇f‖ = 5.4599e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1460.33 s: f = -0.668626672236, ‖∇f‖ = 3.2841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1471.90 s: f = -0.668639698673, ‖∇f‖ = 4.2163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1484.21 s: f = -0.668655954756, ‖∇f‖ = 5.4299e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1496.56 s: f = -0.668673395222, ‖∇f‖ = 5.4970e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1508.06 s: f = -0.668687854178, ‖∇f‖ = 3.7476e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1520.29 s: f = -0.668698487673, ‖∇f‖ = 3.3854e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1532.55 s: f = -0.668705111514, ‖∇f‖ = 3.8959e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1545.51 s: f = -0.668720251413, ‖∇f‖ = 4.7426e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1557.82 s: f = -0.668726232684, ‖∇f‖ = 7.2528e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1570.65 s: f = -0.668739847818, ‖∇f‖ = 2.5815e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1583.68 s: f = -0.668746591101, ‖∇f‖ = 2.1675e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1595.79 s: f = -0.668754620918, ‖∇f‖ = 3.0368e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1608.17 s: f = -0.668767041464, ‖∇f‖ = 3.2464e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1633.44 s: f = -0.668775239596, ‖∇f‖ = 4.1610e-03, α = 4.24e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   68, time 1646.37 s: f = -0.668784522482, ‖∇f‖ = 2.0585e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1658.66 s: f = -0.668792875861, ‖∇f‖ = 2.5737e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1671.00 s: f = -0.668799555353, ‖∇f‖ = 3.0991e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1683.96 s: f = -0.668807510786, ‖∇f‖ = 3.9740e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1697.16 s: f = -0.668815529198, ‖∇f‖ = 2.8312e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1709.35 s: f = -0.668820072176, ‖∇f‖ = 1.8277e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1722.17 s: f = -0.668823045663, ‖∇f‖ = 2.1189e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1735.22 s: f = -0.668829224417, ‖∇f‖ = 3.3000e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1748.24 s: f = -0.668834661996, ‖∇f‖ = 3.3841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1761.10 s: f = -0.668839125190, ‖∇f‖ = 1.5494e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1774.64 s: f = -0.668842178047, ‖∇f‖ = 1.8074e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1788.22 s: f = -0.668846182434, ‖∇f‖ = 2.1704e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1801.30 s: f = -0.668849851199, ‖∇f‖ = 4.3895e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1814.28 s: f = -0.668854730352, ‖∇f‖ = 1.9014e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1827.19 s: f = -0.668857445978, ‖∇f‖ = 1.6213e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 1840.79 s: f = -0.668861083091, ‖∇f‖ = 2.1544e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1853.61 s: f = -0.668865914837, ‖∇f‖ = 2.4867e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 1866.53 s: f = -0.668869945971, ‖∇f‖ = 4.5125e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.6688699459708735

````

Note that for the specified parameters $J=\Delta=1$, we simulated the same Hamiltonian as
in the [Heisenberg example](@ref examples_heisenberg). In that example, with a non-symmetric
$D=2$ PEPS simulation, we reached a ground-state energy of around $E_\text{D=2} = -0.6625\dots$.
Again comparing against [Sandvik's](@cite sandvik_computational_2011) accurate QMC estimate
``E_{\text{ref}}=−0.6694421``, we see that we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

