```@meta
EditURL = "../../../../examples/fermi_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//fermi_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//fermi_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//fermi_hubbard)


# Fermi-Hubbard model with $f\mathbb{Z}_2 \boxtimes U(1)$ symmetry, at large $U$ and half-filling

In this example, we will demonstrate how to handle fermionic PEPS tensors and how to
optimize them. To that end, we consider the two-dimensional Hubbard model

```math
H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- -
c_{i,\sigma}^- c_{j,\sigma}^+ \right) + U \sum_i n_{i,\uparrow}n_{i,\downarrow} - \mu \sum_i n_i
```

where $\sigma \in \{\uparrow,\downarrow\}$ and $n_{i,\sigma} = c_{i,\sigma}^+ c_{i,\sigma}^-$
is the fermionic number operator. As in previous examples, using fermionic degrees of freedom
is a matter of creating tensors with the right symmetry sectors - the rest of the simulation
workflow remains the same.

First though, we make the example deterministic by seeding the RNG, and we make our imports:

````julia
using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528937);
````

## Defining the fermionic Hamiltonian

Let us start by fixing the parameters of the Hubbard model. We're going to use a hopping of
$t=1$ and a large $U=8$ on a $2 \times 2$ unit cell:

````julia
t = 1.0
U = 8.0
lattice = InfiniteSquare(2, 2);
````

In order to create fermionic tensors, one needs to define symmetry sectors using TensorKit's
`FermionParity`. Not only do we want use fermion parity but we also want our
particles to exploit the global $U(1)$ symmetry. The combined product sector can be obtained
using the [Deligne product](https://jutho.github.io/TensorKit.jl/stable/lib/sectors/#TensorKitSectors.deligneproduct-Tuple{Sector,%20Sector}),
called through `⊠` which is obtained by typing `\boxtimes+TAB`. We will not impose any extra
spin symmetry, so we have:

````julia
fermion = fℤ₂
particle_symmetry = U1Irrep
spin_symmetry = Trivial
S = fermion ⊠ particle_symmetry
````

````
TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}
````

The next step is defining graded virtual PEPS and environment spaces using `S`. Here we also
use the symmetry sector to impose half-filling. That is all we need to define the Hubbard
Hamiltonian:

````julia
D, χ = 1, 1
V_peps = Vect[S]((0, 0) => 2 * D, (1, 1) => D, (1, -1) => D)
V_env = Vect[S](
    (0, 0) => 4 * χ, (1, -1) => 2 * χ, (1, 1) => 2 * χ, (0, 2) => χ, (0, -2) => χ
)
S_aux = S((1, -1))
H₀ = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, lattice; t, U)
H = add_physical_charge(H₀, fill(S_aux, size(H₀.lattice)...));
````

## Finding the ground state

Again, the procedure of ground state optimization is very similar to before. First, we
define all algorithmic parameters:

````julia
boundary_alg = (; tol=1e-8, alg=:simultaneous, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, maxiter=80, ls_maxiter=3, ls_maxfg=3)
````

````
(tol = 0.0001, alg = :lbfgs, maxiter = 80, ls_maxiter = 3, ls_maxfg = 3)
````

Second, we initialize a PEPS state and environment (which we converge) constructed from
symmetric physical and virtual spaces:

````julia
physical_spaces = H.lattice
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = +5.484842275412e+04 +4.469243203539e+04im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +8.371681846538e+04 -3.791428753175e-07im	err = 7.4963852327e-09	time = 8.54 sec

````

And third, we start the ground state search (this does take quite long):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 6.680719803101, ‖∇f‖ = 9.5842e+00
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}, TensorKit.SortedVectorDict{TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.49e-01, ϕ - ϕ₀ = -2.88e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  955.42 s: f = 3.801394787694, ‖∇f‖ = 2.3457e+01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}, TensorKit.SortedVectorDict{TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}, TensorKit.SortedVectorDict{TensorKitSectors.ProductSector{Tuple{TensorKitSectors.FermionParity, TensorKitSectors.U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -5.73e-03, ϕ - ϕ₀ = -3.81e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time 1011.68 s: f = -0.009753189324, ‖∇f‖ = 3.2047e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time 1020.91 s: f = -0.115219717423, ‖∇f‖ = 2.7847e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time 1029.61 s: f = -0.616462986123, ‖∇f‖ = 2.3685e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, time 1037.43 s: f = -0.817865359874, ‖∇f‖ = 1.9095e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time 1044.81 s: f = -0.990425015686, ‖∇f‖ = 2.3830e+00, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time 1051.68 s: f = -1.142986439459, ‖∇f‖ = 1.5684e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time 1058.34 s: f = -1.239591181120, ‖∇f‖ = 3.4861e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time 1064.29 s: f = -1.438708542563, ‖∇f‖ = 1.3377e+00, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time 1070.87 s: f = -1.524142766825, ‖∇f‖ = 1.3499e+00, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time 1083.72 s: f = -1.620143211649, ‖∇f‖ = 1.1928e+00, α = 1.75e-01, m = 9, nfg = 2
[ Info: LBFGS: iter   12, time 1096.24 s: f = -1.682030774949, ‖∇f‖ = 9.4585e-01, α = 2.41e-01, m = 10, nfg = 2
[ Info: LBFGS: iter   13, time 1102.22 s: f = -1.722173660258, ‖∇f‖ = 1.3961e+00, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time 1108.73 s: f = -1.771649839243, ‖∇f‖ = 6.2967e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time 1114.74 s: f = -1.809425620292, ‖∇f‖ = 5.1874e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time 1121.44 s: f = -1.860257660187, ‖∇f‖ = 7.0707e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time 1127.36 s: f = -1.894073433816, ‖∇f‖ = 6.7099e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time 1133.92 s: f = -1.923565778264, ‖∇f‖ = 5.6311e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time 1139.83 s: f = -1.948747056517, ‖∇f‖ = 4.7890e-01, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time 1146.40 s: f = -1.969585552903, ‖∇f‖ = 4.1660e-01, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time 1152.44 s: f = -1.982637358938, ‖∇f‖ = 4.3422e-01, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time 1159.01 s: f = -1.993882710416, ‖∇f‖ = 3.1362e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time 1165.05 s: f = -2.002938619798, ‖∇f‖ = 3.0798e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time 1171.70 s: f = -2.014146064233, ‖∇f‖ = 3.3262e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time 1177.90 s: f = -2.022239330954, ‖∇f‖ = 4.2937e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time 1184.55 s: f = -2.030245493641, ‖∇f‖ = 2.0179e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time 1190.57 s: f = -2.035169726141, ‖∇f‖ = 1.6346e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time 1198.05 s: f = -2.038915730445, ‖∇f‖ = 1.6570e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time 1204.18 s: f = -2.041961016975, ‖∇f‖ = 2.2790e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1210.81 s: f = -2.045467456219, ‖∇f‖ = 1.0966e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1216.81 s: f = -2.047243458561, ‖∇f‖ = 9.2405e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1223.40 s: f = -2.049202803483, ‖∇f‖ = 1.2184e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1229.55 s: f = -2.050191917638, ‖∇f‖ = 1.3044e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1236.28 s: f = -2.050986114708, ‖∇f‖ = 5.9665e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1242.29 s: f = -2.051548091457, ‖∇f‖ = 5.5253e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time 1248.94 s: f = -2.051993308206, ‖∇f‖ = 6.2588e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1254.93 s: f = -2.052324002624, ‖∇f‖ = 1.1928e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1261.55 s: f = -2.052936230102, ‖∇f‖ = 4.9216e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1267.62 s: f = -2.053164325823, ‖∇f‖ = 3.3410e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1274.63 s: f = -2.053418129203, ‖∇f‖ = 3.7314e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1281.04 s: f = -2.053649981748, ‖∇f‖ = 6.3612e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1288.03 s: f = -2.053879953203, ‖∇f‖ = 3.4038e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1294.19 s: f = -2.054050515673, ‖∇f‖ = 2.9152e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1300.90 s: f = -2.054259903099, ‖∇f‖ = 3.9095e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1307.08 s: f = -2.054388805929, ‖∇f‖ = 6.7475e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1313.89 s: f = -2.054563154978, ‖∇f‖ = 3.0486e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1320.10 s: f = -2.054666133101, ‖∇f‖ = 2.3929e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1327.11 s: f = -2.054764670097, ‖∇f‖ = 2.9961e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1333.39 s: f = -2.054936790198, ‖∇f‖ = 3.5407e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1345.97 s: f = -2.055058405443, ‖∇f‖ = 5.1106e-02, α = 5.17e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   51, time 1352.51 s: f = -2.055253894176, ‖∇f‖ = 3.1080e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1358.46 s: f = -2.055461219872, ‖∇f‖ = 2.9077e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1365.09 s: f = -2.055733194309, ‖∇f‖ = 4.5784e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1371.09 s: f = -2.055960164237, ‖∇f‖ = 7.1631e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1377.71 s: f = -2.056334000687, ‖∇f‖ = 5.1447e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1383.71 s: f = -2.056801416149, ‖∇f‖ = 4.8803e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1390.38 s: f = -2.057222872354, ‖∇f‖ = 5.7077e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1396.39 s: f = -2.057705132019, ‖∇f‖ = 8.5536e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1403.02 s: f = -2.058233824137, ‖∇f‖ = 6.6099e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1409.04 s: f = -2.058618411767, ‖∇f‖ = 8.2058e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1415.83 s: f = -2.058860905381, ‖∇f‖ = 8.8034e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1421.91 s: f = -2.059344181668, ‖∇f‖ = 6.7163e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1428.66 s: f = -2.059884025175, ‖∇f‖ = 1.1005e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1434.76 s: f = -2.060366638147, ‖∇f‖ = 7.3906e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1441.44 s: f = -2.060748895891, ‖∇f‖ = 5.7350e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1447.53 s: f = -2.061217694695, ‖∇f‖ = 1.0218e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1454.24 s: f = -2.061747836243, ‖∇f‖ = 6.5473e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1460.31 s: f = -2.061935488163, ‖∇f‖ = 7.7435e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1467.07 s: f = -2.062292588164, ‖∇f‖ = 1.1031e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1473.20 s: f = -2.062776748901, ‖∇f‖ = 8.7133e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1479.94 s: f = -2.063311285039, ‖∇f‖ = 6.3871e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1486.01 s: f = -2.063848732928, ‖∇f‖ = 7.8582e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1492.69 s: f = -2.064428066762, ‖∇f‖ = 8.0994e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1498.82 s: f = -2.064797991263, ‖∇f‖ = 2.1328e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1505.51 s: f = -2.065223995463, ‖∇f‖ = 1.0827e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1511.60 s: f = -2.065622191643, ‖∇f‖ = 7.4192e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1518.32 s: f = -2.066318151347, ‖∇f‖ = 9.4270e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1532.05 s: f = -2.067468309902, ‖∇f‖ = 1.5613e-01, α = 4.14e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   79, time 1546.45 s: f = -2.068248367388, ‖∇f‖ = 1.9031e-01, α = 2.81e-01, m = 20, nfg = 2
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 1558.17 s: f = -2.069293119751, ‖∇f‖ = 3.6413e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -2.0692931197508764

````

Finally, let's compare the obtained energy against a reference energy from a QMC study by
[Qin et al.](@cite qin_benchmark_2016). With the parameters specified above, they obtain an
energy of $E_\text{ref} \approx 4 \times -0.5244140625 = -2.09765625$ (the factor 4 comes
from the $2 \times 2$ unit cell that we use here). Thus, we find:

````julia
E_ref = -2.09765625
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -0.013521343284498413

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

