```@meta
EditURL = "../../../../examples/5.fermi_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//5.fermi_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//5.fermi_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//5.fermi_hubbard)


# Fermi-Hubbard model with $f\mathbb{Z}_2 \boxtimes U(1)$ symmetry, at large $U$ and half-filling

In this example, we will demonstrate how to handle fermionic PEPS tensors and how to
optimize them. To that end, we consider the two-dimensional Hubbard model

```math
H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- +
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
ProductSector{Tuple{FermionParity, U1Irrep}}
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
[ Info: CTMRG init:	obj = +1.089691795517e+05 -1.031847476500e+05im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +8.359116284442e+04 -2.209308149759e-07im	err = 7.4963852327e-09	time = 1.72 sec

````

And third, we start the ground state search (this does take quite long):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 6.680719803101, ‖∇f‖ = 9.5851e+00
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{ProductSector{Tuple{FermionParity, U1Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{FermionParity, U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.49e-01, ϕ - ϕ₀ = -2.88e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time   71.29 s: f = 3.801336885086, ‖∇f‖ = 2.3457e+01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{ProductSector{Tuple{FermionParity, U1Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{FermionParity, U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorMap{ComplexF64, GradedSpace{ProductSector{Tuple{FermionParity, U1Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{FermionParity, U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -5.73e-03, ϕ - ϕ₀ = -3.81e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  122.63 s: f = -0.009717029459, ‖∇f‖ = 3.2049e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  134.34 s: f = -0.115193722670, ‖∇f‖ = 2.7846e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  145.23 s: f = -0.616409716851, ‖∇f‖ = 2.3680e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, time  156.06 s: f = -0.817798399433, ‖∇f‖ = 1.9112e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  166.77 s: f = -0.990279759337, ‖∇f‖ = 2.3790e+00, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  176.62 s: f = -1.142781186573, ‖∇f‖ = 1.5680e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  185.65 s: f = -1.238252443477, ‖∇f‖ = 3.5020e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  194.43 s: f = -1.438152734653, ‖∇f‖ = 1.3366e+00, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  204.31 s: f = -1.523106583487, ‖∇f‖ = 1.3495e+00, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  222.04 s: f = -1.619309135102, ‖∇f‖ = 1.1948e+00, α = 1.72e-01, m = 9, nfg = 2
[ Info: LBFGS: iter   12, time  240.81 s: f = -1.681436596876, ‖∇f‖ = 9.4842e-01, α = 2.37e-01, m = 10, nfg = 2
[ Info: LBFGS: iter   13, time  249.75 s: f = -1.720664496573, ‖∇f‖ = 1.4227e+00, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  258.70 s: f = -1.770786384177, ‖∇f‖ = 6.2727e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  267.73 s: f = -1.807472303371, ‖∇f‖ = 5.1285e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  277.56 s: f = -1.859749181748, ‖∇f‖ = 7.1361e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time  286.53 s: f = -1.893132087361, ‖∇f‖ = 6.7317e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time  295.80 s: f = -1.923092881224, ‖∇f‖ = 5.5354e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  304.76 s: f = -1.948135813135, ‖∇f‖ = 4.7674e-01, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time  314.57 s: f = -1.969521615479, ‖∇f‖ = 4.1602e-01, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time  323.68 s: f = -1.982569425643, ‖∇f‖ = 4.5188e-01, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time  332.65 s: f = -1.994023077610, ‖∇f‖ = 3.1544e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  341.71 s: f = -2.002841830905, ‖∇f‖ = 3.0502e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  351.59 s: f = -2.014066310582, ‖∇f‖ = 3.3498e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  360.71 s: f = -2.022003043413, ‖∇f‖ = 4.3896e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  369.80 s: f = -2.030108717392, ‖∇f‖ = 2.0527e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  378.93 s: f = -2.035064147211, ‖∇f‖ = 1.6295e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  389.67 s: f = -2.038644470340, ‖∇f‖ = 1.6908e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  398.79 s: f = -2.041287690928, ‖∇f‖ = 2.4233e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  407.81 s: f = -2.044963035952, ‖∇f‖ = 1.2134e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  416.86 s: f = -2.046709236508, ‖∇f‖ = 9.5293e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  426.74 s: f = -2.048704733827, ‖∇f‖ = 1.0554e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  435.83 s: f = -2.049753805875, ‖∇f‖ = 1.7672e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  444.84 s: f = -2.051012660993, ‖∇f‖ = 6.4429e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  453.91 s: f = -2.051487370981, ‖∇f‖ = 4.8991e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  463.83 s: f = -2.051906999941, ‖∇f‖ = 6.2050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  472.88 s: f = -2.052351426534, ‖∇f‖ = 9.2731e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  481.96 s: f = -2.052848312522, ‖∇f‖ = 4.8571e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  490.98 s: f = -2.053135862679, ‖∇f‖ = 3.5616e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  500.79 s: f = -2.053405789539, ‖∇f‖ = 4.2302e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  509.87 s: f = -2.053600753566, ‖∇f‖ = 5.7965e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  518.97 s: f = -2.053812280854, ‖∇f‖ = 3.2230e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  528.15 s: f = -2.054009907356, ‖∇f‖ = 3.1640e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  538.31 s: f = -2.054189837590, ‖∇f‖ = 4.1575e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  547.45 s: f = -2.054332733432, ‖∇f‖ = 6.9193e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  556.49 s: f = -2.054519400845, ‖∇f‖ = 2.9113e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  565.56 s: f = -2.054613033662, ‖∇f‖ = 2.5330e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  575.48 s: f = -2.054720913698, ‖∇f‖ = 3.1755e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  584.57 s: f = -2.054879195070, ‖∇f‖ = 3.4648e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  593.69 s: f = -2.054968252492, ‖∇f‖ = 8.4876e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  602.62 s: f = -2.055240579153, ‖∇f‖ = 3.1534e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  612.56 s: f = -2.055381107398, ‖∇f‖ = 2.5668e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  621.69 s: f = -2.055572782456, ‖∇f‖ = 3.8027e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  630.87 s: f = -2.055872532380, ‖∇f‖ = 4.6489e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  639.99 s: f = -2.056396587816, ‖∇f‖ = 8.8052e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  650.21 s: f = -2.056855859558, ‖∇f‖ = 8.3624e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  659.33 s: f = -2.057479262333, ‖∇f‖ = 4.4470e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  668.49 s: f = -2.057912152061, ‖∇f‖ = 5.9296e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  677.62 s: f = -2.058287008863, ‖∇f‖ = 6.0141e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  687.77 s: f = -2.058998497280, ‖∇f‖ = 6.2194e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  706.16 s: f = -2.059474789738, ‖∇f‖ = 1.0077e-01, α = 4.81e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   62, time  715.09 s: f = -2.060082192514, ‖∇f‖ = 6.8325e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  725.00 s: f = -2.060482721863, ‖∇f‖ = 7.3259e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  734.39 s: f = -2.060740013280, ‖∇f‖ = 9.5135e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  743.57 s: f = -2.061313322733, ‖∇f‖ = 7.1706e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  752.71 s: f = -2.061712546612, ‖∇f‖ = 5.4970e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  762.80 s: f = -2.062080105542, ‖∇f‖ = 5.4651e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  771.90 s: f = -2.062377423092, ‖∇f‖ = 7.0780e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  790.05 s: f = -2.062702057386, ‖∇f‖ = 9.7500e-02, α = 5.01e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   70, time  799.85 s: f = -2.063176089747, ‖∇f‖ = 7.1783e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  808.93 s: f = -2.063937979922, ‖∇f‖ = 8.8528e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  817.94 s: f = -2.064211612845, ‖∇f‖ = 8.9468e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  826.94 s: f = -2.064625801997, ‖∇f‖ = 8.5359e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  836.93 s: f = -2.065216197065, ‖∇f‖ = 8.9020e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  846.12 s: f = -2.065871494784, ‖∇f‖ = 1.0651e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  855.26 s: f = -2.066880037398, ‖∇f‖ = 1.4447e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time  886.46 s: f = -2.067848102521, ‖∇f‖ = 2.0648e-01, α = 5.31e-01, m = 20, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 4 function evaluations:
│ α = 0.00e+00, dϕ = -9.88e-03, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   78, time  925.19 s: f = -2.067848102521, ‖∇f‖ = 2.0648e-01, α = 0.00e+00, m = 20, nfg = 4
┌ Warning: Linesearch not converged after 2 iterations and 4 function evaluations:
│ α = 0.00e+00, dϕ = -9.88e-03, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   79, time  963.83 s: f = -2.067848102521, ‖∇f‖ = 2.0648e-01, α = 0.00e+00, m = 20, nfg = 4
┌ Warning: Linesearch not converged after 2 iterations and 4 function evaluations:
│ α = 0.00e+00, dϕ = -9.88e-03, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 1002.56 s: f = -2.067848102521, ‖∇f‖ = 2.0648e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -2.0678481025209448

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
(E - E_ref) / E_ref = -0.01421021555798537

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

