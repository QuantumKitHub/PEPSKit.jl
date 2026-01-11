```@meta
EditURL = "../../../../examples/fermi_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/fermi_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/fermi_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/fermi_hubbard)


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
S_aux = S((1, 1))
H₀ = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, lattice; t, U)
H = add_physical_charge(H₀, fill(S_aux, size(H₀.lattice)...));
````

## Finding the ground state

Again, the procedure of ground state optimization is very similar to before. First, we
define all algorithmic parameters:

````julia
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, trunc = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, alg = :eigsolver, maxiter = 10, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, maxiter = 80, ls_maxiter = 3, ls_maxfg = 3)
````

````
(tol = 0.0001, alg = :lbfgs, maxiter = 80, ls_maxiter = 3, ls_maxfg = 3)
````

Second, we initialize a PEPS state and environment (which we converge) constructed from
symmetric physical and virtual spaces:

````julia
physical_spaces = physicalspace(H)
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = +5.484842275412e+04 +4.469243203539e+04im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +8.371681846538e+04 -3.790437403950e-07im	err = 7.4963849845e-09	time = 15.96 sec

````

And third, we start the ground state search (this does take quite long):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity = 3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 6.680719803101e+00, ‖∇f‖ = 9.5851e+00
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.49e-01, ϕ - ϕ₀ = -2.88e+00
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/linesearches.jl:148
[ Info: LBFGS: iter    1, Δt  1.53 m: f = 3.801336895973e+00, ‖∇f‖ = 2.3457e+01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -5.73e-03, ϕ - ϕ₀ = -3.81e+00
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/linesearches.jl:148
[ Info: LBFGS: iter    2, Δt  1.37 m: f = -9.717028383144e-03, ‖∇f‖ = 3.2049e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, Δt 17.57 s: f = -1.151937236622e-01, ‖∇f‖ = 2.7846e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, Δt 17.11 s: f = -6.164097155293e-01, ‖∇f‖ = 2.3680e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, Δt 15.95 s: f = -8.177983978529e-01, ‖∇f‖ = 1.9112e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, Δt 15.19 s: f = -9.902797572194e-01, ‖∇f‖ = 2.3790e+00, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, Δt 14.17 s: f = -1.142781184740e+00, ‖∇f‖ = 1.5680e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, Δt 13.65 s: f = -1.238252408083e+00, ‖∇f‖ = 3.5020e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, Δt 12.48 s: f = -1.438152725373e+00, ‖∇f‖ = 1.3366e+00, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, Δt 13.47 s: f = -1.523106558123e+00, ‖∇f‖ = 1.3495e+00, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, Δt 26.36 s: f = -1.619309116769e+00, ‖∇f‖ = 1.1948e+00, α = 1.72e-01, m = 9, nfg = 2
[ Info: LBFGS: iter   12, Δt 26.04 s: f = -1.681436583910e+00, ‖∇f‖ = 9.4842e-01, α = 2.37e-01, m = 10, nfg = 2
[ Info: LBFGS: iter   13, Δt 12.48 s: f = -1.720664454158e+00, ‖∇f‖ = 1.4227e+00, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, Δt 12.26 s: f = -1.770786360300e+00, ‖∇f‖ = 6.2727e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, Δt 13.32 s: f = -1.807472248475e+00, ‖∇f‖ = 5.1285e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, Δt 12.57 s: f = -1.859749170859e+00, ‖∇f‖ = 7.1361e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, Δt 13.31 s: f = -1.893132064727e+00, ‖∇f‖ = 6.7317e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, Δt 12.54 s: f = -1.923092873621e+00, ‖∇f‖ = 5.5354e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt 12.31 s: f = -1.948135800861e+00, ‖∇f‖ = 4.7674e-01, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, Δt 13.62 s: f = -1.969521619354e+00, ‖∇f‖ = 4.1602e-01, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, Δt 13.66 s: f = -1.982569428626e+00, ‖∇f‖ = 4.5188e-01, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, Δt 12.60 s: f = -1.994023085799e+00, ‖∇f‖ = 3.1544e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt 13.62 s: f = -2.002841834328e+00, ‖∇f‖ = 3.0502e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt 12.82 s: f = -2.014066311349e+00, ‖∇f‖ = 3.3498e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt 12.94 s: f = -2.022003037531e+00, ‖∇f‖ = 4.3896e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt 13.76 s: f = -2.030108714915e+00, ‖∇f‖ = 2.0527e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt 12.75 s: f = -2.035064144013e+00, ‖∇f‖ = 1.6295e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt 15.33 s: f = -2.038644461742e+00, ‖∇f‖ = 1.6908e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt 12.77 s: f = -2.041287673888e+00, ‖∇f‖ = 2.4233e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt 13.59 s: f = -2.044963019661e+00, ‖∇f‖ = 1.2134e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt 12.74 s: f = -2.046709219209e+00, ‖∇f‖ = 9.5293e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt 13.57 s: f = -2.048704716271e+00, ‖∇f‖ = 1.0554e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt 12.57 s: f = -2.049753790375e+00, ‖∇f‖ = 1.7672e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt 13.56 s: f = -2.051012658206e+00, ‖∇f‖ = 6.4429e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt 12.58 s: f = -2.051487366864e+00, ‖∇f‖ = 4.8991e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt 13.58 s: f = -2.051906996297e+00, ‖∇f‖ = 6.2050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt 12.63 s: f = -2.052351425024e+00, ‖∇f‖ = 9.2730e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt 13.59 s: f = -2.052848309962e+00, ‖∇f‖ = 4.8571e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt 12.59 s: f = -2.053135862188e+00, ‖∇f‖ = 3.5616e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt 13.38 s: f = -2.053405790304e+00, ‖∇f‖ = 4.2302e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt 12.68 s: f = -2.053600752187e+00, ‖∇f‖ = 5.7965e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt 13.39 s: f = -2.053812277599e+00, ‖∇f‖ = 3.2230e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt 12.45 s: f = -2.054009905439e+00, ‖∇f‖ = 3.1640e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt 13.44 s: f = -2.054189832249e+00, ‖∇f‖ = 4.1575e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt 12.65 s: f = -2.054332729403e+00, ‖∇f‖ = 6.9193e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt 13.44 s: f = -2.054519398221e+00, ‖∇f‖ = 2.9113e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt 12.47 s: f = -2.054613030010e+00, ‖∇f‖ = 2.5330e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt 13.46 s: f = -2.054720911227e+00, ‖∇f‖ = 3.1755e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt 12.45 s: f = -2.054879191651e+00, ‖∇f‖ = 3.4648e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt 13.68 s: f = -2.054968269730e+00, ‖∇f‖ = 8.4873e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt 12.51 s: f = -2.055240587980e+00, ‖∇f‖ = 3.1534e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt 13.35 s: f = -2.055381123762e+00, ‖∇f‖ = 2.5668e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt 12.81 s: f = -2.055572801679e+00, ‖∇f‖ = 3.8027e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt 12.65 s: f = -2.055872564535e+00, ‖∇f‖ = 4.6489e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 13.70 s: f = -2.056396561541e+00, ‖∇f‖ = 8.8064e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, Δt 13.92 s: f = -2.056856024867e+00, ‖∇f‖ = 8.3599e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt 12.73 s: f = -2.057479287674e+00, ‖∇f‖ = 4.4470e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt 13.75 s: f = -2.057912193743e+00, ‖∇f‖ = 5.9314e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt 12.73 s: f = -2.058287076203e+00, ‖∇f‖ = 6.0139e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt 12.71 s: f = -2.058998629347e+00, ‖∇f‖ = 6.2208e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 26.47 s: f = -2.059475226949e+00, ‖∇f‖ = 1.0081e-01, α = 4.82e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   62, Δt 13.76 s: f = -2.060082547535e+00, ‖∇f‖ = 6.8334e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt 12.90 s: f = -2.060482651966e+00, ‖∇f‖ = 7.3285e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt 13.92 s: f = -2.060740773412e+00, ‖∇f‖ = 9.5341e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt 12.84 s: f = -2.061312903626e+00, ‖∇f‖ = 7.1673e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt 13.74 s: f = -2.061710661630e+00, ‖∇f‖ = 5.4950e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt 12.78 s: f = -2.062078845926e+00, ‖∇f‖ = 5.4629e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt 13.77 s: f = -2.062377274080e+00, ‖∇f‖ = 7.1202e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 26.56 s: f = -2.062699328045e+00, ‖∇f‖ = 9.7057e-02, α = 5.00e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   70, Δt 12.80 s: f = -2.063167668617e+00, ‖∇f‖ = 7.1650e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt 13.99 s: f = -2.063929597328e+00, ‖∇f‖ = 9.0355e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt 12.76 s: f = -2.064218059719e+00, ‖∇f‖ = 8.2741e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt 13.82 s: f = -2.064664984361e+00, ‖∇f‖ = 7.7230e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 12.90 s: f = -2.065239846433e+00, ‖∇f‖ = 1.0121e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt 13.86 s: f = -2.066014135860e+00, ‖∇f‖ = 9.7697e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt 12.98 s: f = -2.066932040862e+00, ‖∇f‖ = 1.6559e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt 15.54 s: f = -2.067203376711e+00, ‖∇f‖ = 3.9032e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 13.03 s: f = -2.067518198272e+00, ‖∇f‖ = 2.6538e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt 14.04 s: f = -2.069457237771e+00, ‖∇f‖ = 1.1802e-01, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 27.43 m: f = -2.071174488368e+00, ‖∇f‖ = 2.2576e-01
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/lbfgs.jl:199
E = -2.0711744883679684

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
(E - E_ref) / E_ref = -0.012624452472625886

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

