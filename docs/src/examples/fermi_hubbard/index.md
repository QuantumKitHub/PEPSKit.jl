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
gradient_alg = (; tol = 1.0e-6, alg = :eigsolver, maxiter = 10, iterscheme = :fixed)
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
[ Info: CTMRG conv 26:	obj = +8.371681846538e+04 -3.790119080804e-07im	err = 7.4963849914e-09	time = 36.38 sec

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
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/linesearches.jl:151
[ Info: LBFGS: iter    1, Δt 38.64 s: f = 3.801336895927e+00, ‖∇f‖ = 2.3457e+01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -5.73e-03, ϕ - ϕ₀ = -3.81e+00
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/linesearches.jl:151
[ Info: LBFGS: iter    2, Δt 33.51 s: f = -9.717027903025e-03, ‖∇f‖ = 3.2049e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, Δt  6.56 s: f = -1.151937230814e-01, ‖∇f‖ = 2.7846e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, Δt  7.11 s: f = -6.164097151643e-01, ‖∇f‖ = 2.3680e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, Δt  5.99 s: f = -8.177983968326e-01, ‖∇f‖ = 1.9112e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, Δt  5.39 s: f = -9.902797555067e-01, ‖∇f‖ = 2.3790e+00, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, Δt  6.47 s: f = -1.142781183014e+00, ‖∇f‖ = 1.5680e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, Δt  4.97 s: f = -1.238252399424e+00, ‖∇f‖ = 3.5020e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, Δt  5.92 s: f = -1.438152723955e+00, ‖∇f‖ = 1.3366e+00, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, Δt  4.83 s: f = -1.523106553003e+00, ‖∇f‖ = 1.3495e+00, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, Δt 11.09 s: f = -1.619309113033e+00, ‖∇f‖ = 1.1948e+00, α = 1.72e-01, m = 9, nfg = 2
[ Info: LBFGS: iter   12, Δt 10.75 s: f = -1.681436580003e+00, ‖∇f‖ = 9.4842e-01, α = 2.37e-01, m = 10, nfg = 2
[ Info: LBFGS: iter   13, Δt  4.81 s: f = -1.720664442091e+00, ‖∇f‖ = 1.4227e+00, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, Δt  5.91 s: f = -1.770786352947e+00, ‖∇f‖ = 6.2727e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, Δt  4.69 s: f = -1.807472232437e+00, ‖∇f‖ = 5.1285e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, Δt  6.02 s: f = -1.859749167278e+00, ‖∇f‖ = 7.1361e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, Δt  4.65 s: f = -1.893132058603e+00, ‖∇f‖ = 6.7317e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, Δt  4.84 s: f = -1.923092871787e+00, ‖∇f‖ = 5.5354e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt  5.89 s: f = -1.948135797818e+00, ‖∇f‖ = 4.7674e-01, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, Δt  4.83 s: f = -1.969521620203e+00, ‖∇f‖ = 4.1602e-01, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, Δt  6.12 s: f = -1.982569429583e+00, ‖∇f‖ = 4.5188e-01, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, Δt  4.79 s: f = -1.994023088009e+00, ‖∇f‖ = 3.1544e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt  6.09 s: f = -2.002841835268e+00, ‖∇f‖ = 3.0502e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt  4.93 s: f = -2.014066311538e+00, ‖∇f‖ = 3.3498e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt  5.04 s: f = -2.022003034919e+00, ‖∇f‖ = 4.3896e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt  5.98 s: f = -2.030108714405e+00, ‖∇f‖ = 2.0527e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt  4.94 s: f = -2.035064142878e+00, ‖∇f‖ = 1.6295e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt  6.60 s: f = -2.038644459020e+00, ‖∇f‖ = 1.6908e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt  4.92 s: f = -2.041287669506e+00, ‖∇f‖ = 2.4233e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt  5.95 s: f = -2.044963015191e+00, ‖∇f‖ = 1.2134e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt  4.83 s: f = -2.046709214218e+00, ‖∇f‖ = 9.5293e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt  6.05 s: f = -2.048704711627e+00, ‖∇f‖ = 1.0554e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt  4.78 s: f = -2.049753785745e+00, ‖∇f‖ = 1.7672e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt  5.96 s: f = -2.051012657058e+00, ‖∇f‖ = 6.4429e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt  4.80 s: f = -2.051487365494e+00, ‖∇f‖ = 4.8991e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt  4.82 s: f = -2.051906995005e+00, ‖∇f‖ = 6.2050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt  6.04 s: f = -2.052351424092e+00, ‖∇f‖ = 9.2730e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt  4.80 s: f = -2.052848308926e+00, ‖∇f‖ = 4.8571e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt  5.93 s: f = -2.053135861688e+00, ‖∇f‖ = 3.5616e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt  4.80 s: f = -2.053405790277e+00, ‖∇f‖ = 4.2303e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt  4.88 s: f = -2.053600751472e+00, ‖∇f‖ = 5.7965e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt  5.95 s: f = -2.053812276343e+00, ‖∇f‖ = 3.2230e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt  4.68 s: f = -2.054009904322e+00, ‖∇f‖ = 3.1640e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt  4.74 s: f = -2.054189829937e+00, ‖∇f‖ = 4.1575e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt  5.93 s: f = -2.054332727237e+00, ‖∇f‖ = 6.9194e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt  4.71 s: f = -2.054519396790e+00, ‖∇f‖ = 2.9113e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt  5.92 s: f = -2.054613028290e+00, ‖∇f‖ = 2.5330e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt  4.69 s: f = -2.054720909675e+00, ‖∇f‖ = 3.1755e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt  5.72 s: f = -2.054879189481e+00, ‖∇f‖ = 3.4648e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt  4.90 s: f = -2.054968277255e+00, ‖∇f‖ = 8.4871e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt  4.71 s: f = -2.055240591253e+00, ‖∇f‖ = 3.1534e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt  5.92 s: f = -2.055381130477e+00, ‖∇f‖ = 2.5669e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt  4.75 s: f = -2.055572809496e+00, ‖∇f‖ = 3.8027e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt  4.88 s: f = -2.055872578414e+00, ‖∇f‖ = 4.6489e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt  6.02 s: f = -2.056396546910e+00, ‖∇f‖ = 8.8069e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, Δt  4.94 s: f = -2.056856099700e+00, ‖∇f‖ = 8.3587e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt  6.07 s: f = -2.057479296853e+00, ‖∇f‖ = 4.4471e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt  4.81 s: f = -2.057912211557e+00, ‖∇f‖ = 5.9322e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt  6.08 s: f = -2.058287106374e+00, ‖∇f‖ = 6.0138e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt  4.78 s: f = -2.058998688168e+00, ‖∇f‖ = 6.2214e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 10.81 s: f = -2.059475430386e+00, ‖∇f‖ = 1.0083e-01, α = 4.82e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   62, Δt  4.85 s: f = -2.060082712314e+00, ‖∇f‖ = 6.8338e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt  6.18 s: f = -2.060482611701e+00, ‖∇f‖ = 7.3298e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt  4.92 s: f = -2.060741140972e+00, ‖∇f‖ = 9.5443e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt  4.88 s: f = -2.061312696274e+00, ‖∇f‖ = 7.1658e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt  5.97 s: f = -2.061709741165e+00, ‖∇f‖ = 5.4940e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt  4.87 s: f = -2.062078223556e+00, ‖∇f‖ = 5.4620e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt  6.00 s: f = -2.062377140676e+00, ‖∇f‖ = 7.1416e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 10.93 s: f = -2.062698062544e+00, ‖∇f‖ = 9.6845e-02, α = 5.00e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   70, Δt  5.04 s: f = -2.063163727897e+00, ‖∇f‖ = 7.1580e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt  6.13 s: f = -2.063925627174e+00, ‖∇f‖ = 9.0915e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt  5.02 s: f = -2.064219513055e+00, ‖∇f‖ = 8.0698e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt  4.91 s: f = -2.064673250535e+00, ‖∇f‖ = 7.5326e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt  6.12 s: f = -2.065226882556e+00, ‖∇f‖ = 1.0638e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt  4.85 s: f = -2.066059171824e+00, ‖∇f‖ = 9.5076e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt  4.99 s: f = -2.067195111916e+00, ‖∇f‖ = 1.5006e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt  5.94 s: f = -2.068289278812e+00, ‖∇f‖ = 2.3385e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 16.76 s: f = -2.068482519585e+00, ‖∇f‖ = 6.6865e-01, α = 1.56e-01, m = 20, nfg = 3
[ Info: LBFGS: iter   79, Δt 12.04 s: f = -2.068666326649e+00, ‖∇f‖ = 3.3508e-01, α = 2.43e-01, m = 20, nfg = 2
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 21.90 m: f = -2.069886404460e+00, ‖∇f‖ = 2.6573e-01
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/lbfgs.jl:199
E = -2.069886404460094

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
(E - E_ref) / E_ref = -0.013238511095374089

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

