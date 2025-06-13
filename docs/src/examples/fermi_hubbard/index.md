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
S_aux = S((1, 1))
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
physical_spaces = physicalspace(H)
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = +5.484842275412e+04 +4.469243203539e+04im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +8.371681846538e+04 -3.790928531089e-07im	err = 7.4963852436e-09	time = 0.86 sec

````

And third, we start the ground state search (this does take quite long):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 6.680719803101, ‖∇f‖ = 9.5849e+00
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{ProductSector{Tuple{FermionParity, U1Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{FermionParity, U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.49e-01, ϕ - ϕ₀ = -2.88e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  507.64 s: f = 3.801380487744, ‖∇f‖ = 2.3456e+01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{ProductSector{Tuple{FermionParity, U1Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{FermionParity, U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorMap{ComplexF64, GradedSpace{ProductSector{Tuple{FermionParity, U1Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{FermionParity, U1Irrep}}, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -5.73e-03, ϕ - ϕ₀ = -3.81e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  552.42 s: f = -0.009727650286, ‖∇f‖ = 3.2049e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  559.75 s: f = -0.115210826428, ‖∇f‖ = 2.7846e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  565.97 s: f = -0.616412169228, ‖∇f‖ = 2.3680e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, time  572.13 s: f = -0.817801148604, ‖∇f‖ = 1.9111e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  577.66 s: f = -0.990286615265, ‖∇f‖ = 2.3790e+00, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  583.15 s: f = -1.142787566798, ‖∇f‖ = 1.5680e+00, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  587.88 s: f = -1.238274330219, ‖∇f‖ = 3.5015e+00, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  592.68 s: f = -1.438136282421, ‖∇f‖ = 1.3366e+00, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  597.56 s: f = -1.523107107396, ‖∇f‖ = 1.3496e+00, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  607.69 s: f = -1.619305193101, ‖∇f‖ = 1.1951e+00, α = 1.72e-01, m = 9, nfg = 2
[ Info: LBFGS: iter   12, time  617.37 s: f = -1.681451834691, ‖∇f‖ = 9.4848e-01, α = 2.37e-01, m = 10, nfg = 2
[ Info: LBFGS: iter   13, time  622.15 s: f = -1.720734533825, ‖∇f‖ = 1.4216e+00, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  626.95 s: f = -1.770831967062, ‖∇f‖ = 6.2747e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  631.80 s: f = -1.807572162185, ‖∇f‖ = 5.1320e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  636.66 s: f = -1.859768355558, ‖∇f‖ = 7.1320e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time  641.44 s: f = -1.893160382125, ‖∇f‖ = 6.7323e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time  646.23 s: f = -1.923092489407, ‖∇f‖ = 5.5419e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  651.04 s: f = -1.948142544093, ‖∇f‖ = 4.7661e-01, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time  655.83 s: f = -1.969512080077, ‖∇f‖ = 4.1608e-01, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time  660.66 s: f = -1.982557838199, ‖∇f‖ = 4.5138e-01, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time  665.58 s: f = -1.994007805763, ‖∇f‖ = 3.1538e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  670.43 s: f = -2.002836016203, ‖∇f‖ = 3.0511e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  675.29 s: f = -2.014062852739, ‖∇f‖ = 3.3491e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  680.15 s: f = -2.022023251661, ‖∇f‖ = 4.3758e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  685.03 s: f = -2.030112566345, ‖∇f‖ = 2.0509e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  689.85 s: f = -2.035073683394, ‖∇f‖ = 1.6307e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  695.36 s: f = -2.038663850456, ‖∇f‖ = 1.6880e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  700.30 s: f = -2.041323592429, ‖∇f‖ = 2.4114e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  705.14 s: f = -2.044997390531, ‖∇f‖ = 1.2115e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  709.91 s: f = -2.046747469530, ‖∇f‖ = 9.5108e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  714.70 s: f = -2.048741416293, ‖∇f‖ = 1.0509e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  719.55 s: f = -2.049793769908, ‖∇f‖ = 1.7378e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  724.32 s: f = -2.051022900848, ‖∇f‖ = 6.4055e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  729.12 s: f = -2.051499900828, ‖∇f‖ = 4.9307e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  733.90 s: f = -2.051918795787, ‖∇f‖ = 6.2013e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  738.75 s: f = -2.052357188363, ‖∇f‖ = 9.4494e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  743.54 s: f = -2.052855317283, ‖∇f‖ = 4.8219e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  748.28 s: f = -2.053138284528, ‖∇f‖ = 3.5599e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  753.10 s: f = -2.053404037719, ‖∇f‖ = 4.1844e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  757.90 s: f = -2.053605747242, ‖∇f‖ = 5.7514e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  762.67 s: f = -2.053822345457, ‖∇f‖ = 3.1996e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  767.40 s: f = -2.054015631924, ‖∇f‖ = 3.1314e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  772.24 s: f = -2.054206835742, ‖∇f‖ = 4.1588e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  777.12 s: f = -2.054349141892, ‖∇f‖ = 6.7905e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  781.87 s: f = -2.054531571463, ‖∇f‖ = 2.9227e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  786.69 s: f = -2.054628027248, ‖∇f‖ = 2.5100e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  791.46 s: f = -2.054735541814, ‖∇f‖ = 3.1538e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  796.19 s: f = -2.054896782689, ‖∇f‖ = 3.4823e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  805.78 s: f = -2.055018285181, ‖∇f‖ = 5.2680e-02, α = 5.17e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   51, time  810.53 s: f = -2.055214629205, ‖∇f‖ = 3.0513e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  815.28 s: f = -2.055401907932, ‖∇f‖ = 2.8740e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  820.10 s: f = -2.055643036846, ‖∇f‖ = 4.1540e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  824.92 s: f = -2.055979753449, ‖∇f‖ = 6.0310e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  829.73 s: f = -2.056292876565, ‖∇f‖ = 6.4503e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  834.49 s: f = -2.056764405334, ‖∇f‖ = 4.5709e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  839.28 s: f = -2.057301128966, ‖∇f‖ = 5.8535e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  844.17 s: f = -2.057684443651, ‖∇f‖ = 7.0407e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  849.00 s: f = -2.058273607981, ‖∇f‖ = 6.4287e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  853.79 s: f = -2.058991887289, ‖∇f‖ = 8.8941e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  858.70 s: f = -2.059459011162, ‖∇f‖ = 1.1553e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  863.60 s: f = -2.060066395744, ‖∇f‖ = 6.9440e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  868.37 s: f = -2.060520108883, ‖∇f‖ = 8.4931e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  878.07 s: f = -2.060815447647, ‖∇f‖ = 1.2115e-01, α = 5.26e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   65, time  887.65 s: f = -2.060925751708, ‖∇f‖ = 8.3903e-02, α = 5.47e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   66, time  892.44 s: f = -2.061209735712, ‖∇f‖ = 5.4010e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  897.41 s: f = -2.061580165140, ‖∇f‖ = 5.5975e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  902.28 s: f = -2.062036980846, ‖∇f‖ = 7.8898e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  907.08 s: f = -2.062251708560, ‖∇f‖ = 1.1537e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  911.89 s: f = -2.062519627033, ‖∇f‖ = 1.2953e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  916.76 s: f = -2.063059957695, ‖∇f‖ = 7.2868e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  921.58 s: f = -2.063313170019, ‖∇f‖ = 5.2530e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  926.45 s: f = -2.063715486835, ‖∇f‖ = 5.0261e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  931.39 s: f = -2.064332648652, ‖∇f‖ = 7.7209e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  936.21 s: f = -2.064773368898, ‖∇f‖ = 1.2451e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  941.07 s: f = -2.065371921743, ‖∇f‖ = 6.8015e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time  945.89 s: f = -2.065945935292, ‖∇f‖ = 7.6664e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  950.75 s: f = -2.066640749519, ‖∇f‖ = 1.1191e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  955.60 s: f = -2.067648364662, ‖∇f‖ = 2.3836e-01, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 960.48 s: f = -2.069253148415, ‖∇f‖ = 2.0413e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -2.0692531484152648

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
(E - E_ref) / E_ref = -0.013540398520842128

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

