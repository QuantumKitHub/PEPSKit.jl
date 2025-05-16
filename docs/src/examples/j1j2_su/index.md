```@meta
EditURL = "../../../../examples/j1j2_su/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/j1j2_su/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/j1j2_su/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/j1j2_su)


# Three-site simple update for the $J_1$-$J_2$ model

In this example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional $J_1$-$J_2$ model, which contains next-nearest neighbor interactions:

```math
H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
+ J_2 \sum_{\langle \langle i,j \rangle \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
```

Note that the $J_1$-$J_2$ model exhibits a $U(1)$ spin rotation symmetry, which we want to
exploit here. The goal will be to calculate the energy at $J_1 = 1$ and $J_2 = 1/2$, first
using the simple update algorithm and then, to refine the energy estimate, using AD-based
variational PEPS optimization.

We first import all required modules and seed the RNG:

````julia
using Random
using TensorKit, PEPSKit
Random.seed!(2025);
````

## Simple updating a challenging phase

Let's start by initializing an `InfiniteWeightPEPS` for which we set the required
parameters as well as physical and virtual vector spaces. Since the $J_1$-$J_2$ model has
*next*-neighbor interactions, the simple update algorithm requires a $2 \times 2$ unit cell:

````julia
Dbond, χenv, symm = 4, 32, U1Irrep
trscheme_env = truncerr(1e-10) & truncdim(χenv)
Nr, Nc, J1 = 2, 2, 1.0

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Pspace = Vect[U1Irrep](1//2 => 1, -1//2 => 1)
Vspace = Vect[U1Irrep](0 => 2, 1//2 => 1, -1//2 => 1)
Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1//2 => χenv ÷ 4, -1//2 => χenv ÷ 4)
wpeps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc));
````

The value $J_2 / J_1 = 0.5$ is close to a possible spin liquid phase, which is challenging
for SU to produce a relatively good state from random initialization. Therefore, we shall
gradually increase $J_2 / J_1$ from 0.1 to 0.5, each time initializing on the previously
evolved PEPS:

````julia
dt, tol, maxiter = 1e-2, 1e-8, 30000
check_interval = 4000
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)
alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
for J2 in 0.1:0.1:0.5
    H = real( ## convert Hamiltonian `LocalOperator` to real floats
        j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false),
    )
    result = simpleupdate(wpeps, H, alg; check_interval)
    global wpeps = result[1]
end
````

````
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1/2=>1, -1/2=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.199e+00,  time = 22.657 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1578   :  dt = 1e-02,  weight diff = 9.954e-09,  time = 56.562 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.409e-04,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 992    :  dt = 1e-02,  weight diff = 9.982e-09,  time = 16.169 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.449e-04,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1199   :  dt = 1e-02,  weight diff = 9.946e-09,  time = 18.754 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.613e-04,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1527   :  dt = 1e-02,  weight diff = 9.989e-09,  time = 24.056 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.830e-04,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 2650   :  dt = 1e-02,  weight diff = 1.000e-08,  time = 41.106 sec

````

After we reach $J_2 / J_1 = 0.5$, we gradually decrease the evolution time step to obtain
a more accurately evolved PEPS:

````julia
dts = [1e-3, 1e-4]
tols = [1e-9, 1e-9]
J2 = 0.5
H = real(j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false))
for (dt, tol) in zip(dts, tols)
    alg′ = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(wpeps, H, alg′; check_interval)
    global wpeps = result[1]
end
````

````
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 4.232e-04,  time = 0.016 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 4000   :  dt = 1e-03,  weight diff = 2.409e-08,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 8000   :  dt = 1e-03,  weight diff = 7.222e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 12000  :  dt = 1e-03,  weight diff = 4.550e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 16000  :  dt = 1e-03,  weight diff = 3.389e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 20000  :  dt = 1e-03,  weight diff = 2.661e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 24000  :  dt = 1e-03,  weight diff = 2.133e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 28000  :  dt = 1e-03,  weight diff = 1.725e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
┌ Warning: SU cancel 30000  :  dt = 1e-03,  weight diff = 1.555e-09,  time = 467.515 sec
└ @ PEPSKit ~/repos/foreign-forks/yue-zhengyuan/PEPSKit.jl/src/algorithms/time_evolution/simpleupdate3site.jl:460
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 4.218e-05,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 4000   :  dt = 1e-04,  weight diff = 1.858e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 8000   :  dt = 1e-04,  weight diff = 1.407e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 12000  :  dt = 1e-04,  weight diff = 1.095e-09,  time = 0.015 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 13532  :  dt = 1e-04,  weight diff = 1.000e-09,  time = 212.300 sec

````

## Computing the simple update energy estimate

Finally, we measure the ground-state energy by converging a CTMRG environment and computing
the expectation value, where we make sure to normalize by the unit cell size:

````julia
peps = InfinitePEPS(wpeps)
normalize!.(peps.A, Inf) ## normalize PEPS with absorbed weights by largest element
env₀ = CTMRGEnv(rand, Float64, peps, Espace)
env, = leading_boundary(env₀, peps; tol=1e-10, alg=:sequential, trscheme=trscheme_env);
E = expectation_value(peps, H, env) / (Nr * Nc)
````

````
-0.49123916142030766
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.006091732078285008

````

## Variational PEPS optimization using AD

As a last step, we will use the SU-evolved PEPS as a starting point for a [`fixedpoint`](@ref)
PEPS optimization. Note that we could have also used a sublattice-rotated version of `H` to
fit the Hamiltonian onto a single-site unit cell which would require us to optimize fewer
parameters and hence lead to a faster optimization. But here we instead take advantage of
the already evolved `peps`, thus giving us a physical initial guess for the optimization:

````julia
peps_opt, env_opt, E_opt, = fixedpoint(
    H, peps, env; optimizer_alg=(; tol=1e-4, maxiter=120)
);
````

````
┌ Warning: the provided real environment was converted to a complex environment since :fixed mode generally produces complex gauges; use :diffgauge mode instead by passing gradient_alg=(; iterscheme=:diffgauge) to the fixedpoint keyword arguments to work with purely real environments
└ @ PEPSKit ~/.julia/packages/PEPSKit/2FBaz/src/algorithms/optimization/peps_optimization.jl:217
[ Info: LBFGS: initializing with f = -1.964948266316, ‖∇f‖ = 2.7544e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{Float64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{Float64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time  724.70 s: f = -1.967094178478, ‖∇f‖ = 1.6146e-01, α = 1.00e+00, m = 0, nfg = 1
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{Float64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{Float64}}}, InfinitePEPS{TensorKit.TensorMap{Float64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{Float64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time  738.59 s: f = -1.968859096094, ‖∇f‖ = 1.0162e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  749.28 s: f = -1.969798623080, ‖∇f‖ = 1.0699e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  760.03 s: f = -1.971722717370, ‖∇f‖ = 7.0920e-02, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  770.31 s: f = -1.973042342743, ‖∇f‖ = 6.0516e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  780.96 s: f = -1.974999288324, ‖∇f‖ = 5.3851e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  792.46 s: f = -1.975497427652, ‖∇f‖ = 7.4920e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  804.36 s: f = -1.975955307302, ‖∇f‖ = 2.9377e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  815.01 s: f = -1.976079539455, ‖∇f‖ = 2.4072e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  825.67 s: f = -1.976518150784, ‖∇f‖ = 3.8542e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  836.62 s: f = -1.976781345187, ‖∇f‖ = 3.5505e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  847.51 s: f = -1.977266474894, ‖∇f‖ = 2.3062e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  858.13 s: f = -1.977665959481, ‖∇f‖ = 2.2747e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  868.72 s: f = -1.977899382423, ‖∇f‖ = 2.0319e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  891.24 s: f = -1.977968121188, ‖∇f‖ = 2.6887e-02, α = 3.87e-01, m = 14, nfg = 2
[ Info: LBFGS: iter   16, time  902.52 s: f = -1.978071515623, ‖∇f‖ = 1.4041e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  912.89 s: f = -1.978158139412, ‖∇f‖ = 1.3544e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  923.58 s: f = -1.978302555160, ‖∇f‖ = 1.9631e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  934.32 s: f = -1.978422371499, ‖∇f‖ = 1.7767e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  958.35 s: f = -1.978458803368, ‖∇f‖ = 2.0975e-02, α = 2.75e-01, m = 19, nfg = 2
[ Info: LBFGS: iter   21, time  969.49 s: f = -1.978530077543, ‖∇f‖ = 9.3603e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  979.90 s: f = -1.978560658816, ‖∇f‖ = 8.7437e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  990.27 s: f = -1.978614704552, ‖∇f‖ = 1.3827e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time 1000.72 s: f = -1.978644893660, ‖∇f‖ = 2.9989e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time 1011.09 s: f = -1.978710378854, ‖∇f‖ = 1.2197e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time 1021.64 s: f = -1.978755636561, ‖∇f‖ = 9.9248e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time 1032.37 s: f = -1.978810592618, ‖∇f‖ = 1.0197e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time 1044.07 s: f = -1.978880652312, ‖∇f‖ = 1.3230e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time 1065.86 s: f = -1.978893546398, ‖∇f‖ = 1.2667e-02, α = 2.02e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   30, time 1076.50 s: f = -1.978913379688, ‖∇f‖ = 8.4297e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1087.29 s: f = -1.978937871229, ‖∇f‖ = 7.3072e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1097.84 s: f = -1.978961659856, ‖∇f‖ = 8.2989e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1108.88 s: f = -1.978995792999, ‖∇f‖ = 9.0362e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1131.27 s: f = -1.979015006562, ‖∇f‖ = 9.8342e-03, α = 3.54e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   35, time 1142.65 s: f = -1.979038167117, ‖∇f‖ = 5.2968e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time 1152.99 s: f = -1.979055327917, ‖∇f‖ = 5.1642e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1163.08 s: f = -1.979067394612, ‖∇f‖ = 5.6902e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1174.11 s: f = -1.979081709842, ‖∇f‖ = 1.2188e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1186.03 s: f = -1.979101893328, ‖∇f‖ = 6.0559e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1195.99 s: f = -1.979120148967, ‖∇f‖ = 5.6346e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1206.44 s: f = -1.979132668831, ‖∇f‖ = 5.1812e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1218.28 s: f = -1.979141653518, ‖∇f‖ = 8.3678e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1229.35 s: f = -1.979152302292, ‖∇f‖ = 3.6027e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1239.99 s: f = -1.979157388674, ‖∇f‖ = 3.5709e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1250.66 s: f = -1.979168452908, ‖∇f‖ = 6.1928e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1272.78 s: f = -1.979171015212, ‖∇f‖ = 5.1956e-03, α = 3.01e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   47, time 1283.42 s: f = -1.979174715142, ‖∇f‖ = 3.0583e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1293.94 s: f = -1.979177809683, ‖∇f‖ = 2.5589e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1304.48 s: f = -1.979181346404, ‖∇f‖ = 2.8899e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1315.50 s: f = -1.979188173098, ‖∇f‖ = 5.6958e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1327.08 s: f = -1.979189848801, ‖∇f‖ = 9.5441e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1339.60 s: f = -1.979198641934, ‖∇f‖ = 2.6808e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1349.73 s: f = -1.979202022575, ‖∇f‖ = 2.7911e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1360.67 s: f = -1.979206039617, ‖∇f‖ = 2.8137e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1371.92 s: f = -1.979211331758, ‖∇f‖ = 5.3204e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1383.43 s: f = -1.979214445346, ‖∇f‖ = 2.2704e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1393.89 s: f = -1.979217057782, ‖∇f‖ = 2.1415e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1404.15 s: f = -1.979219878319, ‖∇f‖ = 2.2147e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1415.23 s: f = -1.979221418569, ‖∇f‖ = 4.9353e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1426.28 s: f = -1.979224572662, ‖∇f‖ = 1.9793e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1437.09 s: f = -1.979226628395, ‖∇f‖ = 1.9276e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1447.92 s: f = -1.979228741214, ‖∇f‖ = 2.9813e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1459.09 s: f = -1.979233238950, ‖∇f‖ = 3.8677e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1483.46 s: f = -1.979234527338, ‖∇f‖ = 5.6487e-03, α = 2.37e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   65, time 1494.35 s: f = -1.979238796139, ‖∇f‖ = 3.5747e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1505.17 s: f = -1.979241480852, ‖∇f‖ = 1.7941e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1516.78 s: f = -1.979243418537, ‖∇f‖ = 2.1588e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1527.05 s: f = -1.979244226408, ‖∇f‖ = 5.7334e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1537.38 s: f = -1.979245554250, ‖∇f‖ = 2.8494e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1548.12 s: f = -1.979247297213, ‖∇f‖ = 2.2530e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1559.64 s: f = -1.979248542090, ‖∇f‖ = 2.7993e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1570.80 s: f = -1.979250527636, ‖∇f‖ = 2.7448e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1594.51 s: f = -1.979251555847, ‖∇f‖ = 4.2226e-03, α = 2.48e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   74, time 1605.53 s: f = -1.979253338594, ‖∇f‖ = 1.7576e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1616.22 s: f = -1.979254044161, ‖∇f‖ = 1.4733e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1627.69 s: f = -1.979254516565, ‖∇f‖ = 2.9201e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1639.30 s: f = -1.979255324831, ‖∇f‖ = 1.5754e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1650.47 s: f = -1.979255931167, ‖∇f‖ = 1.4803e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1661.76 s: f = -1.979257230141, ‖∇f‖ = 2.3934e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1672.90 s: f = -1.979258597632, ‖∇f‖ = 2.6436e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1685.08 s: f = -1.979261009407, ‖∇f‖ = 3.1295e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1708.25 s: f = -1.979262065020, ‖∇f‖ = 1.9068e-03, α = 4.70e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   83, time 1719.18 s: f = -1.979262946329, ‖∇f‖ = 1.1711e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1729.94 s: f = -1.979263911927, ‖∇f‖ = 1.5041e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time 1740.69 s: f = -1.979264838247, ‖∇f‖ = 2.1471e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time 1751.60 s: f = -1.979266245522, ‖∇f‖ = 2.8599e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time 1762.85 s: f = -1.979267819630, ‖∇f‖ = 1.9260e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time 1774.01 s: f = -1.979268860509, ‖∇f‖ = 1.5330e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time 1785.51 s: f = -1.979270333367, ‖∇f‖ = 1.6250e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time 1795.81 s: f = -1.979272198125, ‖∇f‖ = 3.3875e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time 1806.51 s: f = -1.979272680119, ‖∇f‖ = 7.0506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time 1817.60 s: f = -1.979274442801, ‖∇f‖ = 2.1449e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time 1828.19 s: f = -1.979275003406, ‖∇f‖ = 1.3937e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time 1838.71 s: f = -1.979275812667, ‖∇f‖ = 1.8402e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time 1850.43 s: f = -1.979276954803, ‖∇f‖ = 2.0759e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time 1871.78 s: f = -1.979277811697, ‖∇f‖ = 3.9184e-03, α = 3.67e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   97, time 1882.69 s: f = -1.979279496101, ‖∇f‖ = 1.8019e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time 1893.23 s: f = -1.979280624929, ‖∇f‖ = 1.4407e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time 1903.82 s: f = -1.979282048752, ‖∇f‖ = 1.7032e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time 1915.12 s: f = -1.979284090238, ‖∇f‖ = 1.8831e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  101, time 1926.01 s: f = -1.979286455731, ‖∇f‖ = 5.4273e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time 1936.56 s: f = -1.979289543364, ‖∇f‖ = 1.7358e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time 1947.19 s: f = -1.979290658748, ‖∇f‖ = 1.4085e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time 1958.62 s: f = -1.979292652664, ‖∇f‖ = 1.9843e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time 1969.26 s: f = -1.979293575951, ‖∇f‖ = 4.1535e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time 1980.24 s: f = -1.979295639890, ‖∇f‖ = 1.3717e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time 1990.91 s: f = -1.979296676915, ‖∇f‖ = 1.1578e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, time 2001.90 s: f = -1.979297917147, ‖∇f‖ = 1.4455e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time 2013.96 s: f = -1.979299557413, ‖∇f‖ = 2.1745e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time 2025.36 s: f = -1.979301678432, ‖∇f‖ = 2.4901e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time 2036.44 s: f = -1.979302765791, ‖∇f‖ = 4.8999e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time 2048.34 s: f = -1.979305521027, ‖∇f‖ = 3.0184e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time 2058.39 s: f = -1.979307125661, ‖∇f‖ = 2.4605e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time 2069.65 s: f = -1.979309110899, ‖∇f‖ = 3.0368e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time 2081.18 s: f = -1.979311751813, ‖∇f‖ = 2.6611e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time 2093.75 s: f = -1.979318808957, ‖∇f‖ = 6.7766e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, time 2105.10 s: f = -1.979323359430, ‖∇f‖ = 4.4583e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time 2116.68 s: f = -1.979327209560, ‖∇f‖ = 6.9767e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time 2128.79 s: f = -1.979330802421, ‖∇f‖ = 4.5201e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 120 iterations and time 2140.39 s: f = -1.979336253182, ‖∇f‖ = 5.4141e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197

````

Finally, we compare the variationally optimized energy against the reference energy. Indeed,
we find that the additional AD-based optimization improves the SU-evolved PEPS and leads to
a more accurate energy estimate.

````julia
E_opt /= (Nr * Nc)
@show E_opt
@show (E_opt - E_ref) / abs(E_ref);
````

````
E_opt = -0.4948340632954646
(E_opt - E_ref) / abs(E_ref) = -0.0011817163287092872

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

