```@meta
EditURL = "../../../../examples/j1j2_su/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/j1j2_su/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/j1j2_su/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/j1j2_su)


# Three-site simple update for the $J_1$-$J_2$ model

In this example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional $J_1$-$J_2$ model, which contains next-nearest-neighbour interactions:

```math
H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
+ J_2 \sum_{\langle \langle i,j \rangle \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
```

Here we will exploit the $U(1)$ spin rotation symmetry in the $J_1$-$J_2$ model. The goal
will be to calculate the energy at $J_1 = 1$ and $J_2 = 1/2$, first using the simple update
algorithm and then, to refine the energy estimate, using AD-based variational PEPS
optimization.

We first import all required modules and seed the RNG:

````julia
using Random
using TensorKit, PEPSKit
Random.seed!(2025);
````

## Simple updating a challenging phase

Let's start by initializing an `InfiniteWeightPEPS` for which we set the required parameters
as well as physical and virtual vector spaces. We use the minimal unit cell size
($2 \times 2$) required by the simple update algorithm for Hamiltonians with
next-nearest-neighbour interactions:

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

The value $J_2 / J_1 = 0.5$ corresponds to a [possible spin liquid phase](@cite liu_gapless_2022),
which is challenging for SU to produce a relatively good state from random initialization.
Therefore, we shall gradually increase $J_2 / J_1$ from 0.1 to 0.5, each time initializing
on the previously evolved PEPS:

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
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1/2=>1, -1/2=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.199e+00,  time = 13.324 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1578   :  dt = 1e-02,  weight diff = 9.954e-09,  time = 35.786 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.409e-04,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 992    :  dt = 1e-02,  weight diff = 9.982e-09,  time = 11.605 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.449e-04,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1199   :  dt = 1e-02,  weight diff = 9.946e-09,  time = 13.999 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.613e-04,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1527   :  dt = 1e-02,  weight diff = 9.989e-09,  time = 17.865 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.830e-04,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 2650   :  dt = 1e-02,  weight diff = 1.000e-08,  time = 30.891 sec

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
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 4.232e-04,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 4000   :  dt = 1e-03,  weight diff = 2.409e-08,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 8000   :  dt = 1e-03,  weight diff = 7.222e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 12000  :  dt = 1e-03,  weight diff = 4.550e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 16000  :  dt = 1e-03,  weight diff = 3.389e-09,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 20000  :  dt = 1e-03,  weight diff = 2.661e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 24000  :  dt = 1e-03,  weight diff = 2.133e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 28000  :  dt = 1e-03,  weight diff = 1.725e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
┌ Warning: SU cancel 30000  :  dt = 1e-03,  weight diff = 1.555e-09,  time = 350.580 sec
└ @ PEPSKit ~/Projects/PEPSKit.jl/src/algorithms/time_evolution/simpleupdate3site.jl:459
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 4.218e-05,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 4000   :  dt = 1e-04,  weight diff = 1.858e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 8000   :  dt = 1e-04,  weight diff = 1.407e-09,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 12000  :  dt = 1e-04,  weight diff = 1.095e-09,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 13532  :  dt = 1e-04,  weight diff = 1.000e-09,  time = 157.949 sec

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
-0.49123916142041596
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.0060917320780658835

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
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{Float64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{Float64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time  486.86 s: f = -1.967094178478, ‖∇f‖ = 1.6146e-01, α = 1.00e+00, m = 0, nfg = 1
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{Float64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{Float64}}}, InfinitePEPS{TensorMap{Float64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{Float64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time  490.66 s: f = -1.968859096094, ‖∇f‖ = 1.0162e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  492.85 s: f = -1.969798623080, ‖∇f‖ = 1.0699e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  495.07 s: f = -1.971722717370, ‖∇f‖ = 7.0920e-02, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  497.21 s: f = -1.973042342743, ‖∇f‖ = 6.0516e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  499.40 s: f = -1.974999288324, ‖∇f‖ = 5.3851e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  501.71 s: f = -1.975497427654, ‖∇f‖ = 7.4920e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  504.38 s: f = -1.975955307302, ‖∇f‖ = 2.9377e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  506.72 s: f = -1.976079539456, ‖∇f‖ = 2.4072e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  509.25 s: f = -1.976518150784, ‖∇f‖ = 3.8542e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  511.35 s: f = -1.976781345186, ‖∇f‖ = 3.5505e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  513.84 s: f = -1.977266474893, ‖∇f‖ = 2.3062e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  516.12 s: f = -1.977665959506, ‖∇f‖ = 2.2747e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  518.28 s: f = -1.977899382564, ‖∇f‖ = 2.0319e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  524.35 s: f = -1.977968122026, ‖∇f‖ = 2.6887e-02, α = 3.87e-01, m = 14, nfg = 2
[ Info: LBFGS: iter   16, time  526.61 s: f = -1.978071517732, ‖∇f‖ = 1.4041e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  528.95 s: f = -1.978158142242, ‖∇f‖ = 1.3544e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  531.26 s: f = -1.978302557989, ‖∇f‖ = 1.9631e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  533.69 s: f = -1.978422373719, ‖∇f‖ = 1.7767e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  538.91 s: f = -1.978458804585, ‖∇f‖ = 2.0974e-02, α = 2.75e-01, m = 19, nfg = 2
[ Info: LBFGS: iter   21, time  541.39 s: f = -1.978530076641, ‖∇f‖ = 9.3601e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  543.92 s: f = -1.978560656993, ‖∇f‖ = 8.7437e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  546.02 s: f = -1.978614702848, ‖∇f‖ = 1.3827e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  548.50 s: f = -1.978644893212, ‖∇f‖ = 2.9988e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  550.81 s: f = -1.978710378042, ‖∇f‖ = 1.2197e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  553.10 s: f = -1.978755635752, ‖∇f‖ = 9.9248e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  555.66 s: f = -1.978810591666, ‖∇f‖ = 1.0197e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  558.39 s: f = -1.978880651034, ‖∇f‖ = 1.3230e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  563.50 s: f = -1.978893545554, ‖∇f‖ = 1.2667e-02, α = 2.02e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   30, time  565.87 s: f = -1.978913378960, ‖∇f‖ = 8.4297e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  568.41 s: f = -1.978937870819, ‖∇f‖ = 7.3072e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  571.04 s: f = -1.978961658941, ‖∇f‖ = 8.2989e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  573.61 s: f = -1.978995792597, ‖∇f‖ = 9.0362e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  578.48 s: f = -1.979015006235, ‖∇f‖ = 9.8341e-03, α = 3.54e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   35, time  580.96 s: f = -1.979038166915, ‖∇f‖ = 5.2968e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  583.21 s: f = -1.979055327689, ‖∇f‖ = 5.1642e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  585.57 s: f = -1.979067394575, ‖∇f‖ = 5.6902e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  588.06 s: f = -1.979081709577, ‖∇f‖ = 1.2187e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  590.55 s: f = -1.979101892685, ‖∇f‖ = 6.0559e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  592.90 s: f = -1.979120149059, ‖∇f‖ = 5.6347e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  595.38 s: f = -1.979132668341, ‖∇f‖ = 5.1813e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  598.14 s: f = -1.979141653517, ‖∇f‖ = 8.3680e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  600.61 s: f = -1.979152302552, ‖∇f‖ = 3.6028e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  603.05 s: f = -1.979157388636, ‖∇f‖ = 3.5709e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  605.48 s: f = -1.979168453186, ‖∇f‖ = 6.1935e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  610.77 s: f = -1.979171015879, ‖∇f‖ = 5.1952e-03, α = 3.01e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   47, time  613.14 s: f = -1.979174715139, ‖∇f‖ = 3.0583e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  615.55 s: f = -1.979177809778, ‖∇f‖ = 2.5590e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  618.08 s: f = -1.979181346423, ‖∇f‖ = 2.8900e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  620.83 s: f = -1.979188173697, ‖∇f‖ = 5.6962e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  623.32 s: f = -1.979189848058, ‖∇f‖ = 9.5453e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  625.71 s: f = -1.979198642059, ‖∇f‖ = 2.6808e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  628.33 s: f = -1.979202022387, ‖∇f‖ = 2.7910e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  630.95 s: f = -1.979206038736, ‖∇f‖ = 2.8136e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  633.56 s: f = -1.979211332101, ‖∇f‖ = 5.3195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  636.38 s: f = -1.979214444089, ‖∇f‖ = 2.2712e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  638.84 s: f = -1.979217057711, ‖∇f‖ = 2.1421e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  641.23 s: f = -1.979219877997, ‖∇f‖ = 2.2151e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  643.81 s: f = -1.979221424825, ‖∇f‖ = 4.9270e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  646.41 s: f = -1.979224574143, ‖∇f‖ = 1.9785e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  649.05 s: f = -1.979226633793, ‖∇f‖ = 1.9307e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  651.59 s: f = -1.979228747165, ‖∇f‖ = 2.9825e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  654.16 s: f = -1.979233255704, ‖∇f‖ = 3.8639e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  659.40 s: f = -1.979234559253, ‖∇f‖ = 5.6800e-03, α = 2.41e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   65, time  661.96 s: f = -1.979238839779, ‖∇f‖ = 3.5885e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  664.62 s: f = -1.979241524115, ‖∇f‖ = 1.7775e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  667.34 s: f = -1.979243457078, ‖∇f‖ = 2.0844e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  670.04 s: f = -1.979244122276, ‖∇f‖ = 6.2078e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  672.58 s: f = -1.979245514751, ‖∇f‖ = 2.9255e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  675.13 s: f = -1.979247098476, ‖∇f‖ = 2.1148e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  677.86 s: f = -1.979248389860, ‖∇f‖ = 2.9019e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  680.26 s: f = -1.979250153582, ‖∇f‖ = 2.5402e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  685.49 s: f = -1.979251166283, ‖∇f‖ = 5.0665e-03, α = 3.60e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   74, time  688.13 s: f = -1.979253323083, ‖∇f‖ = 2.0341e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  690.78 s: f = -1.979254172551, ‖∇f‖ = 1.4352e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  693.26 s: f = -1.979254808251, ‖∇f‖ = 1.7431e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time  695.77 s: f = -1.979255242114, ‖∇f‖ = 2.1139e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  698.24 s: f = -1.979255971001, ‖∇f‖ = 1.7237e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  700.53 s: f = -1.979257820665, ‖∇f‖ = 2.1577e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time  703.17 s: f = -1.979259307993, ‖∇f‖ = 2.2533e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time  705.99 s: f = -1.979259331816, ‖∇f‖ = 4.4053e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time  708.52 s: f = -1.979262001229, ‖∇f‖ = 1.2275e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time  710.96 s: f = -1.979262597089, ‖∇f‖ = 1.0895e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time  713.56 s: f = -1.979263895094, ‖∇f‖ = 1.4424e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   85, time  716.03 s: f = -1.979264928007, ‖∇f‖ = 3.3773e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time  718.66 s: f = -1.979266263869, ‖∇f‖ = 2.4908e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time  721.05 s: f = -1.979267255939, ‖∇f‖ = 1.8046e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time  723.45 s: f = -1.979268326721, ‖∇f‖ = 1.7492e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time  725.84 s: f = -1.979269516239, ‖∇f‖ = 1.8060e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   90, time  728.65 s: f = -1.979271747736, ‖∇f‖ = 1.4482e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time  733.71 s: f = -1.979273222660, ‖∇f‖ = 3.1475e-03, α = 3.85e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   92, time  736.19 s: f = -1.979274573334, ‖∇f‖ = 2.5644e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time  738.78 s: f = -1.979275871427, ‖∇f‖ = 1.7748e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time  741.58 s: f = -1.979277527208, ‖∇f‖ = 1.9489e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time  744.12 s: f = -1.979278933406, ‖∇f‖ = 3.8069e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time  746.86 s: f = -1.979280476638, ‖∇f‖ = 2.0024e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time  749.45 s: f = -1.979281562154, ‖∇f‖ = 1.5589e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time  751.81 s: f = -1.979283136842, ‖∇f‖ = 1.9286e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time  754.70 s: f = -1.979285688209, ‖∇f‖ = 2.5845e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time  759.75 s: f = -1.979287349160, ‖∇f‖ = 2.9232e-03, α = 4.40e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  101, time  762.17 s: f = -1.979289180676, ‖∇f‖ = 1.4840e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time  764.70 s: f = -1.979290072410, ‖∇f‖ = 1.3705e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time  767.28 s: f = -1.979291102131, ‖∇f‖ = 1.8725e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time  769.73 s: f = -1.979292329342, ‖∇f‖ = 1.8836e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time  772.11 s: f = -1.979294093996, ‖∇f‖ = 1.7481e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time  774.76 s: f = -1.979294717176, ‖∇f‖ = 3.9784e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time  777.53 s: f = -1.979297633159, ‖∇f‖ = 1.9640e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, time  780.28 s: f = -1.979298791856, ‖∇f‖ = 1.2162e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time  782.84 s: f = -1.979300321605, ‖∇f‖ = 1.7806e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time  785.48 s: f = -1.979302352779, ‖∇f‖ = 1.8707e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time  788.33 s: f = -1.979306643035, ‖∇f‖ = 2.4972e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time  791.25 s: f = -1.979307583867, ‖∇f‖ = 7.9491e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time  793.99 s: f = -1.979310642528, ‖∇f‖ = 4.1222e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time  796.80 s: f = -1.979312275729, ‖∇f‖ = 3.1753e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time  799.30 s: f = -1.979313964313, ‖∇f‖ = 2.3284e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time  802.07 s: f = -1.979321516929, ‖∇f‖ = 3.4941e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, time  804.92 s: f = -1.979328050666, ‖∇f‖ = 5.3164e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time  807.48 s: f = -1.979335505121, ‖∇f‖ = 6.6154e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time  810.12 s: f = -1.979335751013, ‖∇f‖ = 8.0663e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 120 iterations and time 812.95 s: f = -1.979338538631, ‖∇f‖ = 9.3207e-03
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
E_opt = -0.494834634657737
(E_opt - E_ref) / abs(E_ref) = -0.0011828723474699126

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

