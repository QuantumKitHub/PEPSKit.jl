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
Random.seed!(29385293);
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
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.188e+00,  time = 0.029 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1832   :  dt = 1e-02,  weight diff = 9.942e-09,  time = 52.896 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.400e-04,  time = 0.031 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 523    :  dt = 1e-02,  weight diff = 9.964e-09,  time = 15.817 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.524e-04,  time = 0.029 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 611    :  dt = 1e-02,  weight diff = 9.848e-09,  time = 18.467 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.661e-04,  time = 0.029 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 735    :  dt = 1e-02,  weight diff = 9.962e-09,  time = 22.205 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.823e-04,  time = 0.029 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 901    :  dt = 1e-02,  weight diff = 9.994e-09,  time = 27.331 sec

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
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 4.447e-04,  time = 0.030 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 3236   :  dt = 1e-03,  weight diff = 9.998e-10,  time = 97.398 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 4.436e-05,  time = 0.031 sec
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 796    :  dt = 1e-04,  weight diff = 9.999e-10,  time = 24.018 sec

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
-0.4908482949688868
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.0068825594964354074

````

## Variational PEPS optimization using AD

As a last step, we will use the SU-evolved PEPS as a starting point for a [`fixedpoint`](@ref)
PEPS optimization. Note that we could have also used a sublattice-rotated version of `H` to
fit the Hamiltonian onto a single-site unit cell which would require us to optimize fewer
parameters and hence lead to a faster optimization. But here we instead take advantage of
the already evolved `peps`, thus giving us a physical initial guess for the optimization.
In order to break some of the $C_{4v}$ symmetry of the PEPS, we will add a bit of noise to it
- this is conviently done using MPSKit's `randomize!` function. (Breaking some of the spatial
symmetry can be advantageous for obtaining lower energies.)

````julia
using MPSKit: randomize!

noise_peps = InfinitePEPS(randomize!.(deepcopy(peps.A)))
peps₀ = peps + 1e-1noise_peps
peps_opt, env_opt, E_opt, = fixedpoint(
    H, peps₀, env; optimizer_alg=(; tol=1e-4, maxiter=80)
);
````

````
┌ Warning: the provided real environment was converted to a complex environment since :fixed mode generally produces complex gauges; use :diffgauge mode instead by passing gradient_alg=(; iterscheme=:diffgauge) to the fixedpoint keyword arguments to work with purely real environments
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/optimization/peps_optimization.jl:219
[ Info: LBFGS: initializing with f = -1.906213849448, ‖∇f‖ = 7.1073e-01
[ Info: LBFGS: iter    1, time   11.81 s: f = -1.912844704482, ‖∇f‖ = 6.1629e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, time   16.75 s: f = -1.937956267904, ‖∇f‖ = 2.9827e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time   21.80 s: f = -1.943097546791, ‖∇f‖ = 2.0511e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time   26.69 s: f = -1.951025946434, ‖∇f‖ = 1.4073e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time   31.77 s: f = -1.955168800010, ‖∇f‖ = 1.2051e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time   36.87 s: f = -1.960117968919, ‖∇f‖ = 1.1611e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time   42.26 s: f = -1.961216587562, ‖∇f‖ = 1.6042e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time   47.58 s: f = -1.963394830132, ‖∇f‖ = 7.0500e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time   52.76 s: f = -1.964678528654, ‖∇f‖ = 4.8767e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time   58.23 s: f = -1.965924377851, ‖∇f‖ = 6.0609e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time   64.25 s: f = -1.968120624918, ‖∇f‖ = 7.1520e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time   70.40 s: f = -1.969562421458, ‖∇f‖ = 9.5064e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time   76.48 s: f = -1.970581981978, ‖∇f‖ = 6.1731e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time   82.38 s: f = -1.971121190048, ‖∇f‖ = 3.2138e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time   88.28 s: f = -1.971554447655, ‖∇f‖ = 2.5404e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time   94.48 s: f = -1.972190046057, ‖∇f‖ = 3.2651e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  100.54 s: f = -1.972856179552, ‖∇f‖ = 3.0426e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  106.47 s: f = -1.973765629052, ‖∇f‖ = 3.2210e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  112.59 s: f = -1.974162050742, ‖∇f‖ = 3.5505e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  118.60 s: f = -1.974478515407, ‖∇f‖ = 1.8647e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  124.39 s: f = -1.974711840147, ‖∇f‖ = 1.5939e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  130.52 s: f = -1.974976992882, ‖∇f‖ = 2.3561e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  136.50 s: f = -1.975198552395, ‖∇f‖ = 2.8074e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  141.91 s: f = -1.975397188154, ‖∇f‖ = 2.4018e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  147.64 s: f = -1.975511606909, ‖∇f‖ = 3.1378e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  153.26 s: f = -1.975619359852, ‖∇f‖ = 1.7031e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  158.40 s: f = -1.975706047066, ‖∇f‖ = 1.5318e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  163.89 s: f = -1.975792908561, ‖∇f‖ = 1.3145e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  169.21 s: f = -1.975874444762, ‖∇f‖ = 2.0056e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  174.74 s: f = -1.975957400077, ‖∇f‖ = 1.2628e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  179.85 s: f = -1.976016647203, ‖∇f‖ = 1.0550e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  185.40 s: f = -1.976101267437, ‖∇f‖ = 1.0041e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  190.83 s: f = -1.976173093824, ‖∇f‖ = 1.4127e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  196.38 s: f = -1.976232003564, ‖∇f‖ = 9.3789e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  201.98 s: f = -1.976279429987, ‖∇f‖ = 9.6660e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  207.59 s: f = -1.976319807091, ‖∇f‖ = 1.2942e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  212.95 s: f = -1.976360454958, ‖∇f‖ = 8.8852e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  218.74 s: f = -1.976400234179, ‖∇f‖ = 6.9115e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  224.19 s: f = -1.976428915464, ‖∇f‖ = 7.0766e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  229.67 s: f = -1.976468131930, ‖∇f‖ = 7.3518e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  235.18 s: f = -1.976524294633, ‖∇f‖ = 1.0581e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  240.74 s: f = -1.976526983977, ‖∇f‖ = 1.6104e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  246.33 s: f = -1.976576416539, ‖∇f‖ = 6.5034e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  251.74 s: f = -1.976601737265, ‖∇f‖ = 5.8912e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  257.37 s: f = -1.976637886122, ‖∇f‖ = 8.4756e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  262.96 s: f = -1.976694292720, ‖∇f‖ = 9.7710e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  274.56 s: f = -1.976719690626, ‖∇f‖ = 1.9617e-02, α = 2.83e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   48, time  279.89 s: f = -1.976757180903, ‖∇f‖ = 9.0902e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  285.45 s: f = -1.976778896839, ‖∇f‖ = 8.4485e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  290.77 s: f = -1.976801887998, ‖∇f‖ = 1.3765e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  296.25 s: f = -1.976830529576, ‖∇f‖ = 1.8017e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  301.90 s: f = -1.976861162991, ‖∇f‖ = 2.6927e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  307.18 s: f = -1.976929533535, ‖∇f‖ = 1.0242e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  312.67 s: f = -1.976973353625, ‖∇f‖ = 7.1802e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  317.85 s: f = -1.976993717783, ‖∇f‖ = 8.8958e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  323.48 s: f = -1.977060306915, ‖∇f‖ = 1.1556e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  334.45 s: f = -1.977091448236, ‖∇f‖ = 1.6192e-02, α = 3.03e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   58, time  339.98 s: f = -1.977138624322, ‖∇f‖ = 1.0354e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  345.41 s: f = -1.977208179209, ‖∇f‖ = 8.7382e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  351.07 s: f = -1.977244959697, ‖∇f‖ = 1.7905e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  356.49 s: f = -1.977290172079, ‖∇f‖ = 1.1535e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  361.80 s: f = -1.977346829138, ‖∇f‖ = 9.3061e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  367.29 s: f = -1.977390173782, ‖∇f‖ = 1.0691e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  372.77 s: f = -1.977406788841, ‖∇f‖ = 2.4206e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  378.41 s: f = -1.977475388304, ‖∇f‖ = 8.1223e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  383.66 s: f = -1.977496756947, ‖∇f‖ = 6.2713e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  389.41 s: f = -1.977538498943, ‖∇f‖ = 8.1518e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  394.85 s: f = -1.977582944207, ‖∇f‖ = 1.2523e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  406.53 s: f = -1.977612439512, ‖∇f‖ = 1.7582e-02, α = 3.89e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   70, time  412.07 s: f = -1.977652248007, ‖∇f‖ = 9.1652e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  417.85 s: f = -1.977681403489, ‖∇f‖ = 8.3683e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  423.98 s: f = -1.977721141239, ‖∇f‖ = 9.4541e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  429.73 s: f = -1.977775043848, ‖∇f‖ = 1.1036e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  435.11 s: f = -1.977818026910, ‖∇f‖ = 1.2406e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  440.76 s: f = -1.977848105077, ‖∇f‖ = 1.1495e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  446.06 s: f = -1.977873438780, ‖∇f‖ = 5.9210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time  451.54 s: f = -1.977888581900, ‖∇f‖ = 6.3137e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  456.92 s: f = -1.977922415140, ‖∇f‖ = 1.1600e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  462.53 s: f = -1.977946751759, ‖∇f‖ = 1.2523e-02, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 468.13 s: f = -1.977972943430, ‖∇f‖ = 7.7636e-03
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
E_opt = -0.49449323585738636
(E_opt - E_ref) / abs(E_ref) = -0.0004921312238469133

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

