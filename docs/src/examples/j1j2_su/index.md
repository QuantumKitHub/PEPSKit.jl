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

Let's start by initializing an `InfinitePEPS` for which we set the required parameters
as well as physical and virtual vector spaces.
The `SUWeight` used by simple update will be initialized to identity matrices.
We use the minimal unit cell size ($2 \times 2$) required by the simple update algorithm
for Hamiltonians with next-nearest-neighbour interactions:

````julia
Dbond, symm = 4, U1Irrep
Nr, Nc, J1 = 2, 2, 1.0

# random initialization of 2x2 iPEPS (using real numbers) and SUWeight
Pspace = Vect[U1Irrep](1 // 2 => 1, -1 // 2 => 1)
Vspace = Vect[U1Irrep](0 => 2, 1 // 2 => 1, -1 // 2 => 1)
peps = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (Nr, Nc));
wts = SUWeight(peps);
````

The value $J_2 / J_1 = 0.5$ corresponds to a [possible spin liquid phase](@cite liu_gapless_2022),
which is challenging for SU to produce a relatively good state from random initialization.
Therefore, we shall gradually increase $J_2 / J_1$ from 0.1 to 0.5, each time initializing
on the previously evolved PEPS:

````julia
dt, tol, maxiter = 1.0e-2, 1.0e-8, 30000
check_interval = 4000
trscheme_peps = truncerr(1.0e-10) & truncdim(Dbond)
alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
for J2 in 0.1:0.1:0.5
    H = real( ## convert Hamiltonian `LocalOperator` to real floats
        j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice = false),
    )
    global peps, wts, = simpleupdate(peps, H, alg, wts; check_interval)
end
````

````
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1/2=>1, -1/2=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.188e+00,  time = 33.957 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1832   :  dt = 1e-02,  weight diff = 9.942e-09,  time = 114.130 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.400e-04,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 523    :  dt = 1e-02,  weight diff = 9.964e-09,  time = 18.644 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.524e-04,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 611    :  dt = 1e-02,  weight diff = 9.848e-09,  time = 21.764 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.661e-04,  time = 0.043 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 735    :  dt = 1e-02,  weight diff = 9.962e-09,  time = 26.104 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.823e-04,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 901    :  dt = 1e-02,  weight diff = 9.994e-09,  time = 32.042 sec

````

After we reach $J_2 / J_1 = 0.5$, we gradually decrease the evolution time step to obtain
a more accurately evolved PEPS:

````julia
dts = [1.0e-3, 1.0e-4]
tols = [1.0e-9, 1.0e-9]
J2 = 0.5
H = real(j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice = false))
for (dt, tol) in zip(dts, tols)
    alg′ = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    global peps, wts, = simpleupdate(peps, H, alg′, wts; check_interval)
end
````

````
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 4.447e-04,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 3236   :  dt = 1e-03,  weight diff = 9.998e-10,  time = 114.978 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 4.436e-05,  time = 0.035 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 796    :  dt = 1e-04,  weight diff = 9.999e-10,  time = 28.336 sec

````

## Computing the simple update energy estimate

Finally, we measure the ground-state energy by converging a CTMRG environment and computing
the expectation value, where we first normalize tensors in the PEPS:

````julia
normalize!.(peps.A, Inf) ## normalize each PEPS tensor by largest element
χenv = 32
trscheme_env = truncerr(1.0e-10) & truncdim(χenv)
Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1 // 2 => χenv ÷ 4, -1 // 2 => χenv ÷ 4)
env₀ = CTMRGEnv(rand, Float64, peps, Espace)
env, = leading_boundary(env₀, peps; tol = 1.0e-10, alg = :sequential, trscheme = trscheme_env);
E = expectation_value(peps, H, env) / (Nr * Nc)
````

````
-0.4908482949688707
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.006882559496467979

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
peps₀ = peps + 1.0e-1noise_peps
peps_opt, env_opt, E_opt, = fixedpoint(
    H, peps₀, env; optimizer_alg = (; tol = 1.0e-4, maxiter = 80)
);
````

````
┌ Warning: the provided real environment was converted to a complex environment since :fixed mode generally produces complex gauges; use :diffgauge mode instead by passing gradient_alg=(; iterscheme=:diffgauge) to the fixedpoint keyword arguments to work with purely real environments
└ @ PEPSKit ~/PEPSKit.jl/src/algorithms/optimization/peps_optimization.jl:204
[ Info: LBFGS: initializing with f = -1.906213849448, ‖∇f‖ = 7.1073e-01
[ Info: LBFGS: iter    1, time  898.29 s: f = -1.912844704482, ‖∇f‖ = 6.1629e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, time  913.28 s: f = -1.937956267904, ‖∇f‖ = 2.9827e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  925.98 s: f = -1.943097546791, ‖∇f‖ = 2.0511e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  938.28 s: f = -1.951025946434, ‖∇f‖ = 1.4073e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  951.38 s: f = -1.955168800011, ‖∇f‖ = 1.2051e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  964.82 s: f = -1.960117968919, ‖∇f‖ = 1.1611e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  978.89 s: f = -1.961216587563, ‖∇f‖ = 1.6042e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  991.83 s: f = -1.963394830132, ‖∇f‖ = 7.0500e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time 1005.38 s: f = -1.964678528654, ‖∇f‖ = 4.8767e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time 1019.28 s: f = -1.965924377851, ‖∇f‖ = 6.0609e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time 1034.73 s: f = -1.968120624918, ‖∇f‖ = 7.1520e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time 1050.39 s: f = -1.969562421459, ‖∇f‖ = 9.5064e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time 1065.73 s: f = -1.970581981978, ‖∇f‖ = 6.1731e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time 1080.88 s: f = -1.971121190048, ‖∇f‖ = 3.2138e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time 1096.27 s: f = -1.971554447655, ‖∇f‖ = 2.5404e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time 1112.31 s: f = -1.972190046057, ‖∇f‖ = 3.2651e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time 1127.59 s: f = -1.972856179552, ‖∇f‖ = 3.0426e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time 1142.77 s: f = -1.973765629051, ‖∇f‖ = 3.2210e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time 1158.83 s: f = -1.974162050739, ‖∇f‖ = 3.5505e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time 1174.20 s: f = -1.974478515405, ‖∇f‖ = 1.8647e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time 1188.75 s: f = -1.974711840140, ‖∇f‖ = 1.5939e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time 1203.68 s: f = -1.974976992873, ‖∇f‖ = 2.3561e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time 1218.85 s: f = -1.975198552393, ‖∇f‖ = 2.8074e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time 1233.56 s: f = -1.975397188135, ‖∇f‖ = 2.4018e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time 1248.19 s: f = -1.975511606912, ‖∇f‖ = 3.1378e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time 1262.15 s: f = -1.975619359834, ‖∇f‖ = 1.7031e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time 1276.32 s: f = -1.975706047087, ‖∇f‖ = 1.5318e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time 1290.34 s: f = -1.975792908567, ‖∇f‖ = 1.3145e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time 1304.38 s: f = -1.975874444720, ‖∇f‖ = 2.0056e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1317.96 s: f = -1.975957400085, ‖∇f‖ = 1.2628e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1331.94 s: f = -1.976016647195, ‖∇f‖ = 1.0550e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1345.46 s: f = -1.976101267430, ‖∇f‖ = 1.0041e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1360.19 s: f = -1.976173093751, ‖∇f‖ = 1.4127e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1374.60 s: f = -1.976232003408, ‖∇f‖ = 9.3789e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1388.17 s: f = -1.976279429906, ‖∇f‖ = 9.6660e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time 1402.25 s: f = -1.976319806974, ‖∇f‖ = 1.2942e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1415.86 s: f = -1.976360454784, ‖∇f‖ = 8.8852e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1430.34 s: f = -1.976400233969, ‖∇f‖ = 6.9115e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1444.04 s: f = -1.976428915127, ‖∇f‖ = 7.0765e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1458.00 s: f = -1.976468131639, ‖∇f‖ = 7.3518e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1473.47 s: f = -1.976524295199, ‖∇f‖ = 1.0581e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1488.52 s: f = -1.976526981429, ‖∇f‖ = 1.6105e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1502.59 s: f = -1.976576416756, ‖∇f‖ = 6.5034e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1517.27 s: f = -1.976601736365, ‖∇f‖ = 5.8911e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1531.80 s: f = -1.976637885480, ‖∇f‖ = 8.4757e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1546.97 s: f = -1.976694290296, ‖∇f‖ = 9.7709e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1577.50 s: f = -1.976719688110, ‖∇f‖ = 1.9617e-02, α = 2.83e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   48, time 1591.70 s: f = -1.976757178552, ‖∇f‖ = 9.0902e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1605.99 s: f = -1.976778894853, ‖∇f‖ = 8.4484e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1619.60 s: f = -1.976801880463, ‖∇f‖ = 1.3767e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1633.80 s: f = -1.976830518191, ‖∇f‖ = 1.8018e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1647.61 s: f = -1.976861150678, ‖∇f‖ = 2.6927e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1661.84 s: f = -1.976929519930, ‖∇f‖ = 1.0241e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1675.48 s: f = -1.976973340136, ‖∇f‖ = 7.1800e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1689.70 s: f = -1.976993707664, ‖∇f‖ = 8.8955e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1705.23 s: f = -1.977060299277, ‖∇f‖ = 1.1556e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1736.63 s: f = -1.977091456644, ‖∇f‖ = 1.6191e-02, α = 3.03e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   58, time 1752.52 s: f = -1.977138636484, ‖∇f‖ = 1.0351e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1768.37 s: f = -1.977208174771, ‖∇f‖ = 8.7379e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1784.51 s: f = -1.977244969855, ‖∇f‖ = 1.7908e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1799.80 s: f = -1.977290195277, ‖∇f‖ = 1.1535e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1815.58 s: f = -1.977346837916, ‖∇f‖ = 9.3049e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1831.45 s: f = -1.977390190411, ‖∇f‖ = 1.0692e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1848.06 s: f = -1.977406759237, ‖∇f‖ = 2.4211e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1864.52 s: f = -1.977475372347, ‖∇f‖ = 8.1217e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1882.38 s: f = -1.977496738214, ‖∇f‖ = 6.2709e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1903.00 s: f = -1.977538491432, ‖∇f‖ = 8.1507e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1918.81 s: f = -1.977582956803, ‖∇f‖ = 1.2520e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1955.10 s: f = -1.977612410222, ‖∇f‖ = 1.7568e-02, α = 3.88e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   70, time 1972.82 s: f = -1.977652223436, ‖∇f‖ = 9.1592e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1991.68 s: f = -1.977681395576, ‖∇f‖ = 8.3688e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 2010.03 s: f = -1.977721168632, ‖∇f‖ = 9.4687e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 2028.62 s: f = -1.977774963512, ‖∇f‖ = 1.1042e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 2047.13 s: f = -1.977818449696, ‖∇f‖ = 1.2241e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 2066.07 s: f = -1.977847582882, ‖∇f‖ = 1.1741e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 2084.23 s: f = -1.977873468930, ‖∇f‖ = 5.8834e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 2103.77 s: f = -1.977888447802, ‖∇f‖ = 6.2863e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 2123.02 s: f = -1.977922062390, ‖∇f‖ = 1.1548e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 2144.44 s: f = -1.977946486823, ‖∇f‖ = 1.2495e-02, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 2165.78 s: f = -1.977972769814, ‖∇f‖ = 7.8045e-03
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
E_opt = -0.494493192453459
(E_opt - E_ref) / abs(E_ref) = -0.0004920434060880036

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

