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
trscheme_peps = truncerror(; atol=1.0e-10) & truncrank(Dbond)
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
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.188e+00,  time = 28.698 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 1832   :  dt = 1e-02,  weight diff = 9.942e-09,  time = 104.269 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.400e-04,  time = 0.084 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 523    :  dt = 1e-02,  weight diff = 9.964e-09,  time = 19.444 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.524e-04,  time = 0.065 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 611    :  dt = 1e-02,  weight diff = 9.848e-09,  time = 22.684 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.661e-04,  time = 0.036 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 735    :  dt = 1e-02,  weight diff = 9.962e-09,  time = 27.379 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.823e-04,  time = 0.036 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 901    :  dt = 1e-02,  weight diff = 9.994e-09,  time = 33.440 sec

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
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 4.447e-04,  time = 0.036 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 3236   :  dt = 1e-03,  weight diff = 9.998e-10,  time = 118.922 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 4.436e-05,  time = 0.036 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 796    :  dt = 1e-04,  weight diff = 9.999e-10,  time = 29.313 sec

````

## Computing the simple update energy estimate

Finally, we measure the ground-state energy by converging a CTMRG environment and computing
the expectation value, where we first normalize tensors in the PEPS:

````julia
normalize!.(peps.A, Inf) ## normalize each PEPS tensor by largest element
χenv = 32
trscheme_env = truncerror(; atol=1.0e-10) & truncrank(χenv)
Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1 // 2 => χenv ÷ 4, -1 // 2 => χenv ÷ 4)
env₀ = CTMRGEnv(rand, Float64, peps, Espace)
env, = leading_boundary(env₀, peps; tol = 1.0e-10, alg = :sequential, trscheme = trscheme_env);
E = expectation_value(peps, H, env) / (Nr * Nc)
````

````
-0.4908482949689091
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.00688255949639037

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
[ Info: LBFGS: iter    1, time  652.92 s: f = -1.912844704754, ‖∇f‖ = 6.1629e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, time  664.36 s: f = -1.937956278406, ‖∇f‖ = 2.9827e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  674.98 s: f = -1.943097558049, ‖∇f‖ = 2.0511e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  686.77 s: f = -1.951025956511, ‖∇f‖ = 1.4073e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  697.58 s: f = -1.955168827774, ‖∇f‖ = 1.2051e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  708.82 s: f = -1.960117970278, ‖∇f‖ = 1.1611e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  720.26 s: f = -1.961216550036, ‖∇f‖ = 1.6042e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  731.38 s: f = -1.963394812594, ‖∇f‖ = 7.0500e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  742.56 s: f = -1.964678496250, ‖∇f‖ = 4.8767e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  755.02 s: f = -1.965924354920, ‖∇f‖ = 6.0609e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  767.32 s: f = -1.968120587109, ‖∇f‖ = 7.1520e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  780.79 s: f = -1.969562403653, ‖∇f‖ = 9.5065e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  793.16 s: f = -1.970582005143, ‖∇f‖ = 6.1728e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  805.31 s: f = -1.971121191726, ‖∇f‖ = 3.2137e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  818.38 s: f = -1.971554453753, ‖∇f‖ = 2.5404e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  831.96 s: f = -1.972190061011, ‖∇f‖ = 3.2651e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  844.26 s: f = -1.972856212899, ‖∇f‖ = 3.0425e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  857.72 s: f = -1.973765750218, ‖∇f‖ = 3.2201e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  870.23 s: f = -1.974161602591, ‖∇f‖ = 3.5568e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  883.37 s: f = -1.974478336877, ‖∇f‖ = 1.8682e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  895.17 s: f = -1.974710833057, ‖∇f‖ = 1.5936e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  908.28 s: f = -1.974975741077, ‖∇f‖ = 2.3555e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  920.27 s: f = -1.975198522454, ‖∇f‖ = 2.7962e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  932.03 s: f = -1.975394248410, ‖∇f‖ = 2.5314e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  944.70 s: f = -1.975512455006, ‖∇f‖ = 2.9518e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  956.37 s: f = -1.975616686676, ‖∇f‖ = 1.6802e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  967.76 s: f = -1.975710241181, ‖∇f‖ = 1.5024e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  978.98 s: f = -1.975795271168, ‖∇f‖ = 1.2833e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  989.49 s: f = -1.975865143335, ‖∇f‖ = 2.3418e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1002.34 s: f = -1.975957427509, ‖∇f‖ = 1.1381e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1013.66 s: f = -1.976015850153, ‖∇f‖ = 1.0083e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1024.96 s: f = -1.976080164930, ‖∇f‖ = 1.3119e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1037.67 s: f = -1.976152913537, ‖∇f‖ = 1.4611e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1048.69 s: f = -1.976198207069, ‖∇f‖ = 9.6971e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1061.48 s: f = -1.976250601632, ‖∇f‖ = 9.3909e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time 1073.15 s: f = -1.976291935078, ‖∇f‖ = 1.4028e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1084.60 s: f = -1.976340224013, ‖∇f‖ = 1.0472e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1097.26 s: f = -1.976377344626, ‖∇f‖ = 6.9713e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1109.15 s: f = -1.976411251633, ‖∇f‖ = 6.5882e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1122.25 s: f = -1.976451966987, ‖∇f‖ = 7.7535e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1133.98 s: f = -1.976491819091, ‖∇f‖ = 1.1597e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1145.62 s: f = -1.976529574561, ‖∇f‖ = 8.6643e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1158.08 s: f = -1.976564683972, ‖∇f‖ = 7.6509e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1168.84 s: f = -1.976599313000, ‖∇f‖ = 7.3612e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1181.74 s: f = -1.976636515361, ‖∇f‖ = 1.8259e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1193.53 s: f = -1.976680292934, ‖∇f‖ = 9.8005e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1205.13 s: f = -1.976709176062, ‖∇f‖ = 8.4490e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1216.87 s: f = -1.976735409874, ‖∇f‖ = 1.5557e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1228.56 s: f = -1.976776133103, ‖∇f‖ = 1.3263e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1241.43 s: f = -1.976811137668, ‖∇f‖ = 1.1320e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1265.73 s: f = -1.976843854259, ‖∇f‖ = 1.4911e-02, α = 5.24e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   52, time 1278.46 s: f = -1.976923448868, ‖∇f‖ = 9.7919e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1289.71 s: f = -1.976965559297, ‖∇f‖ = 7.7472e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1301.37 s: f = -1.977044614238, ‖∇f‖ = 1.3785e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1313.26 s: f = -1.977075753938, ‖∇f‖ = 2.0809e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1325.74 s: f = -1.977139445975, ‖∇f‖ = 9.3821e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1336.75 s: f = -1.977165276657, ‖∇f‖ = 7.5268e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1348.24 s: f = -1.977219450466, ‖∇f‖ = 1.1473e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1359.70 s: f = -1.977275314249, ‖∇f‖ = 1.9030e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1372.04 s: f = -1.977335522183, ‖∇f‖ = 1.1000e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1383.33 s: f = -1.977385273141, ‖∇f‖ = 9.0696e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1394.84 s: f = -1.977430128301, ‖∇f‖ = 1.1553e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1406.33 s: f = -1.977469892558, ‖∇f‖ = 8.2877e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1418.79 s: f = -1.977495288994, ‖∇f‖ = 7.9121e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1430.48 s: f = -1.977553209211, ‖∇f‖ = 8.8664e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1443.25 s: f = -1.977597458477, ‖∇f‖ = 1.9052e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1453.49 s: f = -1.977660134707, ‖∇f‖ = 1.1063e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1465.98 s: f = -1.977685405419, ‖∇f‖ = 2.0774e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1476.58 s: f = -1.977717638754, ‖∇f‖ = 8.7390e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1488.80 s: f = -1.977730883236, ‖∇f‖ = 7.0000e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1500.40 s: f = -1.977768669053, ‖∇f‖ = 8.1853e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1511.94 s: f = -1.977810913385, ‖∇f‖ = 1.0740e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1523.64 s: f = -1.977817034893, ‖∇f‖ = 1.7372e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1535.46 s: f = -1.977859790789, ‖∇f‖ = 6.3128e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1547.05 s: f = -1.977877604524, ‖∇f‖ = 5.9647e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1559.50 s: f = -1.977904920943, ‖∇f‖ = 8.3968e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1571.15 s: f = -1.977943608891, ‖∇f‖ = 1.1579e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1582.92 s: f = -1.977960278758, ‖∇f‖ = 1.3388e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1595.75 s: f = -1.977963135797, ‖∇f‖ = 1.0697e-02, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 1606.72 s: f = -1.977988611344, ‖∇f‖ = 1.6698e-02
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
E_opt = -0.4944971528359841
(E_opt - E_ref) / abs(E_ref) = -0.0005000563196440829

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

