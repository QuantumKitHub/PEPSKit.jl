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
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1/2=>1, -1/2=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.204e+00,  time = 21.105 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 833    :  dt = 1e-02,  weight diff = 9.845e-09,  time = 55.476 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.400e-04,  time = 0.030 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 523    :  dt = 1e-02,  weight diff = 9.964e-09,  time = 16.706 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.524e-04,  time = 0.030 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 611    :  dt = 1e-02,  weight diff = 9.848e-09,  time = 19.395 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.661e-04,  time = 0.029 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 735    :  dt = 1e-02,  weight diff = 9.962e-09,  time = 23.365 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 3.823e-04,  time = 0.030 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 901    :  dt = 1e-02,  weight diff = 9.994e-09,  time = 28.680 sec

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
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 4.447e-04,  time = 0.031 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 3236   :  dt = 1e-03,  weight diff = 9.998e-10,  time = 102.956 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 4.436e-05,  time = 0.030 sec
[ Info: Space of x-weight at [1, 1] = Rep[TensorKitSectors.U₁](0=>2, 1=>1, -1=>1)
[ Info: SU conv 796    :  dt = 1e-04,  weight diff = 9.999e-10,  time = 25.509 sec

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
-0.4908482949688836
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.006882559496441921

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
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/optimization/peps_optimization.jl:219
[ Info: LBFGS: initializing with f = -1.963393174144, ‖∇f‖ = 2.5247e-01
[ Info: LBFGS: iter    1, time  789.16 s: f = -1.965293614670, ‖∇f‖ = 1.2937e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, time  796.64 s: f = -1.966093576333, ‖∇f‖ = 4.6405e-02, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  802.31 s: f = -1.966265429828, ‖∇f‖ = 4.7906e-02, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  808.24 s: f = -1.967674858673, ‖∇f‖ = 6.1278e-02, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  813.85 s: f = -1.968844367812, ‖∇f‖ = 6.2868e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  819.85 s: f = -1.971599928634, ‖∇f‖ = 7.5763e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  825.95 s: f = -1.973089568917, ‖∇f‖ = 6.8156e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  832.71 s: f = -1.973501669947, ‖∇f‖ = 3.4191e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  837.68 s: f = -1.973599617444, ‖∇f‖ = 1.9694e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  843.83 s: f = -1.973779575739, ‖∇f‖ = 2.1857e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  861.56 s: f = -1.974146329371, ‖∇f‖ = 7.0759e-02, α = 4.26e+00, m = 10, nfg = 3
[ Info: LBFGS: iter   12, time  867.22 s: f = -1.974498372189, ‖∇f‖ = 5.5973e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  904.05 s: f = -1.974544896993, ‖∇f‖ = 4.0486e-02, α = 1.58e+00, m = 12, nfg = 3
[ Info: LBFGS: iter   14, time  910.18 s: f = -1.975091371167, ‖∇f‖ = 1.3931e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  916.45 s: f = -1.975194168295, ‖∇f‖ = 1.8709e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  922.47 s: f = -1.975342405312, ‖∇f‖ = 2.8151e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  928.16 s: f = -1.975550371588, ‖∇f‖ = 1.7411e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  933.95 s: f = -1.975757801941, ‖∇f‖ = 1.1961e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  939.35 s: f = -1.975844564440, ‖∇f‖ = 2.3589e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  945.42 s: f = -1.975912685001, ‖∇f‖ = 1.2084e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time  950.88 s: f = -1.975949932830, ‖∇f‖ = 1.5151e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time  956.49 s: f = -1.976007337057, ‖∇f‖ = 1.8263e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  962.53 s: f = -1.976064604180, ‖∇f‖ = 1.1529e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  968.79 s: f = -1.976100678630, ‖∇f‖ = 1.2354e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  974.71 s: f = -1.976130373006, ‖∇f‖ = 7.5026e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  981.06 s: f = -1.976155537470, ‖∇f‖ = 1.1048e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  986.78 s: f = -1.976178270604, ‖∇f‖ = 1.0613e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  992.67 s: f = -1.976221667635, ‖∇f‖ = 7.1465e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  998.81 s: f = -1.976236017801, ‖∇f‖ = 1.2458e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1005.26 s: f = -1.976255313459, ‖∇f‖ = 5.8899e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1012.17 s: f = -1.976268990297, ‖∇f‖ = 4.2246e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1017.77 s: f = -1.976282398590, ‖∇f‖ = 4.9870e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1023.97 s: f = -1.976300488749, ‖∇f‖ = 3.7093e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1030.39 s: f = -1.976311130394, ‖∇f‖ = 8.9712e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1036.49 s: f = -1.976325840732, ‖∇f‖ = 4.5886e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time 1043.53 s: f = -1.976340082532, ‖∇f‖ = 4.5558e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1049.57 s: f = -1.976353232643, ‖∇f‖ = 3.8909e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1056.98 s: f = -1.976361179112, ‖∇f‖ = 5.3873e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1063.35 s: f = -1.976371229094, ‖∇f‖ = 4.5733e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1069.30 s: f = -1.976379207936, ‖∇f‖ = 4.1988e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1075.33 s: f = -1.976398109552, ‖∇f‖ = 3.7531e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1081.36 s: f = -1.976412078755, ‖∇f‖ = 8.0933e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1087.34 s: f = -1.976420497525, ‖∇f‖ = 7.0456e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1092.91 s: f = -1.976428316685, ‖∇f‖ = 3.3541e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1098.49 s: f = -1.976434615963, ‖∇f‖ = 3.4019e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1104.26 s: f = -1.976440959212, ‖∇f‖ = 3.5545e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1110.29 s: f = -1.976444046272, ‖∇f‖ = 8.1897e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1115.86 s: f = -1.976450562733, ‖∇f‖ = 2.4357e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1121.19 s: f = -1.976452743652, ‖∇f‖ = 2.5128e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1126.91 s: f = -1.976457589918, ‖∇f‖ = 3.5120e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1133.25 s: f = -1.976466448321, ‖∇f‖ = 4.7502e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1139.01 s: f = -1.976473100689, ‖∇f‖ = 8.6561e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1144.76 s: f = -1.976485054654, ‖∇f‖ = 3.9594e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1150.05 s: f = -1.976490679086, ‖∇f‖ = 3.2039e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1155.54 s: f = -1.976501510670, ‖∇f‖ = 3.7013e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1161.07 s: f = -1.976512898006, ‖∇f‖ = 4.9212e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1166.89 s: f = -1.976515485635, ‖∇f‖ = 8.6330e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning:   Linesearch bisection failure: [a, b] = [1.92e-03, 1.92e-03], b-a = 1.16e-16, dϕᵃ = -6.98e-04, dϕᵇ = -6.98e-04, (ϕᵇ - ϕᵃ)/(b-a) = 7.67e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [1.92e-03, 1.92e-03], dϕᵃ = -6.98e-04, dϕᵇ = -6.98e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   58, time 1437.30 s: f = -1.976514485635, ‖∇f‖ = 8.9523e-03, α = 1.92e-03, m = 20, nfg = 46
┌ Warning:   Linesearch bisection failure: [a, b] = [2.25e-02, 2.25e-02], b-a = 1.73e-16, dϕᵃ = -7.73e-05, dϕᵇ = -7.73e-05, (ϕᵇ - ϕᵃ)/(b-a) = 1.02e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.25e-02, 2.25e-02], dϕᵃ = -7.73e-05, dϕᵇ = -7.73e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   59, time 1725.60 s: f = -1.976513485635, ‖∇f‖ = 9.2239e-03, α = 2.25e-02, m = 20, nfg = 52
┌ Warning:   Linesearch bisection failure: [a, b] = [8.15e-03, 8.15e-03], b-a = 2.20e-16, dϕᵃ = -2.04e-04, dϕᵇ = -2.04e-04, (ϕᵇ - ϕᵃ)/(b-a) = 4.03e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [8.15e-03, 8.15e-03], dϕᵃ = -2.04e-04, dϕᵇ = -2.04e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   60, time 2005.64 s: f = -1.976512485635, ‖∇f‖ = 9.4391e-03, α = 8.15e-03, m = 20, nfg = 50
┌ Warning:   Linesearch bisection failure: [a, b] = [6.82e-05, 6.82e-05], b-a = 1.96e-16, dϕᵃ = -2.20e-02, dϕᵇ = -2.20e-02, (ϕᵇ - ϕᵃ)/(b-a) = 4.54e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [6.82e-05, 6.82e-05], dϕᵃ = -2.20e-02, dϕᵇ = -2.20e-02, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   61, time 2272.08 s: f = -1.976511485635, ‖∇f‖ = 9.6386e-03, α = 6.82e-05, m = 20, nfg = 46
┌ Warning:   Linesearch bisection failure: [a, b] = [3.65e-03, 3.65e-03], b-a = 1.54e-16, dϕᵃ = -3.74e-04, dϕᵇ = -3.74e-04, (ϕᵇ - ϕᵃ)/(b-a) = 8.63e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [3.65e-03, 3.65e-03], dϕᵃ = -3.74e-04, dϕᵇ = -3.74e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   62, time 2560.21 s: f = -1.976510485635, ‖∇f‖ = 9.7947e-03, α = 3.65e-03, m = 20, nfg = 50
┌ Warning:   Linesearch bisection failure: [a, b] = [2.98e-02, 2.98e-02], b-a = 1.46e-16, dϕᵃ = -5.53e-05, dϕᵇ = -5.53e-05, (ϕᵇ - ϕᵃ)/(b-a) = 4.57e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.98e-02, 2.98e-02], dϕᵃ = -5.53e-05, dϕᵇ = -5.53e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   63, time 2864.18 s: f = -1.976509485635, ‖∇f‖ = 9.7583e-03, α = 2.98e-02, m = 20, nfg = 52
[ Info: LBFGS: iter   64, time 2897.72 s: f = -1.976509113260, ‖∇f‖ = 8.5834e-03, α = 1.32e-01, m = 20, nfg = 5
[ Info: LBFGS: iter   65, time 2904.14 s: f = -1.976511728753, ‖∇f‖ = 5.1482e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 2910.53 s: f = -1.976518267054, ‖∇f‖ = 5.8730e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 2918.06 s: f = -1.976528583824, ‖∇f‖ = 1.4587e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 2924.19 s: f = -1.976539290734, ‖∇f‖ = 5.1837e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 2930.37 s: f = -1.976540610481, ‖∇f‖ = 4.2927e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 2937.07 s: f = -1.976542660531, ‖∇f‖ = 8.7561e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 2944.20 s: f = -1.976552824972, ‖∇f‖ = 5.8789e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 2951.15 s: f = -1.976558830598, ‖∇f‖ = 4.8178e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 2979.77 s: f = -1.976565408784, ‖∇f‖ = 4.5885e-03, α = 4.69e-01, m = 20, nfg = 3
┌ Warning:   Linesearch bisection failure: [a, b] = [1.14e-01, 1.14e-01], b-a = 1.25e-16, dϕᵃ = -2.08e-05, dϕᵇ = -2.08e-05, (ϕᵇ - ϕᵃ)/(b-a) = 8.89e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [1.14e-01, 1.14e-01], dϕᵃ = -2.08e-05, dϕᵇ = -2.08e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   74, time 3413.66 s: f = -1.976564408784, ‖∇f‖ = 4.6237e-03, α = 1.14e-01, m = 20, nfg = 53
┌ Warning:   Linesearch bisection failure: [a, b] = [1.13e-02, 1.13e-02], b-a = 1.42e-16, dϕᵃ = -5.62e-05, dϕᵇ = -5.62e-05, (ϕᵇ - ϕᵃ)/(b-a) = 1.56e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [1.13e-02, 1.13e-02], dϕᵃ = -5.62e-05, dϕᵇ = -5.62e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   75, time 3908.05 s: f = -1.976563408784, ‖∇f‖ = 4.6321e-03, α = 1.13e-02, m = 20, nfg = 54
┌ Warning:   Linesearch bisection failure: [a, b] = [4.72e-03, 4.72e-03], b-a = 2.18e-16, dϕᵃ = -1.03e-04, dϕᵇ = -1.03e-04, (ϕᵇ - ϕᵃ)/(b-a) = 1.22e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [4.72e-03, 4.72e-03], dϕᵃ = -1.03e-04, dϕᵇ = -1.03e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   76, time 4408.36 s: f = -1.976562408784, ‖∇f‖ = 4.6299e-03, α = 4.72e-03, m = 20, nfg = 52
┌ Warning:   Linesearch bisection failure: [a, b] = [1.88e-03, 1.88e-03], b-a = 1.16e-16, dϕᵃ = -2.10e-04, dϕᵇ = -2.10e-04, (ϕᵇ - ϕᵃ)/(b-a) = 1.15e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [1.88e-03, 1.88e-03], dϕᵃ = -2.10e-04, dϕᵇ = -2.10e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   77, time 4929.42 s: f = -1.976561408784, ‖∇f‖ = 4.6332e-03, α = 1.88e-03, m = 20, nfg = 52
┌ Warning:   Linesearch bisection failure: [a, b] = [6.08e-04, 6.08e-04], b-a = 2.10e-16, dϕᵃ = -5.97e-04, dϕᵇ = -5.97e-04, (ϕᵇ - ϕᵃ)/(b-a) = 5.28e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [6.08e-04, 6.08e-04], dϕᵃ = -5.97e-04, dϕᵇ = -5.97e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   78, time 5449.91 s: f = -1.976560408784, ‖∇f‖ = 4.6344e-03, α = 6.08e-04, m = 20, nfg = 50
┌ Warning:   Linesearch bisection failure: [a, b] = [2.07e-04, 2.07e-04], b-a = 1.80e-16, dϕᵃ = -1.64e-03, dϕᵇ = -1.64e-03, (ϕᵇ - ϕᵃ)/(b-a) = 1.11e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.07e-04, 2.07e-04], dϕᵃ = -1.64e-03, dϕᵇ = -1.64e-03, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   79, time 5985.60 s: f = -1.976559408784, ‖∇f‖ = 4.6361e-03, α = 2.07e-04, m = 20, nfg = 49
┌ Warning:   Linesearch bisection failure: [a, b] = [2.74e-05, 2.74e-05], b-a = 2.06e-16, dϕᵃ = -1.19e-02, dϕᵇ = -1.19e-02, (ϕᵇ - ϕᵃ)/(b-a) = 8.62e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.74e-05, 2.74e-05], dϕᵃ = -1.19e-02, dϕᵇ = -1.19e-02, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   80, time 6500.38 s: f = -1.976558408784, ‖∇f‖ = 4.6378e-03, α = 2.74e-05, m = 20, nfg = 46
┌ Warning:   Linesearch bisection failure: [a, b] = [2.04e-05, 2.04e-05], b-a = 1.96e-16, dϕᵃ = -1.59e-02, dϕᵇ = -1.59e-02, (ϕᵇ - ϕᵃ)/(b-a) = 7.92e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.04e-05, 2.04e-05], dϕᵃ = -1.59e-02, dϕᵇ = -1.59e-02, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   81, time 7034.06 s: f = -1.976557408784, ‖∇f‖ = 4.6395e-03, α = 2.04e-05, m = 20, nfg = 46
┌ Warning:   Linesearch bisection failure: [a, b] = [9.53e-05, 9.53e-05], b-a = 1.71e-16, dϕᵃ = -3.46e-03, dϕᵇ = -3.46e-03, (ϕᵇ - ϕᵃ)/(b-a) = 5.18e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [9.53e-05, 9.53e-05], dϕᵃ = -3.46e-03, dϕᵇ = -3.46e-03, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   82, time 7582.07 s: f = -1.976556408784, ‖∇f‖ = 4.6410e-03, α = 9.53e-05, m = 20, nfg = 46
┌ Warning:   Linesearch bisection failure: [a, b] = [2.51e-04, 2.51e-04], b-a = 1.50e-16, dϕᵃ = -1.38e-03, dϕᵇ = -1.38e-03, (ϕᵇ - ϕᵃ)/(b-a) = 5.90e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.51e-04, 2.51e-04], dϕᵃ = -1.38e-03, dϕᵇ = -1.38e-03, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   83, time 8140.41 s: f = -1.976555408784, ‖∇f‖ = 4.6422e-03, α = 2.51e-04, m = 20, nfg = 49
[ Info: LBFGS: iter   84, time 8173.63 s: f = -1.976554982357, ‖∇f‖ = 5.3572e-03, α = 1.28e-02, m = 20, nfg = 4
[ Info: LBFGS: iter   85, time 8207.13 s: f = -1.976561323146, ‖∇f‖ = 8.6468e-03, α = 1.49e-01, m = 20, nfg = 3
[ Info: LBFGS: iter   86, time 8257.89 s: f = -1.976561742907, ‖∇f‖ = 7.8022e-03, α = 1.81e-01, m = 20, nfg = 4
┌ Warning:   Linesearch bisection failure: [a, b] = [1.34e-02, 1.34e-02], b-a = 1.27e-16, dϕᵃ = -1.97e-05, dϕᵇ = -1.97e-05, (ϕᵇ - ϕᵃ)/(b-a) = 5.26e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [1.34e-02, 1.34e-02], dϕᵃ = -1.97e-05, dϕᵇ = -1.97e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   87, time 9177.82 s: f = -1.976560742907, ‖∇f‖ = 7.7459e-03, α = 1.34e-02, m = 20, nfg = 54
┌ Warning:   Linesearch bisection failure: [a, b] = [4.08e-03, 4.08e-03], b-a = 2.10e-16, dϕᵃ = -5.38e-05, dϕᵇ = -5.38e-05, (ϕᵇ - ϕᵃ)/(b-a) = 1.38e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [4.08e-03, 4.08e-03], dϕᵃ = -5.38e-05, dϕᵇ = -5.38e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   88, time 10058.91 s: f = -1.976559742907, ‖∇f‖ = 7.6445e-03, α = 4.08e-03, m = 20, nfg = 53
┌ Warning:   Linesearch bisection failure: [a, b] = [2.79e-03, 2.79e-03], b-a = 2.13e-16, dϕᵃ = -7.48e-05, dϕᵇ = -7.48e-05, (ϕᵇ - ϕᵃ)/(b-a) = 8.33e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.79e-03, 2.79e-03], dϕᵃ = -7.48e-05, dϕᵇ = -7.48e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   89, time 10944.23 s: f = -1.976558742907, ‖∇f‖ = 7.5966e-03, α = 2.79e-03, m = 20, nfg = 52
┌ Warning:   Linesearch bisection failure: [a, b] = [1.44e-03, 1.44e-03], b-a = 2.09e-16, dϕᵃ = -1.63e-04, dϕᵇ = -1.63e-04, (ϕᵇ - ϕᵃ)/(b-a) = 6.37e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [1.44e-03, 1.44e-03], dϕᵃ = -1.63e-04, dϕᵇ = -1.63e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   90, time 11806.09 s: f = -1.976557742907, ‖∇f‖ = 7.5378e-03, α = 1.44e-03, m = 20, nfg = 50
┌ Warning:   Linesearch bisection failure: [a, b] = [2.35e-03, 2.35e-03], b-a = 1.37e-16, dϕᵃ = -1.19e-04, dϕᵇ = -1.19e-04, (ϕᵇ - ϕᵃ)/(b-a) = 2.10e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.35e-03, 2.35e-03], dϕᵃ = -1.19e-04, dϕᵇ = -1.19e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   91, time 12635.49 s: f = -1.976556742907, ‖∇f‖ = 7.4892e-03, α = 2.35e-03, m = 20, nfg = 51
┌ Warning:   Linesearch bisection failure: [a, b] = [3.34e-03, 3.34e-03], b-a = 2.18e-16, dϕᵃ = -1.09e-04, dϕᵇ = -1.09e-04, (ϕᵇ - ϕᵃ)/(b-a) = 4.07e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [3.34e-03, 3.34e-03], dϕᵃ = -1.09e-04, dϕᵇ = -1.09e-04, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   92, time 13380.37 s: f = -1.976555742907, ‖∇f‖ = 7.4272e-03, α = 3.34e-03, m = 20, nfg = 51
┌ Warning:   Linesearch bisection failure: [a, b] = [6.06e-03, 6.06e-03], b-a = 1.47e-16, dϕᵃ = -8.27e-05, dϕᵇ = -8.27e-05, (ϕᵇ - ϕᵃ)/(b-a) = 1.36e+01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [6.06e-03, 6.06e-03], dϕᵃ = -8.27e-05, dϕᵇ = -8.27e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   93, time 14069.53 s: f = -1.976554742907, ‖∇f‖ = 7.3582e-03, α = 6.06e-03, m = 20, nfg = 52
[ Info: LBFGS: iter   94, time 14145.98 s: f = -1.976554076781, ‖∇f‖ = 6.5395e-03, α = 6.48e-03, m = 20, nfg = 7
┌ Warning:   Linesearch bisection failure: [a, b] = [2.72e-02, 2.72e-02], b-a = 1.42e-16, dϕᵃ = -1.95e-05, dϕᵇ = -1.95e-05, (ϕᵇ - ϕᵃ)/(b-a) = 6.24e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning:   Linesearch bracket converged to a point without satisfying Wolfe conditions: [a,b] = [2.72e-02, 2.72e-02], dϕᵃ = -1.95e-05, dϕᵇ = -1.95e-05, ϕᵃ - ϕ₀ = 1.00e-06, ϕᵇ - ϕ₀ = 1.00e-06
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:267
[ Info: LBFGS: iter   95, time 14761.44 s: f = -1.976553076781, ‖∇f‖ = 6.6084e-03, α = 2.72e-02, m = 20, nfg = 53
[ Info: LBFGS: iter   96, time 14802.45 s: f = -1.976552359292, ‖∇f‖ = 8.1164e-03, α = 2.27e-01, m = 20, nfg = 3
[ Info: LBFGS: iter   97, time 14847.61 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 1.77e-01, m = 20, nfg = 3
┌ Warning: Expectation value is not real: -1.9765438384889413 + 3.192090457085267e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976538399335986 + 3.668484129177368e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909378 + 3.862865362266322e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134198 + 3.946180327146833e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533405175182 + 3.983552094655078e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432229 + 4.0010507841474475e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545738 + 4.0094886859923163e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479206 + 4.013627948010745e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297691 + 4.0156774296030055e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670852 + 4.0166971047407855e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234855 + 4.0172056724832366e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185133 + 4.017459638380356e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602925 + 4.017586541789798e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311668 + 4.0176499738954456e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166022 + 4.0176816843696233e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675159316 + 4.0176975389001593e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674980675 + 4.017705465790848e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913532 + 4.017709428230637e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466934 + 4.0177114107712864e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674824364 + 4.017712401214287e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131986 + 4.017712896810848e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076164 + 4.0177131445913085e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048242 + 4.01771326820743e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034262 + 4.017713330269802e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027279 + 4.0177133611658583e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023817 + 4.017713376605086e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802207 + 4.0177133842240857e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021186 + 4.017713388170648e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020744 + 4.0177133899561964e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020562 + 4.017713391681825e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020426 + 4.0177133912162553e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020362 + 4.0177133920279093e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713391951635e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133915055483e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.017713391708297e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133921363967e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133924805065e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133920295536e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133921444954e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133921362284e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713392119938e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391829409e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133916739046e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392468239e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392497501e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.017713392107617e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.0177133924226293e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.017713391280491e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133923469675e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391920031e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.017713392292279e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713391552977e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.017713391667047e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   98, time 15788.99 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889416 + 3.1920904570867037e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976538399335989 + 3.6684841294078164e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290937 + 3.862865362182008e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134225 + 3.9461803266548804e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533405175182 + 3.983552094720104e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432226 + 4.001050783641977e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545749 + 4.0094886855197344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479175 + 4.0136279477571506e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297683 + 4.0156774297161215e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670863 + 4.016697105278966e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348527 + 4.017205672586573e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532680518516 + 4.0174596387595253e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602923 + 4.0175865413703453e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311684 + 4.0176499732415267e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166017 + 4.01768168466149e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593177 + 4.0176975383180656e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806746 + 4.017705465443957e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913552 + 4.01770942923541e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466927 + 4.017711410580344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243641 + 4.0177124014812036e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813197 + 4.0177128962754484e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076164 + 4.017713144050695e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048246 + 4.017713268531504e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034258 + 4.0177133298539106e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027292 + 4.017713361066432e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023797 + 4.0177133763873215e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802205 + 4.0177133838212516e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021197 + 4.017713388135749e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020788 + 4.017713390072709e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020526 + 4.0177133913919845e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020418 + 4.0177133912904967e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802037 + 4.0177133915480693e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.017713391415091e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133920497103e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020347 + 4.0177133923604184e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133924400543e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391908543e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.017713391868354e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391771528e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713392202162e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.0177133916711914e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392165301e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133916340363e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133923067414e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.01771339199568e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.017713391948571e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133916063833e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133917490793e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133922907833e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392113418e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133921952475e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.017713391783348e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133915968976e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   99, time 16739.37 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889438 + 3.1920904571247646e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359869 + 3.66848412957672e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909372 + 3.8628653618373953e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134196 + 3.9461803258549017e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653340517518 + 3.9835520942619126e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432229 + 4.0010507837538185e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545738 + 4.009488685693382e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532766247921 + 4.013627948517704e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297691 + 4.0156774298196855e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532697667088 + 4.016697104336011e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348525 + 4.0172056724484553e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532680518515 + 4.0174596376917614e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602927 + 4.017586540997979e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532676231167 + 4.017649973130093e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166002 + 4.017681684286261e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593177 + 4.017697538117978e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806755 + 4.0177054650070674e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913543 + 4.0177094291309284e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466925 + 4.017711410606252e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243621 + 4.0177124014094557e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131977 + 4.0177128963555237e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076153 + 4.0177131447359087e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048249 + 4.0177132681688394e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674803427 + 4.0177133303850553e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027288 + 4.0177133609025364e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023775 + 4.0177133766622636e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022063 + 4.017713384281814e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021186 + 4.0177133880082954e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802075 + 4.017713390361211e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020522 + 4.0177133912123e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020413 + 4.017713391482143e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802038 + 4.0177133914311206e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.017713391367407e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802035 + 4.0177133922144143e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133921985727e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392505157e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713391853279e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133916486593e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.0177133920338396e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.01771339253395e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392294032e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133919482824e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133918435256e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133917673715e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.0177133920708914e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.017713392199334e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392263457e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133920230097e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133917258653e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392593847e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713392517001e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.017713391799734e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133919385976e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  100, time 17690.67 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889422 + 3.1920904569089856e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359877 + 3.66848412950229e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290939 + 3.862865362314421e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134213 + 3.946180326118065e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653340517518 + 3.983552094322691e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432229 + 4.001050783868626e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532857654575 + 4.009488686185116e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479173 + 4.013627948120011e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297687 + 4.0156774303749596e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670836 + 4.0166971051521147e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234854 + 4.0172056723508994e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185166 + 4.0174596377188575e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532677660293 + 4.01758654131618e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311666 + 4.017649973384745e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755165993 + 4.01768168466633e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593186 + 4.0176975388288307e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806755 + 4.017705464955885e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913523 + 4.0177094292340684e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466945 + 4.0177114104300517e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243643 + 4.0177124014445197e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131993 + 4.01771289604109e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674807615 + 4.017713144811395e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048246 + 4.017713268534067e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034258 + 4.017713330197885e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027296 + 4.0177133604366905e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023828 + 4.0177133760369283e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022054 + 4.0177133845504184e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021206 + 4.017713388297285e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802077 + 4.017713390036723e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020549 + 4.017713391444656e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802041 + 4.017713391512357e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802037 + 4.017713392014463e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.0177133918667284e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020349 + 4.0177133918752025e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133923534267e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133911141497e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391936695e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.01771339232718e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392383325e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.0177133921269534e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392256941e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.0177133918858904e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.0177133922981345e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713391948334e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133921574206e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392163107e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392200641e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.0177133920420764e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713392275525e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133922184483e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133923530815e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391189125e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133923377545e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  101, time 18653.54 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889416 + 3.192090457404865e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359877 + 3.668484129714831e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290937 + 3.862865362370842e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134185 + 3.9461803262252494e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751806 + 3.9835520934979084e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533040343223 + 4.00105078364683e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545764 + 4.009488686012043e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532766247918 + 4.013627948029592e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297702 + 4.0156774293976297e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670832 + 4.0166971047554476e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348538 + 4.017205671982491e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185186 + 4.01745963797431e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602936 + 4.017586541306905e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532676231166 + 4.0176499737413e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166006 + 4.0176816841633454e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593182 + 4.017697538579762e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806758 + 4.017705465568931e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913514 + 4.0177094292407076e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466916 + 4.0177114107929397e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243632 + 4.017712401358603e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131977 + 4.017712896568063e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674807615 + 4.0177131445312343e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048233 + 4.017713268828504e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674803427 + 4.0177133298114864e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027294 + 4.0177133609566295e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023784 + 4.017713376783572e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022038 + 4.017713384360685e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802118 + 4.0177133874757165e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020753 + 4.017713389989979e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802052 + 4.0177133913327636e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020418 + 4.0177133920561224e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802037 + 4.017713390938067e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.01771339209431e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713391603326e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133918818294e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133918295405e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.017713391978656e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713392456414e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133920323387e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391922619e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133929833127e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391833583e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133918795054e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391888304e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392114007e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133918365974e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.0177133920763585e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020298 + 4.017713391047534e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391946241e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133918283663e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392369387e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391661735e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713392131865e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  102, time 19621.52 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889435 + 3.192090456744373e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359864 + 3.6684841292121804e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909378 + 3.8628653617533617e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134202 + 3.9461803263441156e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751815 + 3.983552094218564e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432202 + 4.001050783624244e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532857654574 + 4.0094886853675665e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479186 + 4.0136279489506555e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653272052977 + 4.015677429907282e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670865 + 4.0166971048029413e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348542 + 4.0172056722368617e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185164 + 4.017459638017117e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602916 + 4.017586541255165e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311688 + 4.017649973705143e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166024 + 4.017681684723358e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593164 + 4.0176975383088176e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806746 + 4.0177054662596243e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913539 + 4.017709429025449e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466934 + 4.0177114102596795e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243626 + 4.017712401321285e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131948 + 4.017712897386314e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076153 + 4.0177131436878817e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048229 + 4.01771326779504e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034284 + 4.017713330436682e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027279 + 4.017713361400392e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023784 + 4.0177133767716664e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022056 + 4.0177133841332266e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021188 + 4.0177133878228767e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020755 + 4.0177133904417914e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802053 + 4.017713390998584e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020426 + 4.0177133919247385e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020358 + 4.017713392407456e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020349 + 4.0177133924108645e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713391924486e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133924918017e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133913129584e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713391885626e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133917336283e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.0177133920568275e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133920849903e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.01771339219497e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.01771339271388e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133913344646e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133918159716e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713392525361e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713392183246e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392449657e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.0177133920928227e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391591457e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392170068e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392047749e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713392216814e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391744468e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  103, time 20591.49 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889424 + 3.19209045783103e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359869 + 3.6684841297812044e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290939 + 3.862865361757557e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134213 + 3.946180326134942e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751813 + 3.98355209313869e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533040343222 + 4.001050783408988e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532857654576 + 4.0094886860066004e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479169 + 4.013627948327984e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297698 + 4.015677429909817e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670858 + 4.016697104968752e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348558 + 4.0172056726878676e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185146 + 4.017459638618351e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602943 + 4.017586541436769e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311672 + 4.0176499741747023e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166015 + 4.017681684691763e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593162 + 4.017697538212475e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806753 + 4.017705465607975e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913543 + 4.017709429120597e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466927 + 4.0177114103267873e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243623 + 4.017712401388619e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131973 + 4.0177128968611537e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076146 + 4.017713144220637e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674804827 + 4.017713268418511e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674803428 + 4.017713330077289e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802728 + 4.01771336148163e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023802 + 4.017713376668539e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802205 + 4.0177133845269105e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802118 + 4.0177133878635374e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020748 + 4.017713390404565e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020535 + 4.017713390978856e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020435 + 4.017713391920223e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802037 + 4.017713391356653e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133916256676e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133923023125e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.017713392080327e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.01771339236983e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713391412578e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020347 + 4.0177133925560957e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713392260228e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133916433907e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713391968177e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392428685e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391497181e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.017713392270559e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391471604e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133918991523e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713392748505e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133915722034e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133924010193e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133930546096e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133917096123e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133916881585e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133919584875e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  104, time 21568.48 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889444 + 3.1920904575142076e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359869 + 3.6684841299600766e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909378 + 3.862865361735182e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134207 + 3.9461803263164033e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751815 + 3.9835520947758313e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432215 + 4.001050783851117e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545738 + 4.0094886859756663e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479188 + 4.0136279485870476e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297705 + 4.015677429859677e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670854 + 4.0166971051197067e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348536 + 4.0172056726154394e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185142 + 4.017459638502438e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602945 + 4.0175865414802934e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532676231166 + 4.0176499736939344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755165993 + 4.0176816842699725e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593173 + 4.017697539027381e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806746 + 4.017705465930172e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674891352 + 4.0177094295038574e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466918 + 4.017711410573713e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243621 + 4.017712401089244e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813198 + 4.017712896131708e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076149 + 4.017713143535778e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048229 + 4.017713267949575e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034275 + 4.0177133299186763e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027305 + 4.017713361260357e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802381 + 4.017713376565506e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022038 + 4.0177133847695373e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480212 + 4.0177133882970706e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020744 + 4.0177133901876794e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802053 + 4.017713391227922e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802043 + 4.0177133915264356e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020375 + 4.01771339178465e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.0177133920101226e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020347 + 4.017713391919115e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133918236833e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133920812247e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133926696124e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713391888192e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391618174e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713391690393e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133922570804e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392038161e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392082502e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.01771339213154e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392421197e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133921570077e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.01771339205529e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391632634e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392136813e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391444836e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802029 + 4.017713392234761e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.0177133919123925e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133916046204e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  105, time 22548.74 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889433 + 3.192090457626866e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359864 + 3.6684841293760675e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909378 + 3.862865362028786e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653413231342 + 3.946180326180189e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751802 + 3.9835520947887284e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432237 + 4.0010507831840036e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545744 + 4.009488685462327e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479202 + 4.0136279478186235e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297674 + 4.0156774288718504e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670858 + 4.0166971048360497e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348531 + 4.0172056723778917e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185173 + 4.017459637747503e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532677660295 + 4.017586541703012e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311666 + 4.0176499733615687e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755165993 + 4.017681684429412e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593164 + 4.017697538423453e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674980675 + 4.017705465650913e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913528 + 4.017709429112024e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674846693 + 4.017711410795091e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243626 + 4.017712401076524e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131966 + 4.017712896510878e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076126 + 4.0177131442765807e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674804823 + 4.0177132680152935e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034286 + 4.017713330260842e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027288 + 4.0177133606669056e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023793 + 4.017713376648366e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802205 + 4.0177133845879827e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802119 + 4.0177133871497565e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020753 + 4.0177133904085903e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802054 + 4.017713391064292e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020442 + 4.017713391935679e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480204 + 4.017713391241257e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020358 + 4.0177133917661433e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133920269024e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713392089405e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713391976606e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713391887469e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713392105737e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133919633823e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.0177133920345977e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391885508e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133925019936e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.0177133919985146e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.017713392351788e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133917260754e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.0177133918935704e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392092426e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133921466554e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.0177133920348004e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133921001887e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020295 + 4.017713392674153e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713392059284e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133912585884e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  106, time 23533.46 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889438 + 3.1920904578289226e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976538399335987 + 3.66848412969558e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290939 + 3.862865361533583e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134205 + 3.9461803261011246e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533405175183 + 3.983552094114105e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432224 + 4.001050784090797e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545746 + 4.009488685269072e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479188 + 4.013627947892207e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297691 + 4.015677429409281e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670847 + 4.0166971050225033e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348553 + 4.017205672170918e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532680518515 + 4.0174596387098685e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602943 + 4.017586541384207e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311672 + 4.017649972926432e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166002 + 4.0176816844981775e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593164 + 4.0176975385892967e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806764 + 4.0177054661544953e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913534 + 4.017709428710873e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674846693 + 4.017711410228837e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243637 + 4.017712401128182e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813198 + 4.0177128963870556e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674807614 + 4.0177131446385656e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048253 + 4.017713268548801e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034269 + 4.017713330032409e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027285 + 4.017713361314945e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023802 + 4.0177133763930157e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022067 + 4.017713384165953e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021184 + 4.017713388180975e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802075 + 4.017713390285739e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020546 + 4.017713390823838e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020418 + 4.017713391678594e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020373 + 4.0177133920494716e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802036 + 4.017713392284448e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392063511e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133920475996e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020347 + 4.017713391743809e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133921857766e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133919946685e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133923779515e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391839753e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392146849e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133920060373e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.017713391951602e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713392532393e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.01771339192806e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.01771339159794e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391833895e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.0177133916135995e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392703657e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392157128e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133924158017e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713392553054e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.01771339181247e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  107, time 24528.05 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889422 + 3.192090457057173e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976538399335987 + 3.6684841293232206e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909394 + 3.862865361776713e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134187 + 3.946180325860872e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751802 + 3.9835520944611114e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432237 + 4.0010507834189605e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545744 + 4.009488685710029e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479186 + 4.013627948354955e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532720529767 + 4.0156774295427227e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670843 + 4.016697104845595e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348522 + 4.017205672948231e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185155 + 4.0174596384509803e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602923 + 4.017586540803772e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311672 + 4.0176499733867727e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166013 + 4.0176816838080643e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593164 + 4.0176975386051637e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806753 + 4.017705465981028e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913532 + 4.017709429400316e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466923 + 4.017711410435374e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674824362 + 4.017712401653062e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131973 + 4.0177128961231344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076137 + 4.0177131438344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048222 + 4.0177132685059797e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034266 + 4.0177133303389174e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027299 + 4.017713360665872e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023793 + 4.017713376355194e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022056 + 4.017713384242243e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802121 + 4.0177133884412907e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020773 + 4.017713390264861e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020533 + 4.017713391507471e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020446 + 4.017713391273048e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802037 + 4.0177133919368855e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133921128513e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133919275385e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133920533997e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133920237175e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133913885535e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133919122607e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.0177133917439833e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133919194705e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713392031756e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392065534e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391976764e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392122489e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391551006e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713391510216e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133922097657e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133925667885e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392511678e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392266856e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713392133751e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133921378907e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133921268184e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  108, time 25526.75 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889442 + 3.1920904561885856e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359873 + 3.668484129421501e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909347 + 3.862865361569168e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134196 + 3.946180325841923e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751828 + 3.98355209408134e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533040343226 + 4.0010507841850277e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545738 + 4.009488685862927e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479184 + 4.013627948519299e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297698 + 4.015677429444784e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670856 + 4.016697105278877e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348542 + 4.017205672565761e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185168 + 4.0174596381856684e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532677660293 + 4.017586541888927e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311672 + 4.017649973427282e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166004 + 4.017681684015323e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593182 + 4.017697538228634e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806762 + 4.017705465705481e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913534 + 4.017709428793481e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674846692 + 4.0177114108648565e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243628 + 4.0177124011510016e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131964 + 4.01771289638419e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076146 + 4.0177131443503187e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048235 + 4.017713268384465e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034282 + 4.017713329582878e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027268 + 4.01771336137948e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023784 + 4.0177133770248917e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802204 + 4.0177133847358344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021161 + 4.0177133877312076e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020746 + 4.0177133909825356e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020524 + 4.017713390888602e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020433 + 4.0177133913270287e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020386 + 4.017713392455257e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.017713391927398e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392356737e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713391909052e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020347 + 4.017713392686009e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.01771339211637e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392210686e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020298 + 4.017713391727154e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.01771339171197e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.01771339224871e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133921364216e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713391610672e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133925861373e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713391859088e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392014461e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.0177133922626803e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.01771339173054e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.01771339175142e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133919936653e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713391974272e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713391834171e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020295 + 4.017713391904765e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  109, time 26536.86 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.976543838488944 + 3.1920904574921593e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359862 + 3.668484130183799e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909394 + 3.8628653622379985e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134178 + 3.946180326615417e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751808 + 3.983552094572315e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432235 + 4.001050784029897e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545758 + 4.009488685651535e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479193 + 4.0136279479108755e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297696 + 4.0156774297823e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670847 + 4.016697105019503e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234854 + 4.017205672849461e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185173 + 4.01745963794368e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532677660292 + 4.01758654131228e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311666 + 4.0176499733034135e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755165995 + 4.017681684040352e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593186 + 4.017697539120485e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806755 + 4.0177054654142176e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674891353 + 4.017709429017982e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466936 + 4.017711410621045e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243637 + 4.0177124022263856e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131975 + 4.01771289610265e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076133 + 4.017713144759576e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674804822 + 4.0177132680877387e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034286 + 4.0177133301861873e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027279 + 4.01771336173643e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480238 + 4.017713377103501e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022032 + 4.0177133842591354e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021188 + 4.0177133888062343e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020753 + 4.017713390099362e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020535 + 4.017713390678749e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020418 + 4.017713391380122e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020389 + 4.017713392337958e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.0177133910939517e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391811025e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802035 + 4.017713392301744e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.017713391366513e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392014341e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133919247825e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133923691386e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133920043935e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133923185903e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133919463453e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.0177133919179665e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392674234e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391792867e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391291913e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133922362217e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.01771339204797e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133922096095e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392221663e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392039265e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713391984952e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391958285e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  110, time 27552.33 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889427 + 3.192090457182703e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359877 + 3.6684841299684765e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909394 + 3.8628653614980035e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653413231342 + 3.9461803262327493e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751788 + 3.9835520937866884e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533040343223 + 4.0010507842460564e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545749 + 4.0094886858117077e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479153 + 4.0136279481714916e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297691 + 4.015677429565292e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532697667086 + 4.0166971047193974e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234853 + 4.017205671907705e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185148 + 4.0174596378320984e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602945 + 4.0175865415398276e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311686 + 4.017649973305402e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166015 + 4.0176816845482197e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593193 + 4.0176975387182664e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806764 + 4.017705465446787e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913545 + 4.017709428920272e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674846694 + 4.0177114103331745e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243626 + 4.0177124013493894e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131977 + 4.017712896414575e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076164 + 4.017713144265657e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048238 + 4.017713268192831e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034266 + 4.017713329905668e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027292 + 4.01771336130845e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802381 + 4.01771337650568e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022067 + 4.0177133848494824e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021192 + 4.0177133877082853e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020753 + 4.0177133908259727e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020553 + 4.017713390849857e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020435 + 4.017713391921444e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020389 + 4.017713391510052e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802035 + 4.017713392169011e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133920150196e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391944198e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133920575247e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713392217388e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392127341e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391616187e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133923724945e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392595967e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133918176815e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391919647e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133919403885e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392281448e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392008892e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392341968e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713391891087e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020353 + 4.017713392128089e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713392397177e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713391468232e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133924434287e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713392450124e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  111, time 28564.34 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889413 + 3.1920904571193144e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359869 + 3.6684841295415317e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909376 + 3.86286536211685e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653413231342 + 3.9461803256645337e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751828 + 3.983552094392647e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432249 + 4.001050784338646e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545746 + 4.009488685953263e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479195 + 4.0136279481151713e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297702 + 4.015677429776973e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670856 + 4.016697104802015e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348551 + 4.017205672382546e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185182 + 4.0174596382539494e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602936 + 4.017586541444056e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532676231166 + 4.017649973716172e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675516602 + 4.017681684210928e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675159316 + 4.0176975387507226e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806738 + 4.017705465720456e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913539 + 4.017709429656753e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466918 + 4.017711410927582e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674824362 + 4.017712401743735e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813197 + 4.017712896655998e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674807615 + 4.0177131443856643e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048224 + 4.017713268235673e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034284 + 4.017713329712163e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027288 + 4.0177133619114924e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023804 + 4.0177133757100715e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022063 + 4.0177133844304695e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021184 + 4.0177133878774145e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020757 + 4.0177133896784786e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020542 + 4.017713390904283e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020418 + 4.017713391338506e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020362 + 4.017713391789911e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020373 + 4.0177133924328726e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391617205e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391544519e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.017713392071941e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133923201404e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713391823762e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392176792e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391731872e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392498374e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713392463224e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133927425097e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133920657537e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391912846e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133911187777e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133921037145e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392152836e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020355 + 4.017713391788305e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.017713391604913e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392028037e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133915386773e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133917100003e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  112, time 29583.68 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889435 + 3.192090457224191e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359884 + 3.6684841294707e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909387 + 3.8628653625984936e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134185 + 3.9461803261868364e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751833 + 3.9835520944167947e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432224 + 4.0010507839076254e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532857654575 + 4.009488686242485e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479197 + 4.0136279484862665e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297678 + 4.0156774298527124e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670854 + 4.0166971050421127e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348542 + 4.017205671883417e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185153 + 4.0174596388070126e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602934 + 4.0175865413631667e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311686 + 4.017649973189203e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675516601 + 4.017681684307781e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593153 + 4.017697538584584e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674980675 + 4.017705465493174e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913528 + 4.0177094285872434e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466945 + 4.017711410563632e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243626 + 4.0177124015092785e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131968 + 4.017712896673724e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076142 + 4.017713143992945e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674804825 + 4.0177132688612645e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034284 + 4.0177133299502114e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027279 + 4.017713360921745e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023808 + 4.017713376566554e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802204 + 4.0177133842160865e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021184 + 4.0177133886707577e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020744 + 4.0177133903797626e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020537 + 4.017713391428267e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020426 + 4.0177133911600123e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020375 + 4.017713392226585e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713391798277e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133922796395e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391762103e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133924522654e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020298 + 4.0177133924706534e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133921648994e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.01771339197522e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133912755466e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392762777e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.01771339198342e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133917886744e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133920799473e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133921999216e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.0177133916062187e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.0177133917063306e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392470574e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020349 + 4.0177133922743493e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020289 + 4.0177133923586433e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.017713392325364e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392016991e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133919282484e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  113, time 30607.99 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889429 + 3.1920904571463073e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976538399335987 + 3.668484129768224e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909378 + 3.862865361854977e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134196 + 3.9461803260697844e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751833 + 3.983552094627589e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432229 + 4.0010507837242153e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545775 + 4.0094886859579765e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479195 + 4.0136279480802524e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297678 + 4.0156774299968457e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670863 + 4.0166971047838047e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234853 + 4.0172056723977233e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532680518516 + 4.0174596389042123e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602938 + 4.0175865416894756e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311666 + 4.0176499737604503e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675516601 + 4.0176816837312707e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593184 + 4.017697538749928e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806744 + 4.0177054657565296e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913525 + 4.01770942876223e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466931 + 4.0177114103649234e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674824362 + 4.0177124015067713e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813195 + 4.017712896386776e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076155 + 4.01771314438437e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048242 + 4.0177132677175906e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034289 + 4.0177133304908185e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027274 + 4.0177133606893e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802379 + 4.017713376805612e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022054 + 4.0177133844201616e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021195 + 4.0177133884582673e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020755 + 4.017713390786353e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020513 + 4.0177133909472784e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020435 + 4.017713391927356e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480204 + 4.017713391984624e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.017713392205343e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133924599935e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133918957054e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133918991423e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133922569555e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713392253854e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392146167e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133921270974e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133925165525e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133928803214e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713391604669e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133920642983e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392153223e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.017713391708638e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133922220863e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133918907873e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020298 + 4.0177133921495093e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133923021245e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020298 + 4.017713392019548e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391701095e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133922600736e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  114, time 31637.28 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.976543838488942 + 3.1920904569099115e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976538399335989 + 3.6684841296925365e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909365 + 3.8628653616128427e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134211 + 3.9461803268308197e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751802 + 3.9835520939217055e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432233 + 4.0010507835084617e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545733 + 4.0094886860717616e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479188 + 4.0136279483354195e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297705 + 4.015677429580399e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532697667084 + 4.016697104871625e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348522 + 4.0172056725751577e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185168 + 4.017459637740582e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532677660293 + 4.017586541561768e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311692 + 4.0176499737872266e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166017 + 4.017681684577102e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675159318 + 4.017697538841416e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806769 + 4.017705465114481e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913534 + 4.017709428591542e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466931 + 4.0177114096765275e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243632 + 4.017712401558549e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131973 + 4.0177128970990927e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076142 + 4.017713144412378e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048206 + 4.0177132684207444e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674803427 + 4.017713329754735e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027285 + 4.0177133608113735e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802381 + 4.017713377196868e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022054 + 4.0177133841554257e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802118 + 4.017713387662356e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020766 + 4.0177133913697763e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020524 + 4.0177133908737634e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020453 + 4.017713392095277e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020364 + 4.017713392247184e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020349 + 4.017713391951956e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133913853697e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.01771339184805e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133921151717e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391953973e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133918499604e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133919756743e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713391857977e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391370996e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133919287646e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133913695016e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.0177133922472903e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133920244105e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133922718966e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392024768e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133919220487e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713391954307e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133926714606e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392478843e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.0177133917605555e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392030678e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  115, time 32670.38 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889418 + 3.1920904570061964e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359866 + 3.6684841296668137e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290938 + 3.862865361818261e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134205 + 3.9461803264075266e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751806 + 3.9835520942076686e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432233 + 4.001050783748032e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545729 + 4.0094886858806885e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532766247918 + 4.0136279486890335e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532720529771 + 4.015677430024937e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670836 + 4.0166971049782967e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348547 + 4.017205672511135e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185177 + 4.0174596388723194e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602947 + 4.0175865414988603e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311675 + 4.017649973768357e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166015 + 4.0176816848241285e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532675159317 + 4.017697539003318e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806755 + 4.017705465736885e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674891353 + 4.017709428660174e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466931 + 4.017711409848283e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243623 + 4.0177124019004473e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813197 + 4.0177128962954013e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076149 + 4.01771314429264e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674804823 + 4.017713268393304e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034262 + 4.017713330543071e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027285 + 4.0177133611091823e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023815 + 4.0177133766555975e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022067 + 4.017713384337299e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021197 + 4.0177133882202013e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020744 + 4.017713389997521e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020533 + 4.017713391485516e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802044 + 4.017713391670724e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020378 + 4.0177133917139893e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713392972458e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713392388406e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392692543e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.01771339180769e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133919275343e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133917867553e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133925889924e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133916414505e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133912938816e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392064386e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392187366e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713391941634e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713391857473e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391660118e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133919453013e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713391900363e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.017713391901052e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020349 + 4.0177133922893486e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133919950396e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133920854165e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391988989e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  116, time 33712.35 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889427 + 3.192090457707526e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359857 + 3.6684841293266934e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909383 + 3.8628653615860055e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134187 + 3.9461803264408034e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751808 + 3.983552094072936e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432215 + 4.0010507837325787e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532857654574 + 4.0094886857553826e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479193 + 4.013627948324401e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297685 + 4.015677430366601e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532697667086 + 4.016697104631417e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348542 + 4.0172056724392994e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185164 + 4.017459638532038e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602927 + 4.017586541235623e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532676231167 + 4.0176499736306615e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166006 + 4.017681684434237e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593177 + 4.0176975386499606e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806746 + 4.017705465179129e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913525 + 4.017709429511083e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466923 + 4.0177114104754283e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267482436 + 4.017712401496238e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131973 + 4.0177128965278626e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674807616 + 4.0177131442101966e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048222 + 4.017713268809406e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034286 + 4.0177133299776473e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027248 + 4.0177133606856425e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023802 + 4.0177133765979394e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022056 + 4.017713384890574e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021188 + 4.017713388102103e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020748 + 4.01771339031899e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020535 + 4.0177133914284566e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020418 + 4.0177133916984235e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020384 + 4.0177133918462175e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020355 + 4.017713391401066e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133924442254e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133926332064e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392155191e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.0177133916403356e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.0177133923077197e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392362618e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133917668135e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713392556759e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.0177133917163796e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133913288715e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133920310655e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713391799546e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391115042e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133921517344e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.0177133924250814e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392253779e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392476301e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020342 + 4.017713391665894e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133916487636e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392635055e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  117, time 34758.07 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889409 + 3.192090456579901e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359897 + 3.668484129258013e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976535574290937 + 3.862865362019514e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976534132313418 + 3.9461803263860353e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751824 + 3.9835520936264833e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432224 + 4.0010507839103116e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545755 + 4.0094886863278663e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653276624792 + 4.0136279484726023e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297674 + 4.015677429451247e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326976670863 + 4.0166971048720327e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326862348522 + 4.017205672223357e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185153 + 4.0174596386587427e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602927 + 4.01758654113737e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311692 + 4.0176499732738134e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755166015 + 4.0176816847261823e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593169 + 4.017697538351166e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806742 + 4.017705465641403e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674891354 + 4.0177094286002474e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466954 + 4.0177114105954646e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243623 + 4.017712401818422e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131962 + 4.017712896423992e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076149 + 4.0177131438867984e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048244 + 4.017713268362066e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034275 + 4.017713329902319e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802729 + 4.017713360498941e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802381 + 4.0177133765303705e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022045 + 4.0177133840749253e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802118 + 4.0177133883771565e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802075 + 4.017713390358406e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020557 + 4.017713390921267e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020413 + 4.0177133914556406e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802039 + 4.017713391610746e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.017713392145815e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.017713392375248e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.017713391547876e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020307 + 4.0177133920849697e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391982554e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713391571963e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133922357315e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.017713391487576e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133915898164e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133919401297e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133923671333e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802031 + 4.017713391906876e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133914082295e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133918826643e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020302 + 4.017713391825318e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.017713391975171e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133913863e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802032 + 4.0177133921865077e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.017713392440298e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133917400997e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713391900719e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  118, time 35809.65 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889429 + 3.1920904573019655e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359893 + 3.668484129946765e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909376 + 3.8628653620436023e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976534132313422 + 3.946180326559879e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751813 + 3.9835520947285054e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765330403432222 + 4.00105078362572e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545722 + 4.009488685831998e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479184 + 4.0136279487196506e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532720529771 + 4.015677429599157e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532697667084 + 4.0166971054256693e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234851 + 4.017205672023135e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185146 + 4.017459638460668e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532677660292 + 4.017586541494598e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311668 + 4.017649973864134e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755165995 + 4.0176816840229486e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593149 + 4.0176975383727834e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326749806742 + 4.017705465515683e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913536 + 4.017709429014194e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466923 + 4.017711409948753e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243623 + 4.0177124015419555e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748131984 + 4.0177128966611687e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076142 + 4.017713144148655e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048238 + 4.017713268309465e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034284 + 4.0177133304463006e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802731 + 4.0177133615435186e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802378 + 4.0177133768504204e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022078 + 4.017713384120178e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021184 + 4.017713387888853e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020757 + 4.017713390196895e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020526 + 4.0177133910861516e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020435 + 4.017713390954652e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020382 + 4.017713392052759e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020347 + 4.0177133914208386e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133918451423e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133925714477e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133924783106e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.0177133925654496e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.01771339187024e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.017713391373452e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391865766e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133920478765e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391950114e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020324 + 4.0177133922345917e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133915174274e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020333 + 4.0177133922258164e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391421111e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802029 + 4.017713392017219e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020329 + 4.0177133925092257e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133922245924e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392455523e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020344 + 4.0177133923190615e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.0177133914741387e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392153549e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter  119, time 36858.78 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02, α = 0.00e+00, m = 20, nfg = 53
┌ Warning: Expectation value is not real: -1.9765438384889422 + 3.192090457380872e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765383993359864 + 3.668484129367996e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765355742909385 + 3.862865361892343e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765341323134185 + 3.9461803260709533e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765334051751817 + 3.983552094796592e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976533040343222 + 4.0010507838300695e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765328576545735 + 4.0094886860410387e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327662479195 + 4.0136279481470404e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765327205297691 + 4.0156774297012084e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532697667085 + 4.0166971048962774e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532686234854 + 4.017205672264876e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326805185162 + 4.0174596377972054e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326776602934 + 4.01758654106779e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326762311666 + 4.0176499737907873e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326755165988 + 4.017681684244187e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326751593162 + 4.0176975379157387e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674980678 + 4.017705465367824e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748913523 + 4.017709428606136e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748466927 + 4.01771141062375e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748243617 + 4.017712401181856e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674813197 + 4.017712896319649e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748076155 + 4.0177131443224756e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748048229 + 4.017713268970845e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748034273 + 4.017713330025806e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748027294 + 4.017713361130762e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748023815 + 4.017713376375593e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748022054 + 4.01771338429269e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748021215 + 4.017713387821688e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020753 + 4.0177133909326803e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802054 + 4.017713390919655e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020444 + 4.017713391952071e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020378 + 4.017713391850404e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802034 + 4.0177133920376046e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020338 + 4.0177133917515277e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.017713392045142e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020315 + 4.0177133918416785e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020313 + 4.0177133919804146e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713392435248e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020304 + 4.017713392131584e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391736293e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020327 + 4.0177133919301956e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713391731623e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.976532674802033 + 4.0177133923038626e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.017713391741981e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713391880915e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713391746255e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.0177133924569664e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020295 + 4.0177133921968426e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020318 + 4.017713391936403e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020335 + 4.017713392741555e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020322 + 4.017713392480737e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.9765326748020309 + 4.017713391569112e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning: Expectation value is not real: -1.97653267480203 + 4.017713391838146e-7im.
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/toolbox.jl:50
┌ Warning:   Linesearch bisection failure: [a, b] = [0.00e+00, 2.22e-16], b-a = 2.22e-16, dϕᵃ = -9.98e-06, dϕᵇ = -7.73e-06, (ϕᵇ - ϕᵃ)/(b-a) = 1.09e+11
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:338
┌ Warning: Linesearch not converged after 1 iterations and 53 function evaluations:
│ α = 0.00e+00, dϕ = -9.98e-06, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
┌ Warning: LBFGS: not converged to requested tol after 120 iterations and time 37920.23 s: f = -1.976556886920, ‖∇f‖ = 2.5198e-02
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
E_opt = -0.4941392217299078
(E_opt - E_ref) / abs(E_ref) = 0.00022413408212889108

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

