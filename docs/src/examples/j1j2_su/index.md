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
dt, tol, nstep = 1.0e-2, 1.0e-8, 30000
check_interval = 4000
trunc_peps = truncerror(; atol = 1.0e-10) & truncrank(Dbond)
alg = SimpleUpdate(; trunc = trunc_peps)
for J2 in 0.1:0.1:0.5
    # convert Hamiltonian `LocalOperator` to real floats
    H = real(
        j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice = false),
    )
    global peps, wts, = time_evolve(peps, H, dt, nstep, alg, wts; tol, check_interval)
end
````

````
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1/2 => 1, -1/2 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 1.189e+00. Time = 0.034 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1833   : dt = 0.01, |Δλ| = 9.859e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 70.90 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.401e-04. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 523    : dt = 0.01, |Δλ| = 9.965e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 21.18 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.526e-04. Time = 0.038 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 611    : dt = 0.01, |Δλ| = 9.848e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 24.83 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.664e-04. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 735    : dt = 0.01, |Δλ| = 9.963e-09. Time = 0.092 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 29.87 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.828e-04. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 901    : dt = 0.01, |Δλ| = 9.995e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 36.57 s

````

After we reach $J_2 / J_1 = 0.5$, we gradually decrease the evolution time step to obtain
a more accurately evolved PEPS:

````julia
dts = [1.0e-3, 1.0e-4]
tols = [1.0e-9, 1.0e-9]
J2 = 0.5
H = real(j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice = false))
for (dt, tol) in zip(dts, tols)
    global peps, wts, = time_evolve(peps, H, dt, nstep, alg, wts; tol)
end
````

````
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.001, |Δλ| = 4.477e-04. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 500    : dt = 0.001, |Δλ| = 2.767e-08. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1000   : dt = 0.001, |Δλ| = 9.954e-09. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1500   : dt = 0.001, |Δλ| = 5.019e-09. Time = 0.038 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2000   : dt = 0.001, |Δλ| = 3.015e-09. Time = 0.039 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2500   : dt = 0.001, |Δλ| = 1.935e-09. Time = 0.090 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3000   : dt = 0.001, |Δλ| = 1.273e-09. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3295   : dt = 0.001, |Δλ| = 9.994e-10. Time = 0.036 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 132.66 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.0001, |Δλ| = 4.467e-05. Time = 0.036 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 500    : dt = 0.0001, |Δλ| = 1.150e-09. Time = 0.035 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 873    : dt = 0.0001, |Δλ| = 9.998e-10. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elapsed: 34.62 s

````

## Computing the simple update energy estimate

Finally, we measure the ground-state energy by converging a CTMRG environment and computing
the expectation value, where we first normalize tensors in the PEPS:

````julia
normalize!.(peps.A, Inf) ## normalize each PEPS tensor by largest element
χenv = 32
trunc_env = truncerror(; atol = 1.0e-10) & truncrank(χenv)
Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1 // 2 => χenv ÷ 4, -1 // 2 => χenv ÷ 4)
env₀ = CTMRGEnv(rand, Float64, peps, Espace)
env, = leading_boundary(env₀, peps; tol = 1.0e-10, alg = :sequential, trunc = trunc_env);
E = expectation_value(peps, H, env) / (Nr * Nc)
````

````
-0.4908483447932438
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.006882458688429391

````

## Variational PEPS optimization using AD

As a last step, we will use the SU-evolved PEPS as a starting point for a [`fixedpoint`](@ref)
PEPS optimization. Note that we could have also used a sublattice-rotated version of `H` to
fit the Hamiltonian onto a single-site unit cell which would require us to optimize fewer
parameters and hence lead to a faster optimization. But here we instead take advantage of
the already evolved `peps`, thus giving us a physical initial guess for the optimization.
In order to break some of the $C_{4v}$ symmetry of the PEPS, we will add a bit of noise to it.
This is conviently done using MPSKit's `randomize!` function.
(Breaking some of the spatial symmetry can be advantageous for obtaining lower energies.)

````julia
using MPSKit: randomize!

noise_peps = InfinitePEPS(randomize!.(deepcopy(peps.A)))
peps₀ = peps + 1.0e-1noise_peps
peps_opt, env_opt, E_opt, = fixedpoint(
    H, peps₀, env;
    optimizer_alg = (; tol = 1.0e-4, maxiter = 80), gradient_alg = (; iterscheme = :diffgauge)
);
````

````
[ Info: LBFGS: initializing with f = -1.917915769323e+00, ‖∇f‖ = 4.2598e-01
[ Info: LBFGS: iter    1, Δt 16.45 s: f = -1.921875846208e+00, ‖∇f‖ = 3.6931e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt 12.76 s: f = -1.942022618990e+00, ‖∇f‖ = 2.0804e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, Δt 12.69 s: f = -1.948412125614e+00, ‖∇f‖ = 1.4224e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 11.25 s: f = -1.954734282994e+00, ‖∇f‖ = 1.6428e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt 12.79 s: f = -1.957501635760e+00, ‖∇f‖ = 1.8493e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt 12.20 s: f = -1.959162742352e+00, ‖∇f‖ = 1.0888e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt 12.06 s: f = -1.961758227820e+00, ‖∇f‖ = 9.0418e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 12.57 s: f = -1.963102797068e+00, ‖∇f‖ = 7.7890e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 12.02 s: f = -1.965635307562e+00, ‖∇f‖ = 5.7454e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 12.22 s: f = -1.967012786653e+00, ‖∇f‖ = 1.0695e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 12.75 s: f = -1.968331989207e+00, ‖∇f‖ = 4.7357e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 12.52 s: f = -1.968984356192e+00, ‖∇f‖ = 3.6819e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 13.08 s: f = -1.969738509271e+00, ‖∇f‖ = 3.8320e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 13.97 s: f = -1.970765612340e+00, ‖∇f‖ = 4.1807e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 13.32 s: f = -1.971316317211e+00, ‖∇f‖ = 4.5580e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 13.46 s: f = -1.971822370984e+00, ‖∇f‖ = 2.4262e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 12.99 s: f = -1.972203923570e+00, ‖∇f‖ = 2.4564e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 12.95 s: f = -1.972802900923e+00, ‖∇f‖ = 2.8491e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt 13.74 s: f = -1.973589666789e+00, ‖∇f‖ = 3.2039e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt 13.18 s: f = -1.973913379566e+00, ‖∇f‖ = 5.1316e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, Δt 13.55 s: f = -1.974416985201e+00, ‖∇f‖ = 1.8951e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, Δt 13.47 s: f = -1.974639779350e+00, ‖∇f‖ = 1.8591e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt 13.23 s: f = -1.974980106926e+00, ‖∇f‖ = 1.9583e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt 14.00 s: f = -1.975202056049e+00, ‖∇f‖ = 3.9045e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt 13.33 s: f = -1.975442229571e+00, ‖∇f‖ = 1.8554e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt 13.95 s: f = -1.975560352122e+00, ‖∇f‖ = 1.5857e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt 13.25 s: f = -1.975643058810e+00, ‖∇f‖ = 1.2993e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt 13.13 s: f = -1.975704724372e+00, ‖∇f‖ = 1.9944e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt 14.12 s: f = -1.975779000149e+00, ‖∇f‖ = 1.1828e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt 13.48 s: f = -1.975862495962e+00, ‖∇f‖ = 1.0766e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt 13.36 s: f = -1.975947783240e+00, ‖∇f‖ = 9.5066e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt 13.70 s: f = -1.976052804517e+00, ‖∇f‖ = 1.5333e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt 13.19 s: f = -1.976106986012e+00, ‖∇f‖ = 2.1883e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt 13.41 s: f = -1.976164433529e+00, ‖∇f‖ = 7.9599e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt 14.17 s: f = -1.976190058936e+00, ‖∇f‖ = 6.7778e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt 13.45 s: f = -1.976224201954e+00, ‖∇f‖ = 7.6168e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt 13.73 s: f = -1.976267854501e+00, ‖∇f‖ = 1.7393e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt 13.84 s: f = -1.976315102216e+00, ‖∇f‖ = 8.0912e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt 13.22 s: f = -1.976342354040e+00, ‖∇f‖ = 6.1351e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt 13.44 s: f = -1.976371404915e+00, ‖∇f‖ = 6.6216e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt 13.71 s: f = -1.976421353914e+00, ‖∇f‖ = 8.2468e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt 13.47 s: f = -1.976466860391e+00, ‖∇f‖ = 9.9248e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt 13.64 s: f = -1.976507540729e+00, ‖∇f‖ = 6.9877e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt 14.04 s: f = -1.976535673591e+00, ‖∇f‖ = 5.6235e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt 13.18 s: f = -1.976573549708e+00, ‖∇f‖ = 7.4445e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt 13.51 s: f = -1.976590425955e+00, ‖∇f‖ = 1.5662e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt 13.88 s: f = -1.976621753066e+00, ‖∇f‖ = 6.2307e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt 13.08 s: f = -1.976639047822e+00, ‖∇f‖ = 5.1405e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt 13.33 s: f = -1.976657938685e+00, ‖∇f‖ = 1.1028e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt 13.74 s: f = -1.976676808863e+00, ‖∇f‖ = 9.7274e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt 13.29 s: f = -1.976691906282e+00, ‖∇f‖ = 5.8050e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt 13.20 s: f = -1.976710655182e+00, ‖∇f‖ = 5.7333e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt 13.66 s: f = -1.976732623549e+00, ‖∇f‖ = 5.2887e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt 13.26 s: f = -1.976766233304e+00, ‖∇f‖ = 8.5088e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 13.39 s: f = -1.976781456137e+00, ‖∇f‖ = 1.1582e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, Δt 24.51 s: f = -1.976799996356e+00, ‖∇f‖ = 5.1443e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt 13.08 s: f = -1.976812132182e+00, ‖∇f‖ = 3.9200e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt 13.75 s: f = -1.976828056507e+00, ‖∇f‖ = 6.6193e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt 13.05 s: f = -1.976851081866e+00, ‖∇f‖ = 6.9020e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt 13.15 s: f = -1.976872354754e+00, ‖∇f‖ = 5.8100e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 13.76 s: f = -1.976887353263e+00, ‖∇f‖ = 1.5507e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt 13.33 s: f = -1.976913441456e+00, ‖∇f‖ = 5.4576e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt 13.19 s: f = -1.976925022985e+00, ‖∇f‖ = 4.5942e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt 13.90 s: f = -1.976945457629e+00, ‖∇f‖ = 6.1499e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt 13.54 s: f = -1.976948121976e+00, ‖∇f‖ = 1.2404e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt 13.37 s: f = -1.976990622832e+00, ‖∇f‖ = 1.0997e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt 13.89 s: f = -1.977019727681e+00, ‖∇f‖ = 5.3064e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt 13.40 s: f = -1.977038041468e+00, ‖∇f‖ = 3.9154e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 13.22 s: f = -1.977061135927e+00, ‖∇f‖ = 5.5351e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt 25.93 s: f = -1.977069437407e+00, ‖∇f‖ = 9.1542e-03, α = 2.16e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   71, Δt 11.98 s: f = -1.977082632066e+00, ‖∇f‖ = 7.0584e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt 12.86 s: f = -1.977109455547e+00, ‖∇f‖ = 6.5585e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt 12.26 s: f = -1.977124023586e+00, ‖∇f‖ = 1.0470e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 12.29 s: f = -1.977136222448e+00, ‖∇f‖ = 9.2794e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt 12.79 s: f = -1.977174289195e+00, ‖∇f‖ = 6.4735e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt 12.52 s: f = -1.977220054869e+00, ‖∇f‖ = 6.4042e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt 12.26 s: f = -1.977245223602e+00, ‖∇f‖ = 6.9440e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 12.87 s: f = -1.977264616541e+00, ‖∇f‖ = 1.5115e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt 12.17 s: f = -1.977292459833e+00, ‖∇f‖ = 6.6226e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 25.05 m: f = -1.977304651029e+00, ‖∇f‖ = 4.5558e-03
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/lbfgs.jl:199

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
E_opt = -0.49432616275726
(E_opt - E_ref) / abs(E_ref) = -0.0001540976373494541

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

