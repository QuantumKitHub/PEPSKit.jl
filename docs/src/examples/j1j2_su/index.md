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
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 1.189e+00. Time = 33.154 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1833   : dt = 0.01, |Δλ| = 9.859e-09. Time = 0.063 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 158.43 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.401e-04. Time = 0.065 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 523    : dt = 0.01, |Δλ| = 9.965e-09. Time = 0.062 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 34.62 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.526e-04. Time = 0.062 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 611    : dt = 0.01, |Δλ| = 9.848e-09. Time = 0.062 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 40.33 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.664e-04. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 735    : dt = 0.01, |Δλ| = 9.963e-09. Time = 0.064 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 48.49 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 3.828e-04. Time = 0.064 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 901    : dt = 0.01, |Δλ| = 9.995e-09. Time = 0.063 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 59.48 s

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
[ Info: SU iter 1      : dt = 0.001, |Δλ| = 4.477e-04. Time = 0.121 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 500    : dt = 0.001, |Δλ| = 2.767e-08. Time = 0.063 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1000   : dt = 0.001, |Δλ| = 9.954e-09. Time = 0.062 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1500   : dt = 0.001, |Δλ| = 5.019e-09. Time = 0.063 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2000   : dt = 0.001, |Δλ| = 3.015e-09. Time = 0.092 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2500   : dt = 0.001, |Δλ| = 1.935e-09. Time = 0.062 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3000   : dt = 0.001, |Δλ| = 1.273e-09. Time = 0.063 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3295   : dt = 0.001, |Δλ| = 9.994e-10. Time = 0.062 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 217.43 s
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : dt = 0.0001, |Δλ| = 4.467e-05. Time = 0.063 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 500    : dt = 0.0001, |Δλ| = 1.150e-09. Time = 0.063 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 873    : dt = 0.0001, |Δλ| = 9.998e-10. Time = 0.063 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 57.50 s

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
-0.4908483447932549
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.006882458688406928

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
└ @ PEPSKit ~/Projects/PEPSKit.jl/src/algorithms/optimization/peps_optimization.jl:204
[ Info: LBFGS: initializing with f = -1.907301302110e+00, ‖∇f‖ = 5.5641e-01
[ Info: LBFGS: iter    1, Δt 27.56 s: f = -1.912496200062e+00, ‖∇f‖ = 4.8528e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt 22.77 s: f = -1.939590317765e+00, ‖∇f‖ = 3.1781e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, Δt 19.06 s: f = -1.948086619481e+00, ‖∇f‖ = 1.8688e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 18.14 s: f = -1.954903534354e+00, ‖∇f‖ = 1.0567e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt 19.58 s: f = -1.958636003807e+00, ‖∇f‖ = 9.6554e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt 18.80 s: f = -1.961414208875e+00, ‖∇f‖ = 8.8495e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt 18.79 s: f = -1.963670567806e+00, ‖∇f‖ = 5.9165e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 20.23 s: f = -1.965776363520e+00, ‖∇f‖ = 5.0139e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 21.98 s: f = -1.967226453690e+00, ‖∇f‖ = 9.2909e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 19.64 s: f = -1.968251645234e+00, ‖∇f‖ = 4.4439e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 22.41 s: f = -1.969059008087e+00, ‖∇f‖ = 4.6917e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 20.79 s: f = -1.969667913862e+00, ‖∇f‖ = 4.8179e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 21.23 s: f = -1.970804652416e+00, ‖∇f‖ = 3.2505e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 22.79 s: f = -1.971787694409e+00, ‖∇f‖ = 4.3869e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 24.41 s: f = -1.972414025039e+00, ‖∇f‖ = 4.0604e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 22.68 s: f = -1.972867447250e+00, ‖∇f‖ = 2.5133e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 22.72 s: f = -1.973224221322e+00, ‖∇f‖ = 2.3593e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 22.61 s: f = -1.973780793633e+00, ‖∇f‖ = 2.7945e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt 20.78 s: f = -1.974278639630e+00, ‖∇f‖ = 2.8914e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt 19.12 s: f = -1.974533659938e+00, ‖∇f‖ = 1.8380e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, Δt 22.48 s: f = -1.974797746482e+00, ‖∇f‖ = 1.5608e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, Δt 21.72 s: f = -1.975002265713e+00, ‖∇f‖ = 2.0961e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt 20.82 s: f = -1.975178140945e+00, ‖∇f‖ = 3.4077e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt 20.78 s: f = -1.975348043297e+00, ‖∇f‖ = 1.4875e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt 21.24 s: f = -1.975446214398e+00, ‖∇f‖ = 1.3359e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt 20.62 s: f = -1.975598188521e+00, ‖∇f‖ = 1.5129e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt 21.33 s: f = -1.975648975504e+00, ‖∇f‖ = 4.0666e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt 19.24 s: f = -1.975801502894e+00, ‖∇f‖ = 1.2082e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt 19.95 s: f = -1.975838520962e+00, ‖∇f‖ = 1.0012e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt 20.60 s: f = -1.975920699081e+00, ‖∇f‖ = 1.1497e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt 20.98 s: f = -1.975994476122e+00, ‖∇f‖ = 2.0164e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt 19.27 s: f = -1.976049779798e+00, ‖∇f‖ = 1.2778e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt 19.69 s: f = -1.976083052474e+00, ‖∇f‖ = 8.1251e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt 19.55 s: f = -1.976120284177e+00, ‖∇f‖ = 9.1433e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt 19.89 s: f = -1.976178863096e+00, ‖∇f‖ = 1.2556e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt 19.56 s: f = -1.976225564245e+00, ‖∇f‖ = 1.1295e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt 19.28 s: f = -1.976262568889e+00, ‖∇f‖ = 7.0514e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt 18.97 s: f = -1.976300953764e+00, ‖∇f‖ = 8.6312e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt 18.69 s: f = -1.976337659332e+00, ‖∇f‖ = 1.1092e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt 19.72 s: f = -1.976393924161e+00, ‖∇f‖ = 1.1668e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt 20.68 s: f = -1.976436192483e+00, ‖∇f‖ = 8.0157e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt 18.14 s: f = -1.976469672103e+00, ‖∇f‖ = 7.3417e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt 19.70 s: f = -1.976509489620e+00, ‖∇f‖ = 8.4507e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt 19.19 s: f = -1.976583802578e+00, ‖∇f‖ = 1.3151e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt 20.87 s: f = -1.976630307258e+00, ‖∇f‖ = 1.4170e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt 20.33 s: f = -1.976680877868e+00, ‖∇f‖ = 8.3860e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt 19.29 s: f = -1.976710020540e+00, ‖∇f‖ = 1.0325e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt 20.24 s: f = -1.976745581904e+00, ‖∇f‖ = 1.2062e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt 19.17 s: f = -1.976829231643e+00, ‖∇f‖ = 1.2197e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt 20.52 s: f = -1.976899195992e+00, ‖∇f‖ = 1.9229e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt 19.66 s: f = -1.976987140901e+00, ‖∇f‖ = 1.8244e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt 18.09 s: f = -1.977023236629e+00, ‖∇f‖ = 8.3070e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt 19.51 s: f = -1.977056164969e+00, ‖∇f‖ = 8.3182e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt 21.42 s: f = -1.977123528338e+00, ‖∇f‖ = 1.1023e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 40.71 s: f = -1.977157909182e+00, ‖∇f‖ = 1.7552e-02, α = 3.55e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   56, Δt 19.58 s: f = -1.977212923858e+00, ‖∇f‖ = 1.1229e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt 20.60 s: f = -1.977268389200e+00, ‖∇f‖ = 7.8373e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt 19.16 s: f = -1.977326972617e+00, ‖∇f‖ = 1.1772e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt 19.77 s: f = -1.977371513954e+00, ‖∇f‖ = 2.0292e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt 19.22 s: f = -1.977420127940e+00, ‖∇f‖ = 1.0167e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 19.18 s: f = -1.977459871700e+00, ‖∇f‖ = 8.8652e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt 19.88 s: f = -1.977507028354e+00, ‖∇f‖ = 8.3742e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt 21.56 s: f = -1.977570888464e+00, ‖∇f‖ = 1.5706e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt 38.57 s: f = -1.977620567166e+00, ‖∇f‖ = 1.0020e-02, α = 4.86e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   65, Δt 20.72 s: f = -1.977658416479e+00, ‖∇f‖ = 8.1197e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt 20.85 s: f = -1.977708104067e+00, ‖∇f‖ = 1.1151e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt 19.31 s: f = -1.977753273984e+00, ‖∇f‖ = 8.6127e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt 22.56 s: f = -1.977756230819e+00, ‖∇f‖ = 1.1803e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 21.49 s: f = -1.977778298956e+00, ‖∇f‖ = 1.3834e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt 19.10 s: f = -1.977826121915e+00, ‖∇f‖ = 1.0827e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt 19.06 s: f = -1.977853878453e+00, ‖∇f‖ = 9.0049e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt 21.55 s: f = -1.977879275990e+00, ‖∇f‖ = 8.2484e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt 17.90 s: f = -1.977902757838e+00, ‖∇f‖ = 6.2376e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 20.15 s: f = -1.977930234553e+00, ‖∇f‖ = 5.6595e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt 20.50 s: f = -1.977964320717e+00, ‖∇f‖ = 1.1578e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt 18.53 s: f = -1.977994766836e+00, ‖∇f‖ = 7.7846e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt 19.37 s: f = -1.978013673463e+00, ‖∇f‖ = 7.3610e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 18.64 s: f = -1.978027144104e+00, ‖∇f‖ = 6.3493e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt 19.93 s: f = -1.978044594980e+00, ‖∇f‖ = 7.3623e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 34.78 m: f = -1.978065459455e+00, ‖∇f‖ = 5.8505e-03
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/lbfgs.jl:199

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
E_opt = -0.49451636486378536
(E_opt - E_ref) / abs(E_ref) = -0.0005389273925854121

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

