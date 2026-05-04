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
Random.seed!(29385294);
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
[ Info: --- Time evolution (simple update), dt = 0.01 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1/2 => 1, -1/2 => 1)
[ Info: SU iter 1      : |Δλ| = 1.190e+00. Time = 118.568 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1373   : |Δλ| = 9.898e-09. Time = 0.069 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 225.44 s
[ Info: --- Time evolution (simple update), dt = 0.01 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : |Δλ| = 2.985e-04. Time = 0.089 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 523    : |Δλ| = 9.955e-09. Time = 0.070 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 38.09 s
[ Info: --- Time evolution (simple update), dt = 0.01 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : |Δλ| = 3.001e-04. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 610    : |Δλ| = 9.971e-09. Time = 0.070 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 44.48 s
[ Info: --- Time evolution (simple update), dt = 0.01 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : |Δλ| = 3.021e-04. Time = 0.084 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 740    : |Δλ| = 9.953e-09. Time = 0.070 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 53.86 s
[ Info: --- Time evolution (simple update), dt = 0.01 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : |Δλ| = 3.089e-04. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1140   : |Δλ| = 1.000e-08. Time = 0.070 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 83.30 s

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
[ Info: --- Time evolution (simple update), dt = 0.001 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : |Δλ| = 7.604e-04. Time = 0.138 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 500    : |Δλ| = 1.692e-06. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1000   : |Δλ| = 1.002e-06. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1500   : |Δλ| = 6.304e-07. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2000   : |Δλ| = 4.078e-07. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2500   : |Δλ| = 2.688e-07. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3000   : |Δλ| = 1.797e-07. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3500   : |Δλ| = 1.217e-07. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 4000   : |Δλ| = 8.352e-08. Time = 0.084 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 4500   : |Δλ| = 5.817e-08. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 5000   : |Δλ| = 4.132e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 5500   : |Δλ| = 3.002e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 6000   : |Δλ| = 2.216e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 6500   : |Δλ| = 1.651e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 7000   : |Δλ| = 1.235e-08. Time = 0.084 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 7500   : |Δλ| = 9.271e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 8000   : |Δλ| = 6.980e-09. Time = 0.086 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 8500   : |Δλ| = 5.278e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 9000   : |Δλ| = 4.021e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 9500   : |Δλ| = 3.090e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 10000  : |Δλ| = 2.394e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 10500  : |Δλ| = 1.878e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 11000  : |Δλ| = 1.498e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 11500  : |Δλ| = 1.209e-09. Time = 0.071 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 11962  : |Δλ| = 9.999e-10. Time = 0.070 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 870.98 s
[ Info: --- Time evolution (simple update), dt = 0.0001 ---
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1      : |Δλ| = 7.683e-05. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 500    : |Δλ| = 3.277e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1000   : |Δλ| = 2.995e-08. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 1500   : |Δλ| = 2.752e-08. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2000   : |Δλ| = 2.540e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 2500   : |Δλ| = 2.355e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3000   : |Δλ| = 2.193e-08. Time = 0.071 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 3500   : |Δλ| = 2.049e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 4000   : |Δλ| = 1.920e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 4500   : |Δλ| = 1.805e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 5000   : |Δλ| = 1.700e-08. Time = 0.071 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 5500   : |Δλ| = 1.605e-08. Time = 0.087 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 6000   : |Δλ| = 1.518e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 6500   : |Δλ| = 1.438e-08. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 7000   : |Δλ| = 1.363e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 7500   : |Δλ| = 1.294e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 8000   : |Δλ| = 1.230e-08. Time = 0.073 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 8500   : |Δλ| = 1.170e-08. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 9000   : |Δλ| = 1.114e-08. Time = 0.084 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 9500   : |Δλ| = 1.060e-08. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 10000  : |Δλ| = 1.011e-08. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 10500  : |Δλ| = 9.635e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 11000  : |Δλ| = 9.190e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 11500  : |Δλ| = 8.770e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 12000  : |Δλ| = 8.372e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 12500  : |Δλ| = 7.995e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 13000  : |Δλ| = 7.637e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 13500  : |Δλ| = 7.298e-09. Time = 0.093 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 14000  : |Δλ| = 6.976e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 14500  : |Δλ| = 6.670e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 15000  : |Δλ| = 6.379e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 15500  : |Δλ| = 6.102e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 16000  : |Δλ| = 5.839e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 16500  : |Δλ| = 5.588e-09. Time = 0.071 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 17000  : |Δλ| = 5.349e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 17500  : |Δλ| = 5.122e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 18000  : |Δλ| = 4.905e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 18500  : |Δλ| = 4.699e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 19000  : |Δλ| = 4.502e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 19500  : |Δλ| = 4.314e-09. Time = 0.086 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 20000  : |Δλ| = 4.134e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 20500  : |Δλ| = 3.963e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 21000  : |Δλ| = 3.800e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 21500  : |Δλ| = 3.644e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 22000  : |Δλ| = 3.494e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 22500  : |Δλ| = 3.352e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 23000  : |Δλ| = 3.216e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 23500  : |Δλ| = 3.085e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 24000  : |Δλ| = 2.961e-09. Time = 0.086 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 24500  : |Δλ| = 2.842e-09. Time = 0.070 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 25000  : |Δλ| = 2.728e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 25500  : |Δλ| = 2.619e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 26000  : |Δλ| = 2.515e-09. Time = 0.085 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 26500  : |Δλ| = 2.415e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 27000  : |Δλ| = 2.319e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 27500  : |Δλ| = 2.228e-09. Time = 0.084 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 28000  : |Δλ| = 2.140e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 28500  : |Δλ| = 2.056e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 29000  : |Δλ| = 1.976e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 29500  : |Δλ| = 1.899e-09. Time = 0.069 s/it
[ Info: Space of x-weight at [1, 1] = Rep[U₁](0 => 2, 1 => 1, -1 => 1)
[ Info: SU iter 30000  : |Δλ| = 1.825e-09. Time = 0.069 s/it
┌ Warning: SU: bond weights have not converged.
└ @ PEPSKit ~/git/PEPSKit.jl/src/algorithms/time_evolution/simpleupdate.jl:241
[ Info: Time evolution finished in 2182.89 s

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
-0.4908450911219574
````

Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package](https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:

````julia
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref);
````

````
(E - E_ref) / abs(E_ref) = 0.006889041736049819

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

In our optimization, we will use a fixed-point differentiation scheme which requires a
gauge fixing of the contraction environment
(specified by the `gradient_alg = (; iterscheme = :fixed)` setting).
Since this gauge fixing involves potentially complex phases, we have to convert our
real-valued contraction environment to complex numbers before the optimization.

````julia
using MPSKit: randomize!

noise_peps = InfinitePEPS(randomize!.(deepcopy(peps.A)))
peps₀ = peps + 1.0e-1noise_peps
peps_opt, env_opt, E_opt, = fixedpoint(
    H, peps₀, complex(env);
    optimizer_alg = (; tol = 1.0e-4, maxiter = 80), gradient_alg = (; iterscheme = :fixed)
);
````

````
[ Info: LBFGS: initializing with f = -1.887578587634e+00, ‖∇f‖ = 7.8780e-01
[ Info: LBFGS: iter    1, Δt  2.11 m: f = -1.926388092775e+00, ‖∇f‖ = 4.7092e-01, α = 1.19e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    2, Δt 30.68 s: f = -1.936546032899e+00, ‖∇f‖ = 2.1792e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, Δt 29.68 s: f = -1.942713411326e+00, ‖∇f‖ = 2.0267e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 29.00 s: f = -1.952873152491e+00, ‖∇f‖ = 1.6713e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt 26.25 s: f = -1.958619448466e+00, ‖∇f‖ = 1.5871e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt 27.27 s: f = -1.961555293087e+00, ‖∇f‖ = 9.2861e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt 25.52 s: f = -1.962811912498e+00, ‖∇f‖ = 6.7900e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 27.71 s: f = -1.965000830793e+00, ‖∇f‖ = 5.5994e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 26.97 s: f = -1.966386885688e+00, ‖∇f‖ = 5.8303e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 27.50 s: f = -1.967582741430e+00, ‖∇f‖ = 1.1340e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 24.91 s: f = -1.968967360870e+00, ‖∇f‖ = 4.1612e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 24.17 s: f = -1.969387768033e+00, ‖∇f‖ = 3.6625e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 22.91 s: f = -1.970335449656e+00, ‖∇f‖ = 3.8759e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 27.81 s: f = -1.971409090676e+00, ‖∇f‖ = 4.5864e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 28.11 s: f = -1.972264675328e+00, ‖∇f‖ = 3.3125e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 30.00 s: f = -1.972963700423e+00, ‖∇f‖ = 3.8311e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 27.60 s: f = -1.973393808483e+00, ‖∇f‖ = 3.4025e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 26.40 s: f = -1.973719445599e+00, ‖∇f‖ = 2.4697e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt 28.38 s: f = -1.974117003995e+00, ‖∇f‖ = 2.3402e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt 27.75 s: f = -1.974537198653e+00, ‖∇f‖ = 2.3338e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, Δt 27.06 s: f = -1.975099516920e+00, ‖∇f‖ = 2.6565e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, Δt 26.82 s: f = -1.975286936081e+00, ‖∇f‖ = 4.4455e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt 25.91 s: f = -1.975701217338e+00, ‖∇f‖ = 1.5220e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt 28.52 s: f = -1.975821697001e+00, ‖∇f‖ = 1.8130e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt 26.42 s: f = -1.975915354964e+00, ‖∇f‖ = 2.1138e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt 24.35 s: f = -1.975998069768e+00, ‖∇f‖ = 1.3443e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt 24.80 s: f = -1.976153039607e+00, ‖∇f‖ = 1.4680e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt 23.66 s: f = -1.976260274804e+00, ‖∇f‖ = 1.7700e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt 24.93 s: f = -1.976336675890e+00, ‖∇f‖ = 3.6212e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt 22.98 s: f = -1.976481030702e+00, ‖∇f‖ = 1.2667e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt 23.03 s: f = -1.976539819784e+00, ‖∇f‖ = 1.2939e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt 23.95 s: f = -1.976636245246e+00, ‖∇f‖ = 1.3135e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt 48.36 s: f = -1.976684971058e+00, ‖∇f‖ = 1.9116e-02, α = 4.62e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   34, Δt 23.65 s: f = -1.976743262496e+00, ‖∇f‖ = 1.1658e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt 23.80 s: f = -1.976804909822e+00, ‖∇f‖ = 9.8461e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt 23.74 s: f = -1.976867380046e+00, ‖∇f‖ = 1.1271e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt 24.66 s: f = -1.976958913613e+00, ‖∇f‖ = 1.1353e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt 24.14 s: f = -1.977004149888e+00, ‖∇f‖ = 3.0196e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt 24.29 s: f = -1.977132482812e+00, ‖∇f‖ = 8.3199e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt 23.70 s: f = -1.977168854066e+00, ‖∇f‖ = 6.5281e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt 24.26 s: f = -1.977230989059e+00, ‖∇f‖ = 9.2541e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt 24.18 s: f = -1.977285220461e+00, ‖∇f‖ = 1.3935e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt 23.53 s: f = -1.977362590244e+00, ‖∇f‖ = 1.0743e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt 23.86 s: f = -1.977429195025e+00, ‖∇f‖ = 8.6810e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt 25.41 s: f = -1.977458751818e+00, ‖∇f‖ = 2.5717e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt 22.06 s: f = -1.977509209358e+00, ‖∇f‖ = 1.3339e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt 22.96 s: f = -1.977531778536e+00, ‖∇f‖ = 1.0403e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt 22.65 s: f = -1.977584620950e+00, ‖∇f‖ = 1.0581e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt 23.16 s: f = -1.977638977445e+00, ‖∇f‖ = 1.1478e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt 24.26 s: f = -1.977695226616e+00, ‖∇f‖ = 1.1637e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt 22.61 s: f = -1.977738682348e+00, ‖∇f‖ = 1.8362e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt 21.76 s: f = -1.977786348227e+00, ‖∇f‖ = 8.4144e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt 23.05 s: f = -1.977834226107e+00, ‖∇f‖ = 9.5424e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt 24.76 s: f = -1.977872709228e+00, ‖∇f‖ = 1.0069e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 23.20 s: f = -1.977896046364e+00, ‖∇f‖ = 1.1177e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, Δt 22.34 s: f = -1.977948855869e+00, ‖∇f‖ = 1.5958e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt 22.56 s: f = -1.977994579851e+00, ‖∇f‖ = 8.9051e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt 22.35 s: f = -1.978026072493e+00, ‖∇f‖ = 6.8546e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt 24.53 s: f = -1.978061626606e+00, ‖∇f‖ = 9.8392e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt 24.11 s: f = -1.978102554569e+00, ‖∇f‖ = 9.4732e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 22.77 s: f = -1.978139870452e+00, ‖∇f‖ = 9.1003e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt 23.50 s: f = -1.978176133536e+00, ‖∇f‖ = 6.4191e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt 23.05 s: f = -1.978213215021e+00, ‖∇f‖ = 6.9446e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt 24.10 s: f = -1.978234971062e+00, ‖∇f‖ = 1.5980e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt 25.47 s: f = -1.978268225326e+00, ‖∇f‖ = 8.5546e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt 22.02 s: f = -1.978286888113e+00, ‖∇f‖ = 6.4777e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt 23.79 s: f = -1.978300997146e+00, ‖∇f‖ = 8.0692e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt 22.57 s: f = -1.978300528984e+00, ‖∇f‖ = 5.5147e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 23.63 s: f = -1.978328958743e+00, ‖∇f‖ = 6.3994e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt 24.58 s: f = -1.978331153308e+00, ‖∇f‖ = 6.5489e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt 24.48 s: f = -1.978363716954e+00, ‖∇f‖ = 5.9050e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt 24.34 s: f = -1.978377069104e+00, ‖∇f‖ = 1.1874e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt 23.47 s: f = -1.978401095295e+00, ‖∇f‖ = 5.3880e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 23.53 s: f = -1.978411402203e+00, ‖∇f‖ = 3.9472e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt 23.03 s: f = -1.978423812150e+00, ‖∇f‖ = 5.1734e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt 23.97 s: f = -1.978437405148e+00, ‖∇f‖ = 7.4640e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt 23.49 s: f = -1.978449959375e+00, ‖∇f‖ = 4.6023e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 23.47 s: f = -1.978465209487e+00, ‖∇f‖ = 4.0286e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt 23.17 s: f = -1.978478118790e+00, ‖∇f‖ = 6.5526e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 80 iterations and time 52.05 m: f = -1.978491610275e+00, ‖∇f‖ = 7.4246e-03
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
E_opt = -0.4946229025687655
(E_opt - E_ref) / abs(E_ref) = -0.0007544816768143398

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

