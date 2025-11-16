```@meta
EditURL = "../../../../examples/hubbard_su/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/hubbard_su/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/hubbard_su/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/hubbard_su)


# Simple update for the Fermi-Hubbard model at half-filling

Once again, we consider the Hubbard model but this time we obtain the ground-state PEPS by
imaginary time evolution. In particular, we'll use the [`SimpleUpdate`](@ref) algorithm.
As a reminder, we define the Hubbard model as

```math
H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- -
c_{i,\sigma}^- c_{j,\sigma}^+ \right) + U \sum_i n_{i,\uparrow}n_{i,\downarrow} - \mu \sum_i n_i
```

with $\sigma \in \{\uparrow,\downarrow\}$ and $n_{i,\sigma} = c_{i,\sigma}^+ c_{i,\sigma}^-$.

Let's get started by seeding the RNG and importing the required modules:

````julia
using Random
using TensorKit, PEPSKit
Random.seed!(12329348592498);
````

## Defining the Hamiltonian

First, we define the Hubbard model at $t=1$ hopping and $U=6$ using `Trivial` sectors for
the particle and spin symmetries, and set $\mu = U/2$ for half-filling. The model will be
constructed on a $2 \times 2$ unit cell, so we have:

````julia
t = 1
U = 6
Nr, Nc = 2, 2
H = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(Nr, Nc); t, U, mu = U / 2);
physical_space = Vect[fℤ₂](0 => 2, 1 => 2);
````

## Running the simple update algorithm

Suppose the goal is to use imaginary-time simple update to optimize a PEPS
with bond dimension D = 8, and $2 \times 2$ unit cells.
For a challenging model like the Hubbard model, a naive evolution starting from a
random PEPS at D = 8 will almost always produce a sub-optimal state.
In this example, we shall demonstrate some common practices to improve SU result.

First, we shall use a small D for the random PEPS initialization, which is chosen as 4 here.
For convenience, here we work with real tensors with `Float64` entries.
The bond weights are still initialized as identity matrices.

````julia
virtual_space = Vect[fℤ₂](0 => 2, 1 => 2)
peps = InfinitePEPS(rand, Float64, physical_space, virtual_space; unitcell = (Nr, Nc));
wts = SUWeight(peps);
````

Starting from the random state, we first use a relatively large evolution time step
`dt = 1e-2`. After convergence at D = 4, to avoid stucking at some bad local minimum,
we first increase D to 12, and drop it back to D = 8 after a while.
Afterwards, we keep D = 8 and gradually decrease `dt` to `1e-4` to improve convergence.

````julia
dts = [1.0e-2, 1.0e-2, 1.0e-3, 4.0e-4, 1.0e-4]
tols = [1.0e-7, 1.0e-7, 1.0e-8, 1.0e-8, 1.0e-8]
Ds = [4, 12, 8, 8, 8]
maxiter = 20000

for (dt, tol, Dbond) in zip(dts, tols, Ds)
    trunc = truncerror(; atol = 1.0e-10) & truncrank(Dbond)
    alg = SimpleUpdate(; trunc, bipartite = false)
    global peps, wts, = time_evolve(peps, H, dt, maxiter, alg, wts; tol, check_interval = 2000)
end
````

````
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 2, 1 => 2)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 1.316e+00. Time = 21.601 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 2, 1 => 2)
[ Info: SU iter 1045   : dt = 0.01, |Δλ| = 9.843e-08. Time = 0.012 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 35.64 s
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 6, 1 => 6)
[ Info: SU iter 1      : dt = 0.01, |Δλ| = 6.459e-06. Time = 0.134 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 6, 1 => 6)
[ Info: SU iter 584    : dt = 0.01, |Δλ| = 9.946e-08. Time = 0.114 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 71.51 s
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 1      : dt = 0.001, |Δλ| = 5.245e-03. Time = 0.373 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 2000   : dt = 0.001, |Δλ| = 1.418e-07. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 3791   : dt = 0.001, |Δλ| = 9.990e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 131.97 s
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 1      : dt = 0.0004, |Δλ| = 3.256e-04. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 2000   : dt = 0.0004, |Δλ| = 1.888e-08. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 3034   : dt = 0.0004, |Δλ| = 9.997e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 105.93 s
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 1      : dt = 0.0001, |Δλ| = 1.627e-04. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 2000   : dt = 0.0001, |Δλ| = 1.532e-08. Time = 0.037 s/it
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0 => 3, 1 => 5)
[ Info: SU iter 2916   : dt = 0.0001, |Δλ| = 9.997e-09. Time = 0.037 s/it
[ Info: SU: bond weights have converged.
[ Info: Simple update finished. Total time elasped: 100.89 s

````

## Computing the ground-state energy

In order to compute the energy expectation value with evolved PEPS, we need to converge a
CTMRG environment on it. We first converge an environment with a small enviroment dimension,
which is initialized using the simple update bond weights. Next we use it to initialize
another run with bigger environment dimension. The dynamic adjustment of environment dimension
is achieved by using `trunc=truncrank(χ)` with different `χ`s in the CTMRG runs:

````julia
χenv₀, χenv = 6, 16
env_space = Vect[fℤ₂](0 => χenv₀ / 2, 1 => χenv₀ / 2)
normalize!.(peps.A, Inf)
env = CTMRGEnv(wts, peps)
for χ in [χenv₀, χenv]
    global env, = leading_boundary(
        env, peps; alg = :sequential, tol = 1.0e-8, maxiter = 50, trunc = truncrank(χ)
    )
end
````

````
[ Info: CTMRG init:	obj = +3.208695223790e-01	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +1.777694992786e+00	err = 2.2836831592e-09	time = 8.64 sec
[ Info: CTMRG init:	obj = +1.777694992786e+00	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +1.781063096355e+00	err = 3.5793721430e-10	time = 37.81 sec

````

We measure the energy by computing the `H` expectation value, where we have to make sure to
normalize with respect to the unit cell to obtain the energy per site:

````julia
E = expectation_value(peps, H, env) / (Nr * Nc)
@show E;
````

````
E = -3.652497562261351

````

Finally, we can compare the obtained ground-state energy against the literature, namely the
QMC estimates from [Qin et al.](@cite qin_benchmark_2016). We find that the results generally
agree:

````julia
Es_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)
E_exact = Es_exact[U] - U / 2
@show (E - E_exact) / abs(E_exact);
````

````
(E - E_exact) / abs(E_exact) = 0.001149243235334783

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

