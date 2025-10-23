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
    trscheme = truncerror(1.0e-10) & truncrank(Dbond)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    global peps, wts, = simpleupdate(
        peps, H, alg, wts; bipartite = false, check_interval = 2000
    )
end
````

````
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>2, 1=>2)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.316e+00,  time = 27.486 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>2, 1=>2)
[ Info: SU conv 1045   :  dt = 1e-02,  weight diff = 9.843e-08,  time = 34.956 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>6, 1=>6)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 6.459e-06,  time = 0.077 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>6, 1=>6)
[ Info: SU conv 584    :  dt = 1e-02,  weight diff = 9.946e-08,  time = 42.322 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 5.245e-03,  time = 0.235 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU iter 2000   :  dt = 1e-03,  weight diff = 1.418e-07,  time = 0.018 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU conv 3791   :  dt = 1e-03,  weight diff = 9.990e-09,  time = 78.783 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU iter 1      :  dt = 4e-04,  weight diff = 3.256e-04,  time = 0.018 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU iter 2000   :  dt = 4e-04,  weight diff = 1.888e-08,  time = 0.024 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU conv 3034   :  dt = 4e-04,  weight diff = 9.997e-09,  time = 62.113 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 1.627e-04,  time = 0.024 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU iter 2000   :  dt = 1e-04,  weight diff = 1.532e-08,  time = 0.018 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>3, 1=>5)
[ Info: SU conv 2916   :  dt = 1e-04,  weight diff = 9.997e-09,  time = 59.560 sec

````

## Computing the ground-state energy

In order to compute the energy expectation value with evolved PEPS, we need to converge a
CTMRG environment on it. We first converge an environment with a small enviroment dimension
and then use that to initialize another run with bigger environment dimension. We'll use
`trscheme=truncrank(χ)` for that such that the dimension is increased during the second CTMRG
run:

````julia
χenv₀, χenv = 6, 16
env_space = Vect[fℤ₂](0 => χenv₀ / 2, 1 => χenv₀ / 2)
normalize!.(peps.A, Inf)
env = CTMRGEnv(rand, Float64, peps, env_space)
for χ in [χenv₀, χenv]
    global env, = leading_boundary(
        env, peps; alg = :sequential, tol = 1.0e-8, maxiter = 50, trscheme = truncrank(χ)
    )
end
````

````
[ Info: CTMRG init:	obj = +4.034556135739e-13	err = 1.0000e+00
┌ Warning: CTMRG cancel 50:	obj = +1.777694990783e+00	err = 2.1447151954e-06	time = 18.75 sec
└ @ PEPSKit ~/PEPSKit.jl/src/algorithms/ctmrg/ctmrg.jl:152
[ Info: CTMRG init:	obj = +1.777694990783e+00	err = 1.0000e+00
[ Info: CTMRG conv 7:	obj = +1.781063096355e+00	err = 3.5793745596e-10	time = 21.32 sec

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

