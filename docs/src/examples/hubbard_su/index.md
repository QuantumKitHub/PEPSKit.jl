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
H = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(Nr, Nc); t, U, mu=U / 2);
````

## Running the simple update algorithm

Next, we'll specify the virtual PEPS bond dimension and define the fermionic physical and
virtual spaces. The simple update algorithm evolves an infinite PEPS with weights on the
virtual bonds, so we here need to intialize an [`InfiniteWeightPEPS`](@ref). By default,
the bond weights will be identity. Unlike in the other examples, we here use tensors with
real `Float64` entries:

````julia
Dbond = 8
physical_space = Vect[fℤ₂](0 => 2, 1 => 2)
virtual_space = Vect[fℤ₂](0 => Dbond / 2, 1 => Dbond / 2)
wpeps = InfiniteWeightPEPS(rand, Float64, physical_space, virtual_space; unitcell=(Nr, Nc));
````

Let's set the algorithm parameters: The plan is to successively decrease the time interval of
the Trotter-Suzuki as well as the convergence tolerance such that we obtain a more accurate
result at each iteration. To run the simple update, we call [`simpleupdate`](@ref) where we
use the keyword `bipartite=false` - meaning that we use the full $2 \times 2$ unit cell
without assuming a bipartite structure. Thus, we can start evolving:

````julia
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 20000

for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncerr(1e-10) & truncdim(Dbond)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    global wpeps, = simpleupdate(wpeps, H, alg; bipartite=false)
end
````

````
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 2.355e+00,  time = 13.262 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 500    :  dt = 1e-02,  weight diff = 3.984e-04,  time = 0.033 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1000   :  dt = 1e-02,  weight diff = 2.866e-06,  time = 0.031 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU conv 1061   :  dt = 1e-02,  weight diff = 9.956e-07,  time = 31.894 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 6.070e-03,  time = 0.038 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 500    :  dt = 1e-03,  weight diff = 1.874e-06,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1000   :  dt = 1e-03,  weight diff = 6.437e-07,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1500   :  dt = 1e-03,  weight diff = 2.591e-07,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 2000   :  dt = 1e-03,  weight diff = 1.053e-07,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 2500   :  dt = 1e-03,  weight diff = 4.280e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 3000   :  dt = 1e-03,  weight diff = 1.741e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU conv 3309   :  dt = 1e-03,  weight diff = 9.983e-09,  time = 56.141 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1      :  dt = 4e-04,  weight diff = 4.030e-04,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 500    :  dt = 4e-04,  weight diff = 1.776e-07,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1000   :  dt = 4e-04,  weight diff = 7.091e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1500   :  dt = 4e-04,  weight diff = 3.997e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 2000   :  dt = 4e-04,  weight diff = 2.622e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 2500   :  dt = 4e-04,  weight diff = 1.796e-08,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 3000   :  dt = 4e-04,  weight diff = 1.245e-08,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU conv 3303   :  dt = 4e-04,  weight diff = 9.997e-09,  time = 55.693 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1      :  dt = 1e-04,  weight diff = 2.014e-04,  time = 0.033 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 500    :  dt = 1e-04,  weight diff = 5.664e-08,  time = 0.013 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1000   :  dt = 1e-04,  weight diff = 4.106e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 1500   :  dt = 1e-04,  weight diff = 3.033e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 2000   :  dt = 1e-04,  weight diff = 2.290e-08,  time = 0.032 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 2500   :  dt = 1e-04,  weight diff = 1.773e-08,  time = 0.012 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 3000   :  dt = 1e-04,  weight diff = 1.410e-08,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU iter 3500   :  dt = 1e-04,  weight diff = 1.152e-08,  time = 0.011 sec
[ Info: Space of x-weight at [1, 1] = Vect[FermionParity](0=>4, 1=>4)
[ Info: SU conv 3893   :  dt = 1e-04,  weight diff = 9.997e-09,  time = 65.654 sec

````

To obtain the evolved `InfiniteWeightPEPS` as an actual PEPS without weights on the bonds,
we can just call the following constructor:

````julia
peps = InfinitePEPS(wpeps);
````

## Computing the ground-state energy

In order to compute the energy expectation value with evolved PEPS, we need to converge a
CTMRG environment on it. We first converge an environment with a small enviroment dimension
and then use that to initialize another run with bigger environment dimension. We'll use
`trscheme=truncdim(χ)` for that such that the dimension is increased during the second CTMRG
run:

````julia
χenv₀, χenv = 6, 16
env_space = Vect[fℤ₂](0 => χenv₀ / 2, 1 => χenv₀ / 2)

env = CTMRGEnv(rand, Float64, peps, env_space)
for χ in [χenv₀, χenv]
    global env, = leading_boundary(
        env, peps; alg=:sequential, tol=1e-5, trscheme=truncdim(χ)
    )
end
````

````
[ Info: CTMRG init:	obj = -3.050596045736e-08	err = 1.0000e+00
┌ Warning: CTMRG cancel 100:	obj = +6.015068543571e-01	err = 1.5525020185e-02	time = 18.35 sec
└ @ PEPSKit ~/.julia/packages/PEPSKit/2FBaz/src/algorithms/ctmrg/ctmrg.jl:155
[ Info: CTMRG init:	obj = +6.015068543571e-01	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +5.889572774193e-01	err = 8.9970586516e-06	time = 12.18 sec

````

We measure the energy by computing the `H` expectation value, where we have to make sure to
normalize with respect to the unit cell to obtain the energy per site:

````julia
E = expectation_value(peps, H, env) / (Nr * Nc)
@show E;
````

````
E = -3.633461319161078

````

Finally, we can compare the obtained ground-state energy against the literature, namely the
QMC estimates from [Qin et al.](@cite qin_benchmark_2016). We find that the results generally
agree:

````julia
Es_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)
E_exact = Es_exact[U] - U / 2
@show (E - E_exact) / E_exact;
````

````
(E - E_exact) / E_exact = -0.006355096354341832

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

