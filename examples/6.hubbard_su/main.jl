using Markdown #hide
md"""
# Hubbard model imaginary time evolution using simple update

Once again, we consider the Hubbard model but this time we obtain the ground-state PEPS by
imaginary time evoluation. In particular, we'll use the [`SimpleUpdate`](@ref) algorithm.
As a reminder, we define the Hubbard model as

```math
H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- +
c_{i,\sigma}^- c_{j,\sigma}^+ \right) + U \sum_i n_{i,\uparrow}n_{i,\downarrow} - \mu \sum_i n_i
```

with $\sigma \in \{\uparrow,\downarrow\}$ and $n_{i,\sigma} = c_{i,\sigma}^+ c_{i,\sigma}^-$.

Let's get started by seeding the RNG and importing the required modules:
"""

using Random
using TensorKit, PEPSKit
Random.seed!(1298351928);

md"""
## Defining the Hamiltonian

First, we define the Hubbard model at $t=1$ hopping and $U=6$ using `Trivial` sectors for
the particle and spin symmetries, and set $\mu = U/2$ for half-filling. The model will be
constructed on a $2 \times 2$ unit cell, so we have:
"""

t = 1
U = 6
Nr, Nc = 2, 2
H = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(Nr, Nc); t, U, mu=U / 2);

md"""
## Running the simple update algorithm

Next, we'll specify the virtual PEPS bond dimension and define the fermionic physical and
virtual spaces. The simple update algorithm evolves an infinite PEPS with weights on the
virtual bonds, so we here need to intialize an [`InfiniteWeightPEPS`](@ref). By default,
the bond weights will be identity. Unlike in the other examples, we here use tensors with
real `Float64` entries:
"""

Dbond = 8
physical_space = Vect[fℤ₂](0 => 2, 1 => 2)
virtual_space = Vect[fℤ₂](0 => Dbond / 2, 1 => Dbond / 2)
wpeps = InfiniteWeightPEPS(rand, Float64, physical_space, virtual_space; unitcell=(Nr, Nc));

md"""
Before starting the simple update routine, we normalize the vertex tensors of `wpeps` by
dividing with the maximal vertex element (using the infinity norm):
"""

for ind in CartesianIndices(wpeps.vertices)
    wpeps.vertices[ind] /= norm(wpeps.vertices[ind], Inf)
end

md"""
Let's set algorithm parameters: The plan is to successively decrease the time interval of
the Trotter-Suzuki as well as the convergence tolerance such that we obtain a more accurate
result at each iteration. To run the simple update, we call [`simpleupdate`](@ref) where we
use the keyword `bipartite=false` - meaning that we use the full $2 \times 2$ unit cell
without assuming a bipartite structure. Thus, we can start evolving:
"""

dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 20000

for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncerr(1e-10) & truncdim(Dbond)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    global wpeps, = simpleupdate(wpeps, H, alg; bipartite=false)
end

md"""
To obtain the evolved `InfiniteWeightPEPS` as an actual PEPS without weights on the bonds,
we can just call the following constructor:
"""

peps = InfinitePEPS(wpeps);

md"""
## Computing the ground-state energy

In order to compute the energy expectation value with evolved PEPS, we need to converge a
CTMRG environment on it. We first converge an environment with a small enviroment dimension
and then use that to initialize another run with bigger environment dimension. We'll use
`trscheme=truncdim(χ)` for that such that the dimension is increased during the second CTMRG
run:
"""

χenv₀, χenv = 6, 20
env_space = Vect[fℤ₂](0 => χenv₀ / 2, 1 => χenv₀ / 2)

env = CTMRGEnv(rand, Float64, peps, env_space)
for χ in [χenv₀, χenv]
    global env, = leading_boundary(
        env, peps; alg=:sequential, tol=1e-5, trscheme=truncdim(χ)
    )
end

md"""
We measure the energy by computing the `H` expectation value, where we have to make sure to
normalize with respect to the unit cell to obtain the energy per site:
"""

E = expectation_value(peps, H, env) / (Nr * Nc)
@show E;

md"""
Finally, we can compare the obtained ground-state energy against the literature, namely the
QMC estimates from [Qin et al.](@cite qin_benchmark_2016). We find that the results generally
agree:
"""

Es_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)
E_exact = Es_exact[U] - U / 2
@show (E - E_exact) / E_exact;
