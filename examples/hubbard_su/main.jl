using Markdown #hide
md"""
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
"""

using Random
using TensorKit, PEPSKit
Random.seed!(12329348592498);

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
physical_space = Vect[fℤ₂](0 => 2, 1 => 2);

md"""
## Running the simple update algorithm

Suppose the goal is to use imaginary-time simple update to optimize a PEPS 
with bond dimension D = 8, and $2 \times 2$ unit cells.
For a challenging model like the Hubbard model, a naive evolution starting from a 
random PEPS at D = 8 will almost always produce a sub-optimal state.
In this example, we shall demonstrate some common practices to improve SU result.

First, we shall use a small D for the random PEPS initialization, which is chosen as 4 here.
For convenience, here we work with real tensors with `Float64` entries.
The bond weights are still initialized as identity matrices. 
"""

virtual_space = Vect[fℤ₂](0 => 2, 1 => 2)
peps = InfinitePEPS(rand, Float64, physical_space, virtual_space; unitcell=(Nr, Nc));
wts = SUWeight(peps)

md"""
Starting from the random state, we first use a relatively large evolution time step 
`dt = 1e-2`. After convergence at D = 4, to avoid stucking at some bad local minimum,
we first increase D to 12, and drop it back to D = 8 after a while. 
Afterwards, we keep D = 8 and gradually decrease `dt` to `1e-4` to improve convergence.
"""

dts = [1e-2, 1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-7, 1e-7, 1e-8, 1e-8, 1e-8]
Ds = [4, 12, 8, 8, 8]
maxiter = 20000

for (dt, tol, Dbond) in zip(dts, tols, Ds)
    trscheme = truncerr(1e-10) & truncdim(Dbond)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    global peps, wts, = simpleupdate(
        peps, wts, H, alg; bipartite=false, check_interval=2000
    )
end

md"""
## Computing the ground-state energy

In order to compute the energy expectation value with evolved PEPS, we need to converge a
CTMRG environment on it. We first converge an environment with a small enviroment dimension
and then use that to initialize another run with bigger environment dimension. We'll use
`trscheme=truncdim(χ)` for that such that the dimension is increased during the second CTMRG
run:
"""

χenv₀, χenv = 6, 16
env_space = Vect[fℤ₂](0 => χenv₀ / 2, 1 => χenv₀ / 2)
normalize!.(peps.A, Inf)
env = CTMRGEnv(rand, Float64, peps, env_space)
for χ in [χenv₀, χenv]
    global env, = leading_boundary(
        env, peps; alg=:sequential, tol=1e-8, maxiter=50, trscheme=truncdim(χ)
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
@show (E - E_exact) / abs(E_exact);
