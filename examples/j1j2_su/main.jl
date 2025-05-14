using Markdown #hide
md"""
# Three-site simple update for the $J_1$-$J_2$ model

In this example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional $J_1$-$J_2$ model, which contains next-nearest neighbor interactions:

```math
H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
+ J_2 \sum_{\langle \langle i,j \rangle \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
```

Note that the $J_1$-$J_2$ model exhibits a $U(1)$ spin rotation symmetry, which we want to 
exploit here. The goal will be to calculate the energy at $J_1 = 1$ and $J_2 = 1/2$, first
using the simple update algorithm and then, to refine the energy estimate, using AD-based
variational PEPS optimization.

We first import all required modules and seed the RNG:
"""

using Random
using TensorKit, PEPSKit
Random.seed!(2025);

md"""
## Simple updating a challenging phase

Let's start by initializing an `InfiniteWeightPEPS` for which we set the required
parameters as well as physical and virtual vector spaces. Since the $J_1$-$J_2$ model has
*next*-neighbor interactions, the simple update algorithm requires a $2 \times 2$ unit cell:
"""

Dbond, χenv, symm = 4, 32, U1Irrep
trscheme_env = truncerr(1e-10) & truncdim(χenv)
Nr, Nc, J1 = 2, 2, 1.0

## random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Pspace = Vect[U1Irrep](1//2 => 1, -1//2 => 1)
Vspace = Vect[U1Irrep](0 => 2, 1//2 => 1, -1//2 => 1)
Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1//2 => χenv ÷ 4, -1//2 => χenv ÷ 4)
wpeps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc));

md"""
The value $J_2 / J_1 = 0.5$ is close to a possible spin liquid phase, which is challenging
for SU to produce a relatively good state from random initialization. Therefore, we shall
gradually increase $J_2 / J_1$ from 0.1 to 0.5, each time initializing on the previously
evolved PEPS:
"""

dt, tol, maxiter = 1e-2, 1e-8, 30000
check_interval = 4000
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)
alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
for J2 in 0.1:0.1:0.5
    H = real( ## convert Hamiltonian `LocalOperator` to real floats
        j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false)
    )
    result = simpleupdate(wpeps, H, alg; check_interval)
    global wpeps = result[1]
end

md"""
After we reach $J_2 / J_1 = 0.5$, we gradually decrease the evolution time step to obtain 
a more accurately evolved PEPS:
"""

dts = [1e-3, 1e-4]
tols = [1e-9, 1e-9]
J2 = 0.5
H = real(j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false))
for (dt, tol) in zip(dts, tols)
    alg′ = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(wpeps, H, alg′; check_interval)
    global wpeps = result[1]
end

md"""
## Computing the simple update energy estimate

Finally, we measure the ground-state energy by converging a CTMRG environment and computing
the expectation value, where we make sure to normalize by the unit cell size:
"""

peps = InfinitePEPS(wpeps)
normalize!.(peps.A, Inf) ## normalize PEPS with absorbed weights by largest element
env₀ = CTMRGEnv(rand, Float64, peps, Espace)
env, = leading_boundary(env₀, peps; tol=1e-10, trscheme=trscheme_env)
E = expectation_value(peps, H, env) / (Nr * Nc)

md"""
Let us compare that estimate with benchmark data obtained from the
[YASTN/peps-torch package]((https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat)).
which utilizes AD-based PEPS optimization to find $E_\text{ref}=-0.49425$:
"""

E_ref = -0.49425
@show (E - E_ref) / abs(E_ref)

md"""
## Variational PEPS optimization using AD

As a last step, we will use the SU-evolved PEPS as a starting point for a [`fixedpoint`](@ref)]
PEPS optimization.
"""

peps_opt, env_opt, E_opt, = fixedpoint(H, peps, env; tol=1e-4);

