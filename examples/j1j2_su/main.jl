using Markdown #hide
md"""
# Three-site simple update for the $J_1$-$J_2$ model

In this example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional $J_1$-$J_2$ model, which contains next-nearest neighbor interaction:

```math
H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j + J_2 \sum_{\langle \langle i,j \rangle \rangle} \mathbf{S}_i \cdot \mathbf{S}_j
```

In this example, we impose the U(1) spin rotation symmetry, 
and calculate the energy when $J_1 = 1$, $J_2 = 1/2$.
"""

using Random
using TensorKit, PEPSKit

Dbond, χenv, symm = 4, 32, U1Irrep
trscheme_env = truncerr(1e-10) & truncdim(χenv)
Nr, Nc, J1 = 2, 2, 1.0

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Pspace = Vect[U1Irrep](1//2 => 1, -1//2 => 1)
Vspace = Vect[U1Irrep](0 => 2, 1//2 => 1, -1//2 => 1)
Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1//2 => χenv ÷ 4, -1//2 => χenv ÷ 4)
Random.seed!(2025)
wpeps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc))

md"""
The value $J_2 / J_1 = 0.5$ is close to a possible spin liquid phase, 
which is challenging for SU to produce a relatively good state from random initialization.
Therefore, we shall gradually increase $J_2 / J_1$ from 0.1 to 0.5. 
"""

dt, tol, maxiter = 1e-2, 1e-8, 30000
check_interval = 4000
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)
alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
for J2 in 0.1:0.1:0.5
    ham = real(
        j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false)
    )
    result = simpleupdate(wpeps, ham, alg; check_interval)
    global wpeps = result[1]
end

md"""
After we reach $J_2 / J_1 = 0.5$, we gradually decrease the evolution time step. 
"""

dts = [1e-3, 1e-4]
tols = [1e-9, 1e-9]
J2 = 0.5
ham = real(j1_j2_model(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false))
for (dt, tol) in zip(dts, tols)
    local alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(wpeps, ham, alg; check_interval)
    global wpeps = result[1]
end

md"""
Finally, we measure the physical quantities with CTMRG, and compare with the benchmark data obtained from [YASTN auto differentiation]((https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1_internal-U1/j20.5/state_1s_A1_U1B_j20.5_D4_chi_opt96.dat)).
"""

peps = InfinitePEPS(wpeps)
normalize!.(peps.A, Inf)
Random.seed!(100)
env = CTMRGEnv(rand, Float64, peps, Espace)
ctm_alg = SequentialCTMRG(; tol=1e-10, verbosity=3, trscheme=trscheme_env)
env, = leading_boundary(env, peps, ctm_alg)

E = expectation_value(peps, ham, env) / (Nr * Nc)
E_ref = -0.49425
@show (E - E_ref) / abs(E_ref)
