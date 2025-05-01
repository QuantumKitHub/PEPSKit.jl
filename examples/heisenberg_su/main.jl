using Markdown #hide
md"""
# Simple update for the Heisenberg model

In this next example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional Heisenberg model once again:

```math
H = \sum_{\langle i,j \rangle} J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j.
```

In the previous examples, we used a sublattice rotation to simulate antiferromagnetic
Hamiltonian on a single-site unit cell. Here, we will instead use a $2 \times 2$ unit cell
and set $J_x = J_y = J_z = 1$.

Let's get started by seeding the RNG and importing all required modules:
"""

using Random
import Statistics: mean
using TensorKit, PEPSKit
import MPSKitModels: S_x, S_y, S_z, S_exchange
Random.seed!(0);

md"""
## Defining the Hamiltonian

To construct the Heisenberg Hamiltonian as just discussed, we'll use `heisenberg_XYZ` and,
in addition, make it real (`real` and `imag` works for `LocalOperator`s) since we want to
use PEPS and environments with real entries. We can either initialize the Hamiltonian with
no internal symmetries (`symm = Trivial`) or use the global $U(1)$ symmetry
(`symm = U1Irrep`):
"""

symm = Trivial ## ∈ {Trivial, U1Irrep}
Nr, Nc = 2, 2
H = real(heisenberg_XYZ(ComplexF64, symm, InfiniteSquare(Nr, Nc); Jx=1, Jy=1, Jz=1));

md"""
## Simple updating

We proceed by initializing a random weighted PEPS that will be evolved. Again, we'll
normalize its vertex tensors. First though, we need to define the appropriate
(symmetric) spaces:
"""

Dbond = 4
χenv = 16
if symm == Trivial
    physical_space = ℂ^2
    bond_space = ℂ^Dbond
    env_space = ℂ^χenv
elseif symm == U1Irrep
    physical_space = ℂ[U1Irrep](1//2 => 1, -1//2 => 1)
    bond_space = ℂ[U1Irrep](0 => Dbond ÷ 2, 1//2 => Dbond ÷ 4, -1//2 => Dbond ÷ 4)
    env_space = ℂ[U1Irrep](0 => χenv ÷ 2, 1//2 => χenv ÷ 4, -1//2 => χenv ÷ 4)
else
    error("not implemented")
end

wpeps = InfiniteWeightPEPS(rand, Float64, physical_space, bond_space; unitcell=(Nr, Nc))
for ind in CartesianIndices(wpeps.vertices)
    wpeps.vertices[ind] /= norm(wpeps.vertices[ind], Inf)
end

md"""
Next, we can start the `SimpleUpdate` routine, successively decreasing the time intervals
and singular value convergence tolerances. Note that TensorKit allows to combine SVD
truncation schemes, which we use here to set a maximal bond dimension and at the same time
fix a truncation error (if that can be reached by remaining below `Dbond`):
"""

dts = [1e-2, 1e-3, 4e-4]
tols = [1e-6, 1e-8, 1e-8]
maxiter = 10000
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)

for (dt, tol) in zip(dts, tols)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(wpeps, H, alg; bipartite=true)
    global wpeps = result[1]
end

md"""
## Computing the ground-state energy and magnetizations

In order to compute observable expectation values, we need to converge a CTMRG environment
on the evolved PEPS. Let's do so:
"""

peps = InfinitePEPS(wpeps) ## absorb the weights
env₀ = CTMRGEnv(rand, Float64, peps, env_space)
trscheme_env = truncerr(1e-10) & truncdim(χenv)
env, = leading_boundary(
    env₀,
    peps;
    alg=:sequential,
    projector_alg=:fullinfinite,
    tol=1e-10,
    trscheme=trscheme_env,
);

md"""
Finally, we'll measure the energy and different magnetizations. For the magnetizations,
the plan is to compute the expectation values unit cell entry-wise in different spin
directions:
"""

function compute_mags(peps::InfinitePEPS, env::CTMRGEnv)
    lattice = collect(space(t, 1) for t in peps.A)

    ## detect symmetry on physical axis
    symm = sectortype(space(peps.A[1, 1]))
    if symm == Trivial
        S_ops = real.([S_x(symm), im * S_y(symm), S_z(symm)])
    elseif symm == U1Irrep
        S_ops = real.([S_z(symm)]) ## only Sz preserves <Sz>
    end

    return [
        collect(
            expectation_value(
                peps, LocalOperator(lattice, (CartesianIndex(r, c),) => S), env
            ) for (r, c) in Iterators.product(1:size(peps, 1), 1:size(peps, 2))
        ) for S in S_ops
    ]
end

E = expectation_value(peps, H, env) / (Nr * Nc)
Ms = compute_mags(peps, env)
M_norms = collect(norm([m[r, c] for m in Ms]) for (r, c) in Iterators.product(1:Nr, 1:Nc))
@show E Ms M_norms;

md"""
To assess the results, we will benchmark against data from [Corboz](@cite corboz_variational_2016),
which use manual gradients to perform a variational optimization of the Heisenberg model.
In particular, for the energy and magnetization they find $E_\text{ref} = -0.6675$ and
$M_\text{ref} = 0.3767$. Looking at the relative errors, we find general agreement, although
the accuracy is limited by the methodological limitations of the simple update algorithm as
well as finite bond dimension effects and a lacking extrapolation:
"""

E_ref = -0.6675
M_ref = 0.3767
@show (E - E_ref) / E_ref
@show (mean(M_norms) - M_ref) / E_ref;
