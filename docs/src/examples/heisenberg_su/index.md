```@meta
EditURL = "../../../../examples/heisenberg_su/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//heisenberg_su/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//heisenberg_su/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//heisenberg_su)


# Simple update for the Heisenberg model

In this next example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional Heisenberg model once again:

```math
H = \sum_{\langle i,j \rangle} J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j.
```

In the previous examples, we used a sublattice rotation to simulate antiferromagnetic
Hamiltonian on a single-site unit cell. Here, we will instead use a $2 \times 2$ unit cell
and set $J_x=J_y=J_z=1$.

Let's get started by seeding the RNG and importing all required modules:

````julia
using Random
import Statistics: mean
using TensorKit, PEPSKit
import MPSKitModels: S_x, S_y, S_z, S_exchange
Random.seed!(0);
````

## Defining the Hamiltonian

To construct the Heisenberg Hamiltonian as just discussed, we'll use `heisenberg_XYZ` and,
in addition, make it real (`real` and `imag` works for `LocalOperator`s) since we want to
use PEPS and environments with real entries. We can either initialize the Hamiltonian with
no internal symmetries (`symm = Trivial`) or use the global $U(1)$ symmetry
(`symm = U1Irrep`):

````julia
symm = Trivial ## ∈ {Trivial, U1Irrep}
Nr, Nc = 2, 2
H = real(heisenberg_XYZ(ComplexF64, symm, InfiniteSquare(Nr, Nc); Jx=1, Jy=1, Jz=1));
````

## Simple updating

We proceed by initializing a random weighted PEPS that will be evolved. Again, we'll
normalize its vertex tensors. First though, we need to take of defining the appropriate
(symmetric) spaces:

````julia
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
````

Next, we can start the `SimpleUpdate` routine, successively decreasing the time intervals
and singular value convergence tolerances. Note that TensorKit allows to combine SVD
truncation schemes, which we use here to set a maximal bond dimension and at the same time
fix a truncation error (if that can be reached by remaining below `Dbond`):

````julia
dts = [1e-2, 1e-3, 4e-4]
tols = [1e-6, 1e-8, 1e-8]
maxiter = 10000
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)

for (dt, tol) in zip(dts, tols)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(wpeps, H, alg; bipartite=true)
    global wpeps = result[1]
end
````

````
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.683e+00,  time = 10.318 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    :  dt = 1e-02,  weight diff = 3.879e-06,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU conv 596    :  dt = 1e-02,  weight diff = 9.933e-07,  time = 11.765 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1      :  dt = 1e-03,  weight diff = 2.135e-03,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    :  dt = 1e-03,  weight diff = 9.632e-07,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1000   :  dt = 1e-03,  weight diff = 2.415e-07,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1500   :  dt = 1e-03,  weight diff = 6.291e-08,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 2000   :  dt = 1e-03,  weight diff = 1.683e-08,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU conv 2205   :  dt = 1e-03,  weight diff = 9.978e-09,  time = 4.599 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1      :  dt = 4e-04,  weight diff = 1.418e-04,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    :  dt = 4e-04,  weight diff = 6.377e-08,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1000   :  dt = 4e-04,  weight diff = 3.544e-08,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1500   :  dt = 4e-04,  weight diff = 2.013e-08,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 2000   :  dt = 4e-04,  weight diff = 1.157e-08,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU conv 2133   :  dt = 4e-04,  weight diff = 9.999e-09,  time = 4.428 sec

````

## Computing the ground-state energy and magnetizations

In order to compute observable expectation values, we need to converge a CTMRG environment
on the evolved PEPS. Let's do so:

````julia
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
````

````
[ Info: CTMRG init:	obj = +8.705922473442e-05	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +9.514115680898e-01	err = 3.3105929548e-11	time = 9.37 sec

````

Finally, we'll measure the energy and different magnetizations. For the magnetizations,
the plan is to compute the expectation values unit cell entry-wise in different spin
directions:

````julia
function compute_mags(peps::InfinitePEPS, env::CTMRGEnv)
    lattice = collect(space(t, 1) for t in peps.A)

    # detect symmetry on physical axis
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
````

````
E = -0.6674685583160916
Ms = [[0.03199644951239792 -0.029802620495572755; -0.029802620502593708 0.03199644954611694], [2.289729198376439e-12 -1.051667672083661e-12; -2.117130756284402e-12 8.869747853359352e-13], [0.3755961090665969 -0.3757765476186198; -0.3757765476169762 0.37559610906659047]]
M_norms = [0.3769565093314697 0.37695650933306873; 0.3769565093319854 0.37695650933432534]

````

To assess the results, we will benchmark against data from [Corboz](@cite corboz_variational_2016),
which use manual gradients to perform a variational optimization of the Heisenberg model.
In particular, for the energy and magnetization they find $E_\text{ref} = -0.6675$ and
$M_\text{ref} = 0.3767$. Looking at the relative errors, we find general agreement, although
the accuracy is limited by the methodological limitations of the simple update algorithm as
well as finite bond dimension effects and a lacking extrapolation:

````julia
E_ref = -0.6675
M_ref = 0.3767
@show (E - E_ref) / E_ref
@show (mean(M_norms) - M_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -4.71036463046289e-5
(mean(M_norms) - M_ref) / E_ref = -0.00038428364451283597

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

