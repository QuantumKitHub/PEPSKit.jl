```@meta
EditURL = "../../../../examples/heisenberg_su/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/heisenberg_su/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/heisenberg_su/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/heisenberg_su)


# Simple update for the Heisenberg model

In this example, we will use [`SimpleUpdate`](@ref) imaginary time evolution to treat
the two-dimensional Heisenberg model once again:

```math
H = \sum_{\langle i,j \rangle} J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j.
```

In order to simulate the antiferromagnetic order of the Hamiltonian on a single-site unit
cell one typically applies a unitary sublattice rotation. Here, we will instead use a
$2 \times 2$ unit cell and set $J_x = J_y = J_z = 1$.

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

We proceed by initializing a random weighted PEPS that will be evolved. First though, we
need to define the appropriate (symmetric) spaces:

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

wpeps = InfiniteWeightPEPS(rand, Float64, physical_space, bond_space; unitcell=(Nr, Nc));
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
[ Info: SU iter 1      :  dt = 1e-02,  weight diff = 1.683e+00,  time = 10.602 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    :  dt = 1e-02,  weight diff = 3.879e-06,  time = 0.002 sec
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU conv 596    :  dt = 1e-02,  weight diff = 9.933e-07,  time = 12.073 sec
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
[ Info: SU conv 2205   :  dt = 1e-03,  weight diff = 9.978e-09,  time = 4.545 sec
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
[ Info: SU conv 2133   :  dt = 4e-04,  weight diff = 9.999e-09,  time = 4.379 sec

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
[ Info: CTMRG init:	obj = +8.705922473439e-05	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +9.514115680898e-01	err = 6.0310868148e-11	time = 9.35 sec

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

    return map(Iterators.product(axes(peps, 1), axes(peps, 2), S_ops)) do (r, c, S)
        expectation_value(peps, LocalOperator(lattice, (CartesianIndex(r, c),) => S), env)
    end
end

E = expectation_value(peps, H, env) / (Nr * Nc)
Ms = compute_mags(peps, env)
M_norms = map(
    rc -> norm(Ms[rc[1], rc[2], :]), Iterators.product(axes(peps, 1), axes(peps, 2))
)
@show E Ms M_norms;
````

````
E = -0.6674685583160895
Ms = [0.03199644951247372 -0.02980262049564095; -0.029802620502662757 0.03199644954619281;;; 2.2896091139871975e-12 -1.0506898420176606e-12; -2.118251543953617e-12 8.853279136399547e-13;;; 0.37559610906659674 -0.3757765476186203; -0.3757765476169772 0.37559610906659097]
M_norms = [0.37695650933147595 0.3769565093330746; 0.3769565093319919 0.3769565093343323]

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
(E - E_ref) / E_ref = -4.7103646307789086e-5
(mean(M_norms) - M_ref) / E_ref = -0.0003842836445223997

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

