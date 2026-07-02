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
import TensorKitTensors.SpinOperators as SO
Random.seed!(0);
````

## Defining the Hamiltonian

To construct the Heisenberg Hamiltonian as just discussed, we'll use `heisenberg_XYZ` and,
in addition, make it real (`real` and `imag` works for `LocalOperator`s) since we want to
use PEPS and environments with real entries. We can either initialize the Hamiltonian with
no internal symmetries (`symm = Trivial`) or use the global spin $U(1)$ symmetry
(`symm = U1Irrep`):

````julia
symm = Trivial ## ∈ {Trivial, U1Irrep}
Nr, Nc = 2, 2
H = real(heisenberg_XYZ(ComplexF64, symm, InfiniteSquare(Nr, Nc); Jx = 1, Jy = 1, Jz = 1));
````

## Simple updating

We proceed by initializing a random PEPS that will be evolved. Since we want to make use of
the bipartite structure of the Heisenberg ground state when we run the simple update routine,
we will make the initial PEPS bipartite explicitly. The weights used for simple update are
initialized as identity matrices. First though, we need to define the appropriate (symmetric) spaces:

````julia
Dbond = 4
χenv = 16
if symm == Trivial
    physical_space = ℂ^2
    bond_space = ℂ^Dbond
    env_space = ℂ^χenv
elseif symm == U1Irrep
    physical_space = ℂ[U1Irrep](1 // 2 => 1, -1 // 2 => 1)
    bond_space = ℂ[U1Irrep](0 => Dbond ÷ 2, 1 // 2 => Dbond ÷ 4, -1 // 2 => Dbond ÷ 4)
    env_space = ℂ[U1Irrep](0 => χenv ÷ 2, 1 // 2 => χenv ÷ 4, -1 // 2 => χenv ÷ 4)
else
    error("not implemented")
end

peps = InfinitePEPS(rand, Float64, physical_space, bond_space; unitcell = (Nr, Nc));
peps[2, 2] = copy(peps[1, 1]) ## make initial random state bipartite
peps[2, 1] = copy(peps[1, 2])
wts = SUWeight(peps);
````

Next, we can start the `SimpleUpdate` routine, successively decreasing the time intervals
and singular value convergence tolerances. Here we set `bipartite = true` to exploit the
underlying bipartite lattice which requires that we input a bipartite PEPS where the diagonal
and off-diagonal unit cell entries are equivalent. Note that TensorKit allows us to combine SVD
truncation schemes, which we use here to set a maximal bond dimension and at the same time fix
a truncation error (if that can be reached by remaining below `Dbond`):

````julia
dts = [1.0e-2, 1.0e-3, 4.0e-4]
tols = [1.0e-6, 1.0e-8, 1.0e-8]
nstep = 10000
trunc_peps = truncerror(; atol = 1.0e-10) & truncrank(Dbond)
alg = SimpleUpdate(; trunc = trunc_peps, bipartite = true)
for (dt, tol) in zip(dts, tols)
    global peps, wts, = time_evolve(peps, H, dt, nstep, alg, wts; tol, check_interval = 500)
end
````

````
[ Info: --- Time evolution (simple update), dt = 0.01 ---
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1      : E ≈ 0.49737, |Δλ| = 1.683e+00. Time = 16.236 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    : E ≈ -0.64898, |Δλ| = 3.879e-06. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 596    : E ≈ -0.65038, |Δλ| = 9.933e-07. Time = 0.002 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 19.31 s
[ Info: --- Time evolution (simple update), dt = 0.001 ---
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1      : E ≈ -0.65215, |Δλ| = 2.135e-03. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    : E ≈ -0.65248, |Δλ| = 9.632e-07. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1000   : E ≈ -0.65261, |Δλ| = 2.415e-07. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1500   : E ≈ -0.65268, |Δλ| = 6.291e-08. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 2000   : E ≈ -0.65271, |Δλ| = 1.683e-08. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 2205   : E ≈ -0.65272, |Δλ| = 9.978e-09. Time = 0.003 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 7.14 s
[ Info: --- Time evolution (simple update), dt = 0.0004 ---
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1      : E ≈ -0.65284, |Δλ| = 1.418e-04. Time = 0.004 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 500    : E ≈ -0.65285, |Δλ| = 6.377e-08. Time = 0.002 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1000   : E ≈ -0.65285, |Δλ| = 3.544e-08. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 1500   : E ≈ -0.65285, |Δλ| = 2.013e-08. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 2000   : E ≈ -0.65286, |Δλ| = 1.157e-08. Time = 0.003 s/it
[ Info: Space of x-weight at [1, 1] = ℂ^4
[ Info: SU iter 2133   : E ≈ -0.65286, |Δλ| = 9.999e-09. Time = 0.003 s/it
[ Info: SU: bond weights have converged.
[ Info: Time evolution finished in 6.59 s

````

## Computing the ground-state energy and magnetizations

In order to compute observable expectation values, we need to converge a CTMRG environment
on the evolved PEPS. Let's do so:

````julia
normalize!.(peps.A, Inf)
env₀ = CTMRGEnv(rand, Float64, peps, env_space)
trunc_env = truncerror(; atol = 1.0e-10) & truncrank(χenv)
env, = leading_boundary(
    env₀,
    peps;
    alg = :SequentialCTMRG,
    projector_alg = :FullInfiniteProjector,
    tol = 1.0e-10,
    trunc = trunc_env,
);
````

````
[ Info: CTMRG init:	obj = +6.853977357765e-16	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +1.300293120452e+00	err = 6.1878669077e-11	time = 8.08 sec

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
        S_ops = real.([SO.S_x(symm), im * SO.S_y(symm), SO.S_z(symm)])
    elseif symm == U1Irrep
        S_ops = real.([SO.S_z(symm)]) ## only Sz preserves <Sz>
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
E = -0.6674685583160874
Ms = [0.03199644968307137 -0.02980262049211579; -0.029802620725989104 0.03199644959546846;;; 1.573600069693093e-10 -2.3817596506159333e-10; 1.3793997194477825e-11 6.715037553783887e-11;;; 0.375596109051991 -0.3757765476189344; -0.3757765475994436 0.37559610906239854]
M_norms = [0.3769565093314034 0.3769565093331091; 0.3769565093321696 0.3769565093343376]

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
@show (E - E_ref) / abs(E_ref)
@show (mean(M_norms) - M_ref) / M_ref;
````

````
(E - E_ref) / abs(E_ref) = 4.710364631094927e-5
(mean(M_norms) - M_ref) / M_ref = 0.0006809379685557489

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

