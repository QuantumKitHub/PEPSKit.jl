```@meta
EditURL = "../../../../examples/3d_ising_partition_function/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/3d_ising_partition_function/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/3d_ising_partition_function/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/3d_ising_partition_function)


# [The 3D classical Ising model](@id e_3d_ising)

In this example, we will showcase how one can use PEPSKit to study 3D classical statistical
mechanics models. In particular, we will consider a specific case of the 3D classical Ising
model, but the same techniques can be applied to other 3D classical models as well.

As compared to simulations of [2D partition functions](@ref e_2d_ising), the workflow
presented in this example is a bit more experimental and less 'black-box'. Therefore, it
also serves as a demonstration of some of the more internal functionality of PEPSKit,
and how one can adapt it to less 'standard' kinds of problems.

Let us consider the partition function of the classical Ising model,

```math
\mathcal{Z}(\beta) = \sum_{\{s\}} \exp(-\beta H(s)) \text{ with } H(s) = -J \sum_{\langle i, j \rangle} s_i s_j .
```

where the classical spins $s_i \in \{+1, -1\}$ are located on the vertices $i$ of a 3D
cubic lattice. The partition function of this model can be represented as a 3D tensor
network with a rank-6 tensor at each vertex of the lattice. Such a network can be contracted
by finding the fixed point of the corresponding transfer operator, in exactly the same
spirit as the [boundary MPS methods](@ref e_boundary_mps) demonstrated in another example.

Let's start by making the example deterministic and importing the required packages:

````julia
using Random
using LinearAlgebra
using PEPSKit, TensorKit
using KrylovKit, OptimKit, Zygote

Random.seed!(81812781144);
````

## Defining the partition function

Just as in the 2D case, the first step is to define the partition function as a tensor
network. The procedure is exactly the same as before, the only difference being that now
every spin participates in interactions associated to six links adjacent to that site. This
means that the partition function can be written as an infinite 3D network with a single
constituent rank-6 [`PEPSKit.PEPOTensor`](@ref) `O` located at each site of the cubic
lattice. To verify our example we will check the magnetization and energy, so we also define
the corresponding rank-6 tensors `M` and `E` while we're at it.

````julia
function three_dimensional_classical_ising(; beta, J=1.0)
    K = beta * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    # magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4 -5 -6] :=
        M[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* q
    @tensor e_x[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * e[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_y[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * e[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_z[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * e[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    e = e_x + e_y + e_z

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end;
````

Let's initialize these tensors at inverse temperature ``\beta=0.2391``, which corresponds to
a slightly lower temperature than the critical value ``\beta_c=0.2216544…``

````julia
beta = 0.2391
O, M, E = three_dimensional_classical_ising(; beta)
O isa PEPSKit.PEPOTensor
````

````
true
````

## Contracting the partition function

To contract our infinite 3D partition function, we first reinterpret it as an infinite power
of a slice-to-slice transfer operator ``T``, where ``T`` can be seen as an infinite 2D
projected entangled-pair operator (PEPO) which consists of the rank-6 tensor `O` at each
site of an infinite 2D square lattice. In the same spirit as the boundary MPS approach, all
we need to contract the whole partition function is to find the leading eigenvector of this
PEPO. The fixed point of such a PEPO can be parametrized as a PEPS, and for the case of a
Hermitian transfer operator we can find this PEPS through [variational optimization](@cite
vanderstraeten_residual_2018).

Indeed, for a Hermitian transfer operator ``T`` we can characterize the fixed point PEPS
``|\psi\rangle`` which satisfies the eigenvalue equation
``T |\psi\rangle = \Lambda |\psi\rangle`` corresponding to the largest magnitude eigenvalue
``\Lambda`` as the solution of a variational problem

```math
|\psi\rangle = \text{argmin}_{|\psi\rangle} \left ( \lim_{N \to ∞} - \frac{1}{N} \log \left( \frac{\langle \psi | T | \psi \rangle}{\langle \psi | \psi \rangle} \right) \right ) ,
```

where ``N`` is the diverging number of sites of the 2D transfer operator ``T``. The function
minimized in this expression is exactly the free energy per site of the partition function,
so we essentially find the fixed-point PEPS by variationally minimizing the free energy.

### Defining the cost function

Using PEPSKit.jl, this cost function and its gradient can be computed, after which we can
use [OptimKit.jl](https://github.com/Jutho/OptimKit.jl) to actually optimize it. We can
immediately recognize the denominator ``\langle \psi | \psi \rangle`` as the familiar PEPS
norm, where we can compute the norm per site as the [`network_value`](@ref) of the
corresponding [`InfiniteSquareNetwork`](@ref) by contracting it with the CTMRG algorithm.
Similarly, the numerator ``\langle \psi | T | \psi \rangle`` is nothing more than an
`InfiniteSquareNetwork` consisting of three layers corresponding to the ket, transfer
operator and bra objects. This object can also be constructed and contracted in a
straightforward way, so we can again compute its `network_value`.

To define our cost function, we then need to construct the transfer operator as an
[`InfinitePEPO`](@ref), construct the two infinite 2D contractible networks for the
numerator and denominator from the current PEPS and this transfer operator, and specify a
contraction algorithm we can use to compute the values of these two networks. In addition,
we'll specify the specific reverse rule algorithm that will be used to compute the gradient
of this cost function.

````julia
boundary_alg = SimultaneousCTMRG(; maxiter=150, tol=1e-8, verbosity=1)
rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
)
T = InfinitePEPO(O)

function pepo_costfun((peps, env_double_layer, env_triple_layer))
    # use Zygote to compute the gradient automatically
    E, gs = withgradient(peps) do ψ
        # construct the PEPS norm network
        n_double_layer = InfiniteSquareNetwork(ψ)
        # contract this network
        env_double_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_double_layer,
            n_double_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        # construct the PEPS-PEPO-PEPS overlap network
        n_triple_layer = InfiniteSquareNetwork(ψ, T)
        # contract this network
        env_triple_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_triple_layer,
            n_triple_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        # update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_double_layer, env_double_layer′)
            PEPSKit.update!(env_triple_layer, env_triple_layer′)
        end
        # compute the network values per site
        λ3 = network_value(n_triple_layer, env_triple_layer)
        λ2 = network_value(n_double_layer, env_double_layer)
        # use this to compute the actual cost function
        return -log(real(λ3 / λ2))
    end
    g = only(gs)
    return E, g
end;
````

There are a few things to note about this cost function definition. Since we will pass it to
the `OptimKit.optimize`, we require it to return both our cost function and the
corresponding gradient. To do this, we simply use the `withgradient` method from Zygote.jl
to automatically compute the gradient of the cost function straight from the primal
computation. Since our cost function involves contractions using `leading_boundary`, we also
have to specify exactly how Zygote should handle the backpropagation of the gradient through
this function. This can be done using the [`PEPSKit.hook_pullback`](@ref) function from
PEPSKit.jl, which allows to hook into the pullback of a given function by specifying a
specific algorithm for the pullback computation. Here, we opted to use an Arnoldi method to
solve the linear problem defining the gradient of the network contraction at its fixed
point. This is exactly the workflow that internally underlies [`PEPSKit.fixedpoint`](@ref), and
more info on particular gradient algorithms can be found in the corresponding docstrings.

### Characterizing the optimization manifold

In order to make the best use of OptimKit.jl, we should specify some properties of the
manifold on which we are optimizing. Looking at our cost function defined above, a point on
our optimization manifold corresponds to a `Tuple` of three objects. The first is an
`InfinitePEPS` encoding the fixed point we are actually optimizing, while the second and
third are `CTMRGEnv` objects corresponding to the environments of the double and triple
layer networks ``\langle \psi | \psi \rangle`` and ``\langle \psi | T | \psi \rangle``
respectively. While the environments are just there so we can reuse them between subsequent
contractions and we don't need to think about them much, optimizing over the manifold of
`InfinitePEPS` requires a bit more care.

In particular, we need to define two kinds of operations on this manifold: a retraction and
a transport. The retraction, corresponding to the `retract` keyword argument of
`OptimKit.optimize`, specifies how to move from a point on a manifold along a given descent
direction to obtain a new manifold point. The transport, corresponding to the `transport!`
keyword argument of `OptimKit.optimize`, specifies how to transport a descent direction at a
given manifold point to a valid descent direction at a different manifold point according to
the appropriate metric. For a more detailed explanation we refer to the
[OptimKit.jl README](https://github.com/Jutho/OptimKit.jl). In PEPSKit.jl, these two
procedures are defined through the [`PEPSKit.peps_retract`](@ref) and
[`PEPSKit.peps_transport!`](@ref) methods. While it is instructive to read the corresponding
docstrings in order to understand what these actually do, here we can just blindly reuse
them where the only difference is that we have to pass along an extra environment since our
cost function requires two distinct contractions as opposed to the setting of Hamiltonian
PEPS optimization which only requires a double-layer contraction.

````julia
function pepo_retract((peps, env_double_layer, env_triple_layer), η, α)
    (peps´, env_double_layer´), ξ = PEPSKit.peps_retract((peps, env_double_layer), η, α)
    env_triple_layer´ = deepcopy(env_triple_layer)
    return (peps´, env_double_layer´, env_triple_layer´), ξ
end
function pepo_transport!(
    ξ,
    (peps, env_double_layer, env_triple_layer),
    η,
    α,
    (peps´, env_double_layer´, env_triple_layer´),
)
    return PEPSKit.peps_transport!(
        ξ, (peps, env_double_layer), η, α, (peps´, env_double_layer´)
    )
end;
````

### Finding the fixed point

All that is left then is to specify the virtual spaces of the PEPS and the two environments,
initialize them in the appropriate way, choose an optimization algortithm and call the
`optimize` function from OptimKit.jl to get our desired PEPS fixed point.

````julia
Vpeps = ℂ^2
Venv = ℂ^12

psi0 = initializePEPS(T, Vpeps)
env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), Venv)
env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, T), Venv)

optimizer_alg = LBFGS(32; maxiter=100, gradtol=1e-5, verbosity=3)

(psi_final, env2_final, env3_final), f, = optimize(
    pepo_costfun,
    (psi0, env2_0, env3_0),
    optimizer_alg;
    inner=PEPSKit.real_inner,
    retract=pepo_retract,
    (transport!)=(pepo_transport!),
);
````

````
[ Info: LBFGS: initializing with f = -0.554073395182, ‖∇f‖ = 7.7844e-01
┌ Warning: CTMRG cancel 150:	obj = +1.702942228759e+01 +1.443123032606e-07im	err = 2.4386740957e-05	time = 1.12 sec
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/ctmrg/ctmrg.jl:155
[ Info: LBFGS: iter    1, time  117.39 s: f = -0.777080930369, ‖∇f‖ = 3.1305e-02, α = 7.10e+02, m = 0, nfg = 7
[ Info: LBFGS: iter    2, time  118.47 s: f = -0.784111515961, ‖∇f‖ = 2.0103e-02, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  118.64 s: f = -0.792705733484, ‖∇f‖ = 2.3327e-02, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  118.78 s: f = -0.796289732476, ‖∇f‖ = 2.2475e-02, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  118.89 s: f = -0.799674902374, ‖∇f‖ = 7.0288e-03, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  118.98 s: f = -0.800082100121, ‖∇f‖ = 1.2717e-03, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  119.08 s: f = -0.800110603125, ‖∇f‖ = 1.3384e-03, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  119.18 s: f = -0.800262201996, ‖∇f‖ = 2.4945e-03, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  119.27 s: f = -0.800450505448, ‖∇f‖ = 2.9259e-03, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  119.36 s: f = -0.800764917087, ‖∇f‖ = 1.7221e-03, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  119.45 s: f = -0.800876048838, ‖∇f‖ = 2.2475e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  119.53 s: f = -0.801100867467, ‖∇f‖ = 1.5561e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  119.63 s: f = -0.801317048856, ‖∇f‖ = 1.1561e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  119.73 s: f = -0.801373050545, ‖∇f‖ = 7.1300e-04, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  119.82 s: f = -0.801388615264, ‖∇f‖ = 2.8462e-04, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  119.91 s: f = -0.801394633333, ‖∇f‖ = 2.7607e-04, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  119.99 s: f = -0.801408061564, ‖∇f‖ = 3.6096e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  120.09 s: f = -0.801509542169, ‖∇f‖ = 1.9822e-03, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  120.20 s: f = -0.801578405251, ‖∇f‖ = 1.8040e-03, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  120.59 s: f = -0.801694524424, ‖∇f‖ = 2.9356e-03, α = 5.48e-01, m = 19, nfg = 3
[ Info: LBFGS: iter   21, time  121.07 s: f = -0.801761920683, ‖∇f‖ = 1.1993e-03, α = 3.82e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   22, time  121.19 s: f = -0.801797785494, ‖∇f‖ = 6.0337e-04, α = 1.00e+00, m = 21, nfg = 1
[ Info: LBFGS: iter   23, time  121.45 s: f = -0.801808747834, ‖∇f‖ = 3.7053e-04, α = 5.24e-01, m = 22, nfg = 2
[ Info: LBFGS: iter   24, time  121.58 s: f = -0.801812729173, ‖∇f‖ = 3.0781e-04, α = 1.00e+00, m = 23, nfg = 1
[ Info: LBFGS: iter   25, time  121.71 s: f = -0.801816445211, ‖∇f‖ = 2.9994e-04, α = 1.00e+00, m = 24, nfg = 1
[ Info: LBFGS: iter   26, time  121.84 s: f = -0.801824713130, ‖∇f‖ = 3.6496e-04, α = 1.00e+00, m = 25, nfg = 1
[ Info: LBFGS: iter   27, time  121.98 s: f = -0.801839673823, ‖∇f‖ = 5.4222e-04, α = 1.00e+00, m = 26, nfg = 1
[ Info: LBFGS: iter   28, time  122.12 s: f = -0.801857478904, ‖∇f‖ = 2.7917e-04, α = 1.00e+00, m = 27, nfg = 1
[ Info: LBFGS: iter   29, time  122.29 s: f = -0.801864555224, ‖∇f‖ = 1.2319e-04, α = 1.00e+00, m = 28, nfg = 1
[ Info: LBFGS: iter   30, time  122.48 s: f = -0.801865598736, ‖∇f‖ = 8.6048e-05, α = 1.00e+00, m = 29, nfg = 1
[ Info: LBFGS: iter   31, time  122.63 s: f = -0.801867571755, ‖∇f‖ = 8.8636e-05, α = 1.00e+00, m = 30, nfg = 1
[ Info: LBFGS: iter   32, time  122.78 s: f = -0.801870393528, ‖∇f‖ = 2.6554e-04, α = 1.00e+00, m = 31, nfg = 1
[ Info: LBFGS: iter   33, time  122.93 s: f = -0.801874797039, ‖∇f‖ = 2.7841e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   34, time  123.13 s: f = -0.801877566644, ‖∇f‖ = 1.8523e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   35, time  123.33 s: f = -0.801878506245, ‖∇f‖ = 2.0638e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   36, time  123.51 s: f = -0.801878995097, ‖∇f‖ = 5.6081e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   37, time  123.67 s: f = -0.801879153573, ‖∇f‖ = 6.2356e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   38, time  123.82 s: f = -0.801879355075, ‖∇f‖ = 6.0528e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   39, time  123.97 s: f = -0.801880115100, ‖∇f‖ = 6.2768e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   40, time  124.12 s: f = -0.801881475065, ‖∇f‖ = 6.2301e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   41, time  124.29 s: f = -0.801882272425, ‖∇f‖ = 9.5267e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   42, time  124.44 s: f = -0.801882600033, ‖∇f‖ = 5.1283e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   43, time  124.59 s: f = -0.801882711875, ‖∇f‖ = 2.6091e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   44, time  124.73 s: f = -0.801882805828, ‖∇f‖ = 2.9316e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   45, time  124.85 s: f = -0.801883027060, ‖∇f‖ = 2.7982e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   46, time  124.98 s: f = -0.801883402178, ‖∇f‖ = 3.8102e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   47, time  125.13 s: f = -0.801883718321, ‖∇f‖ = 5.3658e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   48, time  125.26 s: f = -0.801883962887, ‖∇f‖ = 2.8728e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   49, time  125.40 s: f = -0.801884158085, ‖∇f‖ = 3.0680e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   50, time  125.55 s: f = -0.801884385940, ‖∇f‖ = 4.1973e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   51, time  125.70 s: f = -0.801884810459, ‖∇f‖ = 6.8881e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   52, time  125.85 s: f = -0.801885011014, ‖∇f‖ = 3.8651e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   53, time  126.00 s: f = -0.801885126625, ‖∇f‖ = 1.9013e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   54, time  126.14 s: f = -0.801885186489, ‖∇f‖ = 3.2919e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   55, time  126.29 s: f = -0.801885309713, ‖∇f‖ = 4.8521e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   56, time  126.45 s: f = -0.801885491631, ‖∇f‖ = 1.1478e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   57, time  126.61 s: f = -0.801885912857, ‖∇f‖ = 7.7221e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   58, time  126.77 s: f = -0.801886451980, ‖∇f‖ = 6.5316e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   59, time  126.91 s: f = -0.801886639804, ‖∇f‖ = 5.1567e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   60, time  127.25 s: f = -0.801886699372, ‖∇f‖ = 4.5540e-05, α = 3.68e-01, m = 32, nfg = 2
[ Info: LBFGS: iter   61, time  127.43 s: f = -0.801886723992, ‖∇f‖ = 2.1992e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   62, time  127.61 s: f = -0.801886735202, ‖∇f‖ = 1.8064e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   63, time  127.80 s: f = -0.801886771395, ‖∇f‖ = 3.8651e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   64, time  127.97 s: f = -0.801886801952, ‖∇f‖ = 4.2630e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   65, time  128.14 s: f = -0.801886837856, ‖∇f‖ = 3.9318e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   66, time  128.30 s: f = -0.801886916784, ‖∇f‖ = 3.8747e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   67, time  128.45 s: f = -0.801887030055, ‖∇f‖ = 3.7139e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   68, time  128.62 s: f = -0.801887141198, ‖∇f‖ = 5.7017e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   69, time  128.95 s: f = -0.801887199205, ‖∇f‖ = 3.0700e-05, α = 5.24e-01, m = 32, nfg = 2
[ Info: LBFGS: iter   70, time  129.10 s: f = -0.801887246613, ‖∇f‖ = 1.3885e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   71, time  129.25 s: f = -0.801887263716, ‖∇f‖ = 1.5769e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   72, time  129.40 s: f = -0.801887319464, ‖∇f‖ = 2.1424e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   73, time  129.56 s: f = -0.801887406143, ‖∇f‖ = 1.9896e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   74, time  129.88 s: f = -0.801887467460, ‖∇f‖ = 1.9800e-05, α = 3.61e-01, m = 32, nfg = 2
[ Info: LBFGS: converged after 75 iterations and time 130.03 s: f = -0.801887535670, ‖∇f‖ = 9.9339e-06

````

### Verifying the result

Having found the fixed point, we have essentially contracted the entire partition function
and we can start computing observables. The free energy per site for example is just given by
the final value of the cost function we have just optimized.

````julia
@show f
````

````
-0.8018875356699146
````

As another check, we can compute the magnetization per site and compare it to a [reference
value obtaind through Monte-Carlo simulations](@cite hasenbusch_monte_2001).

````julia
n3_final = InfiniteSquareNetwork(psi_final, T)
num = PEPSKit.contract_local_tensor((1, 1, 1), M, n3_final, env3_final)
denom = PEPSKit._contract_site((1, 1), n3_final, env3_final)
m = abs(num / denom)

m_ref = 0.667162

@show abs(m - m_ref)
````

````
0.00011315233182807027
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

