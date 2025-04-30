```@meta
EditURL = "../../../../examples/3d_ising_partition_function/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//3d_ising_partition_function/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//3d_ising_partition_function/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//3d_ising_partition_function)


# [The 3D classical Ising model](@id e_3d_ising)

In a previous example we have already demonstrated an application of PEPSKit.jl to the study
of 2D classical statistical mechanics models. In this example, we will take this one step
further, and showcase how one can use PEPSKit.jl to study 3D classical statistical mechanics
models. We will demonstrate this for the specific case of the 3D classical Ising model, but
the same techniques can be applied to other 3D classical models as well.

The workflow showcased in this example is a bit more experimental and less 'black-box' than
previous examples. Therefore it also serves as a demonstration of some of the more internal
functionality of PEPSKit.jl, and how one can adapt it to less 'standard' kinds of problems.

Let us consider again the partition function of the classical Ising model,

```math
\mathcal{Z}(\beta) = \sum_{\{s\}} \exp(-\beta H(s)) \text{ with } H(s) = -J \sum_{\langle i, j \rangle} s_i s_j .
```

where now the classical spins $s_i \in \{+1, -1\}$ are located on the vertices $i$ of a 3D
cubic lattice. The partition function of this model can be represented as a 3D tensor
network with a rank-6 tensor at each vertex of the lattice. Such a network can be contracted
by finding the fixed point of the corresponding transfer operator, in exactly the same
spirit as the [boundary MPS methods](@ref e_boundary_mps) demonstrated in another example.

Let's start by making the example deterministic and we doing our imports:

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
PEPO. The fixed point of such a PEPO exactly corresponds to a PEPS, and for the case of a
Hermitian transfer operator we can find this PEPS through [variational optimization](@cite
vanderstraeten_residual_2018).

Indeed, for a Hermition transfer operator ``T`` we can formulate the eigenvalue equation as
for a fixed point PEPS ``|\psi\rangle`` as a variational problem

```math
|\psi\rangle = \text{argmin}_{|\psi\rangle} \left ( \lim_{N \to ∞} - \frac{1}{N} \log \left( \frac{\langle \psi | T | \psi \rangle}{\langle \psi | \psi \rangle} \right) \right )
```
where ``N`` is the diverging number of sites of the 2D transfer operator ``T``.

### Defining the cost function

Using PEPSKit.jl, this cost function and its gradient can be easily computed, after which we
can use [OptimKit.jl](https://github.com/Jutho/OptimKit.jl) to actually optimize it. We can
immediately recognize the denominator ``\langle \psi | \psi \rangle`` as the familiar PEPS
norm, where we can compute the norm per site as the [`network_value`](@ref) of the
corresponding [`InfiniteSquareNetwork`](@ref) by contracting it with the CTMRG algorithm.
Similarly, the numerator ``\langle \psi | T | \psi \rangle`` is nothing more than an
`InfiniteSquareNetwork` consisting of three layers corresponding to the ket, transfer
operator and bra objects. This object can also be constructed and contracted in a
straightforward way, after we can again compute its `network_value`.

So to define our cost function, we just need to construct the transfer operator as an
[`InfinitePEPO`](@ref), contruct the both relevant infinite 2D contractible networks from
the current PEPS and this transfer operator, and specify a contraction algorithm we can use
to compute the values of these two networks. In addition, we'll specify the specific reverse
rule algorithm that will be used to compute the gradient of this cost function.

````julia
boundary_alg = SimultaneousCTMRG(; maxiter=150, tol=1e-8, verbosity=1)
rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
)
T = InfinitePEPO(O)

function pepo_costfun((psi, env2, env3))
    # use Zygote to compute the gradient automatically
    E, gs = withgradient(psi) do ψ
        # construct the PEPS norm network
        n2 = InfiniteSquareNetwork(ψ)
        # contract this network
        env2′, info = PEPSKit.hook_pullback(
            leading_boundary, env2, n2, boundary_alg; alg_rrule=rrule_alg
        )
        # construct the PEPS-PEPO-PEPS overlap network
        n3 = InfiniteSquareNetwork(ψ, T)
        # contract this network
        env3′, info = PEPSKit.hook_pullback(
            leading_boundary, env3, n3, boundary_alg; alg_rrule=rrule_alg
        )
        # update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
        end
        # compute the network values per site
        λ3 = network_value(n3, env3)
        λ2 = network_value(n2, env2)
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
point. This is exactly the workflow underlying [`PEPSKit.fixedpoint`](@ref) internally, and
more info on particular gradient algorithms can be found in the corresponding docstring.

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
the appropriate metric. In PEPSKit.jl, these two procedures are defined through the
[`PEPSKit.peps_retract`](@ref) and [`PEPSKit.peps_transport!`](@ref) methods. While it is
instructive to read the corresponding docstrings in order to understand what these actually
do, here we can just blindly reuse them where the only difference is that we have to pass
along an extra environment since our cost function requires two distinct contractions as
opposed to the setting of Hamiltonian PEPS optimization.

````julia
function pepo_retract(x, η, α)
    x´_partial, ξ = PEPSKit.peps_retract(x[1:2], η, α)
    x´ = (x´_partial..., deepcopy(x[3]))
    return x´, ξ
end
function pepo_transport!(ξ, x, η, α, x´)
    return PEPSKit.peps_transport!(ξ, x[1:2], η, α, x´[1:2])
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
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
┌ Warning: CTMRG cancel 150:	obj = +1.702942227203e+01 +1.438609955721e-07im	err = 2.4390784904e-05	time = 9.24 sec
└ @ PEPSKit ~/git/PEPSKit.jl/src/algorithms/ctmrg/ctmrg.jl:155
[ Info: LBFGS: iter    1, time  144.38 s: f = -0.777080930424, ‖∇f‖ = 3.1305e-02, α = 7.10e+02, m = 0, nfg = 7
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time  147.50 s: f = -0.784111515920, ‖∇f‖ = 2.0103e-02, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    3, time  148.61 s: f = -0.792705733264, ‖∇f‖ = 2.3327e-02, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  149.51 s: f = -0.796289732161, ‖∇f‖ = 2.2475e-02, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, time  150.24 s: f = -0.799674902231, ‖∇f‖ = 7.0288e-03, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, time  150.90 s: f = -0.800082100115, ‖∇f‖ = 1.2717e-03, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, time  151.69 s: f = -0.800110603120, ‖∇f‖ = 1.3384e-03, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  152.35 s: f = -0.800262202003, ‖∇f‖ = 2.4945e-03, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  153.02 s: f = -0.800450505439, ‖∇f‖ = 2.9259e-03, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  154.51 s: f = -0.800764917063, ‖∇f‖ = 1.7221e-03, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  155.29 s: f = -0.800876048822, ‖∇f‖ = 2.2475e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  155.96 s: f = -0.801100867388, ‖∇f‖ = 1.5561e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  156.75 s: f = -0.801317048785, ‖∇f‖ = 1.1561e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  157.56 s: f = -0.801373050522, ‖∇f‖ = 7.1300e-04, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  158.53 s: f = -0.801388615258, ‖∇f‖ = 2.8462e-04, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  159.46 s: f = -0.801394633326, ‖∇f‖ = 2.7607e-04, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  160.33 s: f = -0.801408061547, ‖∇f‖ = 3.6096e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  161.44 s: f = -0.801509542237, ‖∇f‖ = 1.9822e-03, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time  162.56 s: f = -0.801578405298, ‖∇f‖ = 1.8040e-03, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time  166.52 s: f = -0.801694524832, ‖∇f‖ = 2.9356e-03, α = 5.48e-01, m = 19, nfg = 3
[ Info: LBFGS: iter   21, time  168.81 s: f = -0.801761920585, ‖∇f‖ = 1.1993e-03, α = 3.82e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   22, time  169.89 s: f = -0.801797785496, ‖∇f‖ = 6.0337e-04, α = 1.00e+00, m = 21, nfg = 1
[ Info: LBFGS: iter   23, time  172.11 s: f = -0.801808747834, ‖∇f‖ = 3.7053e-04, α = 5.24e-01, m = 22, nfg = 2
[ Info: LBFGS: iter   24, time  173.16 s: f = -0.801812729196, ‖∇f‖ = 3.0781e-04, α = 1.00e+00, m = 23, nfg = 1
[ Info: LBFGS: iter   25, time  174.21 s: f = -0.801816445181, ‖∇f‖ = 2.9994e-04, α = 1.00e+00, m = 24, nfg = 1
[ Info: LBFGS: iter   26, time  175.24 s: f = -0.801824712580, ‖∇f‖ = 3.6497e-04, α = 1.00e+00, m = 25, nfg = 1
[ Info: LBFGS: iter   27, time  176.42 s: f = -0.801839673918, ‖∇f‖ = 5.4217e-04, α = 1.00e+00, m = 26, nfg = 1
[ Info: LBFGS: iter   28, time  177.59 s: f = -0.801857480042, ‖∇f‖ = 2.7918e-04, α = 1.00e+00, m = 27, nfg = 1
[ Info: LBFGS: iter   29, time  178.76 s: f = -0.801864556175, ‖∇f‖ = 1.2323e-04, α = 1.00e+00, m = 28, nfg = 1
[ Info: LBFGS: iter   30, time  179.86 s: f = -0.801865599394, ‖∇f‖ = 8.6049e-05, α = 1.00e+00, m = 29, nfg = 1
[ Info: LBFGS: iter   31, time  181.16 s: f = -0.801867571015, ‖∇f‖ = 8.8673e-05, α = 1.00e+00, m = 30, nfg = 1
[ Info: LBFGS: iter   32, time  182.36 s: f = -0.801870391456, ‖∇f‖ = 2.6510e-04, α = 1.00e+00, m = 31, nfg = 1
[ Info: LBFGS: iter   33, time  183.60 s: f = -0.801874799763, ‖∇f‖ = 2.7836e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   34, time  184.85 s: f = -0.801877566434, ‖∇f‖ = 1.8512e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   35, time  186.03 s: f = -0.801878506140, ‖∇f‖ = 2.0603e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   36, time  187.13 s: f = -0.801878994665, ‖∇f‖ = 5.6086e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   37, time  188.25 s: f = -0.801879153394, ‖∇f‖ = 6.2420e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   38, time  189.35 s: f = -0.801879354902, ‖∇f‖ = 6.0532e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   39, time  190.46 s: f = -0.801880114487, ‖∇f‖ = 6.2680e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   40, time  191.69 s: f = -0.801881471358, ‖∇f‖ = 6.2392e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   41, time  192.98 s: f = -0.801882270540, ‖∇f‖ = 9.4942e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   42, time  194.10 s: f = -0.801882598674, ‖∇f‖ = 5.1910e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   43, time  195.22 s: f = -0.801882711254, ‖∇f‖ = 2.6275e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   44, time  196.27 s: f = -0.801882805203, ‖∇f‖ = 2.9425e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   45, time  197.29 s: f = -0.801883026107, ‖∇f‖ = 2.7910e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   46, time  198.44 s: f = -0.801883400172, ‖∇f‖ = 3.7426e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   47, time  199.61 s: f = -0.801883717581, ‖∇f‖ = 5.4717e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   48, time  200.77 s: f = -0.801883966703, ‖∇f‖ = 2.9045e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   49, time  201.95 s: f = -0.801884163647, ‖∇f‖ = 3.0661e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   50, time  203.18 s: f = -0.801884391105, ‖∇f‖ = 4.1905e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   51, time  204.55 s: f = -0.801884815983, ‖∇f‖ = 6.9018e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   52, time  205.72 s: f = -0.801885013427, ‖∇f‖ = 3.8025e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   53, time  206.89 s: f = -0.801885126302, ‖∇f‖ = 1.9306e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   54, time  208.06 s: f = -0.801885184513, ‖∇f‖ = 3.3083e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   55, time  209.30 s: f = -0.801885308658, ‖∇f‖ = 4.9014e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   56, time  210.56 s: f = -0.801885502272, ‖∇f‖ = 1.1303e-04, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   57, time  211.87 s: f = -0.801885922461, ‖∇f‖ = 7.5880e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   58, time  213.22 s: f = -0.801886457901, ‖∇f‖ = 7.2957e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   59, time  214.50 s: f = -0.801886614664, ‖∇f‖ = 6.8816e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   60, time  216.92 s: f = -0.801886696733, ‖∇f‖ = 3.0687e-05, α = 4.26e-01, m = 32, nfg = 2
[ Info: LBFGS: iter   61, time  218.01 s: f = -0.801886716271, ‖∇f‖ = 2.1581e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   62, time  219.10 s: f = -0.801886732686, ‖∇f‖ = 1.7659e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   63, time  220.30 s: f = -0.801886790357, ‖∇f‖ = 4.1045e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   64, time  221.53 s: f = -0.801886827022, ‖∇f‖ = 4.0831e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   65, time  222.57 s: f = -0.801886871472, ‖∇f‖ = 4.1034e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   66, time  223.64 s: f = -0.801886949562, ‖∇f‖ = 5.1171e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   67, time  224.79 s: f = -0.801887066612, ‖∇f‖ = 4.5902e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   68, time  226.00 s: f = -0.801887172301, ‖∇f‖ = 7.4810e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   69, time  227.33 s: f = -0.801887249759, ‖∇f‖ = 3.9619e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   70, time  228.45 s: f = -0.801887292124, ‖∇f‖ = 1.3999e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   71, time  229.64 s: f = -0.801887312574, ‖∇f‖ = 1.0813e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   72, time  230.93 s: f = -0.801887349749, ‖∇f‖ = 1.1335e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   73, time  232.31 s: f = -0.801887427066, ‖∇f‖ = 1.9028e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   74, time  233.68 s: f = -0.801887495625, ‖∇f‖ = 1.9286e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: iter   75, time  237.82 s: f = -0.801887521350, ‖∇f‖ = 2.5284e-05, α = 5.43e-01, m = 32, nfg = 2
[ Info: LBFGS: iter   76, time  239.82 s: f = -0.801887560129, ‖∇f‖ = 2.7094e-05, α = 1.00e+00, m = 32, nfg = 1
[ Info: LBFGS: converged after 77 iterations and time 241.32 s: f = -0.801887571684, ‖∇f‖ = 7.8819e-06

````

### Verifying the result

Having found the fixed point, we have essentially contracted the entire partition function
and we can start computing observables. The free energy per site for example is just given by
the final value of the cost function we have just optimized.

````julia
@show f
````

````
-0.8018875716841548
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
0.00011017609950225715
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

