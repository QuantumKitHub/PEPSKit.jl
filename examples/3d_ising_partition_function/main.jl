using Markdown #hide
md"""
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

Let's start by making the example deterministic and importing the required packages:
"""

using Random
using LinearAlgebra
using PEPSKit, TensorKit
using KrylovKit, OptimKit, Zygote

Random.seed!(81812781144);

md"""
## Defining the partition function

Just as in the 2D case, the first step is to define the partition function as a tensor
network. The procedure is exactly the same as before, the only difference being that now
every spin participates in interactions associated to six links adjacent to that site. This
means that the partition function can be written as an infinite 3D network with a single
constituent rank-6 [`PEPSKit.PEPOTensor`](@ref) `O` located at each site of the cubic
lattice. To verify our example we will check the magnetization and energy, so we also define
the corresponding rank-6 tensors `M` and `E` while we're at it.
"""

function three_dimensional_classical_ising(; beta, J=1.0)
    K = beta * J

    ## Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    ## local partition function tensor
    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    ## magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4 -5 -6] :=
        M[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    ## bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* q
    @tensor e_x[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * e[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_y[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * e[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_z[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * e[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    e = e_x + e_y + e_z

    ## fixed tensor map space for all three
    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end;

md"""
Let's initialize these tensors at inverse temperature ``\beta=0.2391``, which corresponds to
a slightly lower temperature than the critical value ``\beta_c=0.2216544…``
"""

beta = 0.2391
O, M, E = three_dimensional_classical_ising(; beta)
O isa PEPSKit.PEPOTensor

md"""
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

Using PEPSKit.jl, this cost function and its gradient can be computed, after which we
can use [OptimKit.jl](https://github.com/Jutho/OptimKit.jl) to actually optimize it. We can
immediately recognize the denominator ``\langle \psi | \psi \rangle`` as the familiar PEPS
norm, where we can compute the norm per site as the [`network_value`](@ref) of the
corresponding [`InfiniteSquareNetwork`](@ref) by contracting it with the CTMRG algorithm.
Similarly, the numerator ``\langle \psi | T | \psi \rangle`` is nothing more than an
`InfiniteSquareNetwork` consisting of three layers corresponding to the ket, transfer
operator and bra objects. This object can also be constructed and contracted in a
straightforward way, so we can again compute its `network_value`.

So to define our cost function, we just need to construct the transfer operator as an
[`InfinitePEPO`](@ref), contruct the both relevant infinite 2D contractible networks from
the current PEPS and this transfer operator, and specify a contraction algorithm we can use
to compute the values of these two networks. In addition, we'll specify the specific reverse
rule algorithm that will be used to compute the gradient of this cost function.
"""

boundary_alg = SimultaneousCTMRG(; maxiter=150, tol=1e-8, verbosity=1)
rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
)
T = InfinitePEPO(O)

function pepo_costfun((psi, env2, env3))
    ## use Zygote to compute the gradient automatically
    E, gs = withgradient(psi) do ψ
        ## construct the PEPS norm network
        n2 = InfiniteSquareNetwork(ψ)
        ## contract this network
        env2′, info = PEPSKit.hook_pullback(
            leading_boundary, env2, n2, boundary_alg; alg_rrule=rrule_alg
        )
        ## construct the PEPS-PEPO-PEPS overlap network
        n3 = InfiniteSquareNetwork(ψ, T)
        ## contract this network
        env3′, info = PEPSKit.hook_pullback(
            leading_boundary, env3, n3, boundary_alg; alg_rrule=rrule_alg
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
        end
        ## compute the network values per site
        λ3 = network_value(n3, env3)
        λ2 = network_value(n2, env2)
        ## use this to compute the actual cost function
        return -log(real(λ3 / λ2))
    end
    g = only(gs)
    return E, g
end;

md"""
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
"""

function pepo_retract(x, η, α)
    x´_partial, ξ = PEPSKit.peps_retract(x[1:2], η, α)
    x´ = (x´_partial..., deepcopy(x[3]))
    return x´, ξ
end
function pepo_transport!(ξ, x, η, α, x´)
    return PEPSKit.peps_transport!(ξ, x[1:2], η, α, x´[1:2])
end;

md"""
### Finding the fixed point

All that is left then is to specify the virtual spaces of the PEPS and the two environments,
initialize them in the appropriate way, choose an optimization algortithm and call the
`optimize` function from OptimKit.jl to get our desired PEPS fixed point.
"""

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

md"""
### Verifying the result

Having found the fixed point, we have essentially contracted the entire partition function
and we can start computing observables. The free energy per site for example is just given by
the final value of the cost function we have just optimized.
"""

@show f

md"""
As another check, we can compute the magnetization per site and compare it to a [reference
value obtaind through Monte-Carlo simulations](@cite hasenbusch_monte_2001).
"""

n3_final = InfiniteSquareNetwork(psi_final, T)
num = PEPSKit.contract_local_tensor((1, 1, 1), M, n3_final, env3_final)
denom = PEPSKit._contract_site((1, 1), n3_final, env3_final)
m = abs(num / denom)

m_ref = 0.667162

@show abs(m - m_ref)
