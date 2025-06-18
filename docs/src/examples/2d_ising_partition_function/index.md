```@meta
EditURL = "../../../../examples/2d_ising_partition_function/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/2d_ising_partition_function/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/2d_ising_partition_function/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/2d_ising_partition_function)


# [The 2D classical Ising model using CTMRG](@id e_2d_ising)

While PEPSKit has a lot of use in quantum systems, describing states using InfinitePEPS that can be contracted via CTMRG or [boundary MPS techniques](@ref e_boundary_mps), here we shift our focus to classical physics.
We consider the 2D classical Ising model and compute its partition function defined as:

```math
\mathcal{Z}(\beta) = \sum_{\{s\}} \exp(-\beta H(s)) \text{ with } H(s) = -J \sum_{\langle i, j \rangle} s_i s_j .
```

where the classical spins $s_i \in \{+1, -1\}$ are located on the vertices $i$ of a 2D
square lattice. The idea is to encode the partition function as an infinite square network
consisting of local rank-4 tensors, which can then be contracted using CTMRG. An infinite
square network of these rank-4 tensors can be represented as an
[`InfinitePartitionFunction`](@ref) object, as we will see.

But first, let's seed the RNG and import all required modules:

````julia
using Random, LinearAlgebra
using TensorKit, PEPSKit
using QuadGK
Random.seed!(234923);
````

## Defining the partition function

The first step is to define the rank-4 tensor that, when contracted on a square lattice,
evaluates to the partition function value at a given $\beta$. This is done through a
[fairly generic procedure](@cite haegeman_diagonalizing_2017) where the interaction weights
are distributed among vertex tensors in an appropriate way. Concretely, here we first define
a 'link' matrix containing the Boltzmann weights associated to all possible spin
configurations across a given link on the lattice. Next, we define site tensors as
delta-tensors that ensiure that the spin value on all adjacent links is the same. Since we
only want tensors on the sites in the end, we can symmetrically absorb the link weight
tensors into the site tensors, which gives us exactly the kind of network we're looking for.
Since we later want to compute the magnetization and energy to check our results, we define
the appropriate rank-4 tensors here as well while we're at it.

````julia
function classical_ising(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4] := O[3 4; 2 1] * nt[-3; 3] * nt[-4; 4] * nt[-2; 2] * nt[-1; 1]

    # magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4] := M[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* nt
    @tensor e_hor[-1 -2; -3 -4] :=
        O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * e[-4; 4]
    @tensor e_vert[-1 -2; -3 -4] :=
        O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * e[-3; 3] * nt[-4; 4]
    e = e_hor + e_vert

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end;
````

So let's initialize these tensors at inverse temperature ``\beta=0.6``, check that
they are indeed rank-4 and construct the corresponding `InfinitePartitionFunction`:

````julia
beta = 0.6
O, M, E = classical_ising(; beta)
@show space(O)
Z = InfinitePartitionFunction(O)
````

````
InfinitePartitionFunction{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}(TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}[TensorMap((ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2)):
[:, :, 1, 1] =
  3.169519816780443 + 0.0im  0.4999999999999995 + 0.0im
 0.4999999999999995 + 0.0im  0.1505971059561009 + 0.0im

[:, :, 2, 1] =
 0.4999999999999995 + 0.0im  0.1505971059561009 + 0.0im
 0.1505971059561009 + 0.0im  0.4999999999999995 + 0.0im

[:, :, 1, 2] =
 0.4999999999999995 + 0.0im  0.1505971059561009 + 0.0im
 0.1505971059561009 + 0.0im  0.4999999999999995 + 0.0im

[:, :, 2, 2] =
 0.1505971059561009 + 0.0im  0.4999999999999995 + 0.0im
 0.4999999999999995 + 0.0im   3.169519816780443 + 0.0im
;;])
````

## Contracting the partition function

Next, we can contract the partition function as per usual by constructing a `CTMRGEnv` with
a specified environment virtual space and calling `leading_boundary` with appropriate
settings:

````julia
Venv = ℂ^20
env₀ = CTMRGEnv(Z, Venv)
env, = leading_boundary(env₀, Z; tol=1e-8, maxiter=500);
````

````
[ Info: CTMRG init:	obj = +1.784252138312e+00 -1.557258880375e+00im	err = 1.0000e+00
[ Info: CTMRG conv 63:	obj = +3.353928644031e+00	err = 4.5903314811e-09	time = 8.12 sec

````

Note that CTMRG environments for partition functions differ from the PEPS environments only
by the edge tensors. Instead of two legs connecting the edges and the PEPS-PEPS sandwich,
there is only one leg connecting the edges and the partition function tensor, meaning that
the edge tensors are now rank-3:

````julia
space.(env.edges)
````

````
4×1×1 Array{TensorKit.TensorMapSpace{TensorKit.ComplexSpace, 2, 1}, 3}:
[:, :, 1] =
 (ℂ^20 ⊗ ℂ^2) ← ℂ^20
 (ℂ^20 ⊗ ℂ^2) ← ℂ^20
 (ℂ^20 ⊗ (ℂ^2)') ← ℂ^20
 (ℂ^20 ⊗ (ℂ^2)') ← ℂ^20
````

To compute the value of the partition function, we have to contract `Z` with the converged
environment using [`network_value`](@ref). Additionally, we will compute the magnetization
and energy (per site), again using [`expectation_value`](@ref) but this time also specifying
the index in the unit cell where we want to insert the local tensor:

````julia
λ = network_value(Z, env)
m = expectation_value(Z, (1, 1) => M, env)
e = expectation_value(Z, (1, 1) => E, env)
@show λ m e;
````

````
λ = 3.3539286440313782 - 5.873212040152551e-16im
m = 0.9736086674403001 + 1.8262157316829647e-17im
e = -1.8637796145082437 - 1.4609725853463717e-16im

````

## Comparing against the exact Onsager solution

In order to assess our results, we will compare against the
[exact Onsager solution](https://en.wikipedia.org/wiki/Square_lattice_Ising_model#Exact_solution)
of the 2D classical Ising model. To that end, we compute the exact free energy,
magnetization and energy per site (where we use `quadgk` to perform integrals of an
auxiliary variable from $0$ to $\pi/2$):

````julia
function classical_ising_exact(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J

    k = 1 / sinh(2 * K)^2
    F = quadgk(
        theta -> log(cosh(2 * K)^2 + 1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * theta))), 0, pi
    )[1]
    f = -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)

    m = 1 - (sinh(2 * K))^(-4) > 0 ? (1 - (sinh(2 * K))^(-4))^(1 / 8) : 0

    E = quadgk(theta -> 1 / sqrt(1 - (4 * k) * (1 + k)^(-2) * sin(theta)^2), 0, pi / 2)[1]
    e = -J * cosh(2 * K) / sinh(2 * K) * (1 + 2 / pi * (2 * tanh(2 * K)^2 - 1) * E)

    return f, m, e
end

f_exact, m_exact, e_exact = classical_ising_exact(; beta);
````

And indeed, we do find agreement between the exact and CTMRG values (keeping in mind that
energy accuracy is limited by the environment dimension and the lack of proper
extrapolation):

````julia
@show (-log(λ) / beta - f_exact) / f_exact
@show (abs(m) - abs(m_exact)) / abs(m_exact)
@show (e - e_exact) / e_exact;
````

````
(-(log(λ)) / beta - f_exact) / f_exact = -6.605563039765528e-16 - 1.447068124555315e-16im
(abs(m) - abs(m_exact)) / abs(m_exact) = -4.561270094458082e-16
(e - e_exact) / e_exact = -0.023732068099090543 + 7.652732508485748e-17im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

