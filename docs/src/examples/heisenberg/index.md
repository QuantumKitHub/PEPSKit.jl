```@meta
EditURL = "../../../../examples/heisenberg/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/heisenberg/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/heisenberg/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/heisenberg)


# [Optimizing the 2D Heisenberg model](@id examples_heisenberg)

In this example we want to provide a basic rundown of PEPSKit's optimization workflow for
PEPS. To that end, we will consider the two-dimensional Heisenberg model on a square lattice

```math
H = \sum_{\langle i,j \rangle} \left ( J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j \right )
```

Here, we want to set $J_x = J_y = J_z = 1$ where the Heisenberg model is in the antiferromagnetic
regime. Due to the bipartite sublattice structure of antiferromagnetic order one needs a
PEPS ansatz with a $2 \times 2$ unit cell. This can be circumvented by performing a unitary
sublattice rotation on all B-sites resulting in a change of parameters to
$(J_x, J_y, J_z)=(-1, 1, -1)$. This gives us a unitarily equivalent Hamiltonian (with the
same spectrum) with a ground state on a single-site unit cell.

Let us get started by fixing the random seed of this example to make it deterministic:

````julia
using Random
Random.seed!(123456789);
````

We're going to need only two packages: `TensorKit`, since we use that for all the underlying
tensor operations, and `PEPSKit` itself. So let us import these:

````julia
using TensorKit, PEPSKit
````

## Defining the Heisenberg Hamiltonian

To create the sublattice rotated Heisenberg Hamiltonian on an infinite square lattice, we use
the `heisenberg_XYZ` model defined in PEPSKit for the `InfiniteSquare` lattice:

````julia
H = heisenberg_XYZ(InfiniteSquare(); Jx = -1, Jy = 1, Jz = -1)
````

````
LocalOperator{Any, TensorKit.ComplexSpace}(TensorKit.ComplexSpace[ℂ^2;;], Dict{Vector{CartesianIndex{2}}, Any}([CartesianIndex(1, 1), CartesianIndex(1, 2)] => TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}(ComplexF64[-0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.25 + 0.0im], (ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2)), [CartesianIndex(1, 1), CartesianIndex(2, 1)] => TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}(ComplexF64[-0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.25 + 0.0im], (ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2))))
````

## Setting up the algorithms and initial guesses

Next, we set the simulation parameters. During optimization, the PEPS will be contracted
using CTMRG and the PEPS gradient will be computed by differentiating through the CTMRG
routine using AD. Since the algorithmic stack that implements this is rather elaborate,
the amount of settings one can configure is also quite large. To reduce this complexity,
PEPSKit defaults to (presumably) reasonable settings which also dynamically adapts to the
user-specified parameters.

First, we set the bond dimension `Dbond` of the virtual PEPS indices and the environment
dimension `χenv` of the virtual corner and transfer matrix indices.

````julia
Dbond = 2
χenv = 16;
````

To configure the CTMRG algorithm, we create a `NamedTuple` containing different keyword
arguments. To see a description of all arguments, see the docstring of
[`leading_boundary`](@ref). Here, we want to converge the CTMRG environments up to a
specific tolerance and during the CTMRG run keep all index dimensions fixed:

````julia
boundary_alg = (; tol = 1.0e-10, trunc = (; alg = :FixedSpaceTruncation));
````

Let us also configure the optimizer algorithm. We are going to optimize the PEPS using the
L-BFGS optimizer from [OptimKit](https://github.com/Jutho/OptimKit.jl). Again, we specify
the convergence tolerance (for the gradient norm) as well as the maximal number of iterations
and the BFGS memory size (which is used to approximate the Hessian):

````julia
optimizer_alg = (; alg = :LBFGS, tol = 1.0e-4, maxiter = 100, lbfgs_memory = 16);
````

Additionally, during optimization, we want to reuse the previous CTMRG environment to
initialize the CTMRG run of the current optimization step using the `reuse_env` argument.
And to control the output information, we set the `verbosity`:

````julia
reuse_env = true
verbosity = 3;
````

Next, we initialize a random PEPS which will be used as an initial guess for the
optimization. To get a PEPS with physical dimension 2 (since we have a spin-1/2 Hamiltonian)
with complex-valued random Gaussian entries, we set:

````julia
peps₀ = InfinitePEPS(randn, ComplexF64, ℂ^2, ℂ^Dbond)
````

````
InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}(TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}[TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}(ComplexF64[0.07382174258286094 + 0.12820373667088403im, 0.2553716885006697 - 0.4358399804354269im, 0.7897519397510839 + 0.9113654266438473im, -1.0272416446076236 - 0.12635062198157215im, 0.16833628450178303 - 0.10088950122180829im, -1.6804460553576506 + 0.29081053879369084im, -0.9702030532300809 + 0.010730752411986726im, 0.6844811667615024 + 0.09101537356941222im, 0.5085938050744258 + 0.3786892551842583im, -0.6153328223084331 + 0.10417896606055738im, 1.0020057959636561 - 1.4704891009758718im, 0.6024931811537675 - 1.0348374874397468im, -0.027201695938305456 + 0.5778042099380925im, 1.0707115218777772 - 0.5747168579241235im, 0.09232089635078945 + 0.6143070126937361im, -0.5819741818511422 - 0.9842624134267605im, 1.2332543810053822 - 1.7783531996396438im, 1.2251189302516847 - 0.6853683793073324im, 0.8887723728085348 + 0.7809798723615474im, 1.5333834584675397 - 0.13856216581406375im, 0.1406381347783769 + 0.6630243440357264im, 0.7212056487788236 + 0.24320971945037498im, -0.7294596235434386 + 0.40327909254711103im, 0.9991347929322827 + 0.0017902515981375842im, 0.34282910982693904 - 0.4865238029567361im, -0.7437083517319159 - 0.6895708849529253im, 0.9380949844871762 - 0.6985342237892025im, -0.8981092940164176 + 0.9720706252141459im, -0.8897079923413616 - 0.7145412189457411im, -1.6099412157243007 + 0.8855200965611144im, 0.07771261045117502 - 0.6400190994609709im, 0.7357380595021633 + 0.4626916850143416im], ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'));;])
````

The last thing we need before we can start the optimization is an initial CTMRG environment.
Typically, a random environment which we converge on `peps₀` serves as a good starting point.
To contract a PEPS starting from an environment using CTMRG, we call [`leading_boundary`](@ref):

````julia
env_random = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χenv);
env₀, info_ctmrg = leading_boundary(env_random, peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -2.749614463601e+00 +3.639628057806e+00im	err = 1.0000e+00
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201048445e-11	time = 0.39 sec

````

Besides the converged environment, `leading_boundary` also returns a `NamedTuple` of
informational quantities which contains, among other things, a `contraction_metric` tuple.
This may contain different quantities depending on the method of contraction, and for this
CTMRG variant we return the last maximal truncation error (the SVD approximation
error maximized over all spatial directions and unit cell entries) as well as the condition
number of the decomposition (the ratio of largest to smallest singular value):

````julia
@show info_ctmrg.contraction_metrics;
````

````
info_ctmrg.contraction_metrics = (truncation_error = 0.00080763328242187,)

````

## Ground state search

Finally, we can start the optimization by calling [`fixedpoint`](@ref) on `H` with our
settings for the boundary (CTMRG) algorithm and the optimizer. This might take a while
(especially the precompilation of AD code in this case):

````julia
peps, env, E, info_opt = fixedpoint(
    H, peps₀, env₀; boundary_alg, optimizer_alg, reuse_env, verbosity
);
````

````
[ Info: LBFGS: initializing with f = 6.016453104372e-04, ‖∇f‖ = 9.3548e-01
[ Info: LBFGS: iter    1, Δt  4.40 s: f = -4.897965201577e-01, ‖∇f‖ = 6.0022e-01, α = 5.94e+01, m = 0, nfg = 5
[ Info: LBFGS: iter    2, Δt 994.3 ms: f = -5.019846351520e-01, ‖∇f‖ = 5.3738e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, Δt  2.38 s: f = -5.231639268895e-01, ‖∇f‖ = 3.9927e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt  1.07 s: f = -5.386543630087e-01, ‖∇f‖ = 4.1552e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, Δt  3.13 s: f = -5.498211740288e-01, ‖∇f‖ = 4.4002e-01, α = 6.90e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, Δt  1.29 s: f = -5.690169637831e-01, ‖∇f‖ = 4.8450e-01, α = 2.26e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, Δt 529.5 ms: f = -5.871277574762e-01, ‖∇f‖ = 4.1970e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 547.0 ms: f = -6.001554859451e-01, ‖∇f‖ = 2.1792e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 478.2 ms: f = -6.068836019151e-01, ‖∇f‖ = 1.9566e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 460.6 ms: f = -6.250397688264e-01, ‖∇f‖ = 3.0330e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 456.7 ms: f = -6.391660377815e-01, ‖∇f‖ = 2.3075e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 443.5 ms: f = -6.471796200678e-01, ‖∇f‖ = 2.6051e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 409.2 ms: f = -6.503370024355e-01, ‖∇f‖ = 1.6112e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 426.0 ms: f = -6.546061095279e-01, ‖∇f‖ = 7.7752e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 441.0 ms: f = -6.559626479181e-01, ‖∇f‖ = 5.1323e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 484.9 ms: f = -6.570345924643e-01, ‖∇f‖ = 5.6663e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 450.5 ms: f = -6.586101543613e-01, ‖∇f‖ = 4.5249e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 433.2 ms: f = -6.594210783242e-01, ‖∇f‖ = 4.8908e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt 498.2 ms: f = -6.595829405654e-01, ‖∇f‖ = 5.7868e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, Δt 430.3 ms: f = -6.598106978446e-01, ‖∇f‖ = 1.7743e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, Δt 417.0 ms: f = -6.598737919456e-01, ‖∇f‖ = 1.4674e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, Δt  2.58 s: f = -6.600722400318e-01, ‖∇f‖ = 1.9297e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, Δt 465.3 ms: f = -6.602319321218e-01, ‖∇f‖ = 1.7537e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, Δt 503.2 ms: f = -6.603792238918e-01, ‖∇f‖ = 2.3875e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, Δt 612.1 ms: f = -6.604618340573e-01, ‖∇f‖ = 2.3372e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, Δt 560.1 ms: f = -6.605536672329e-01, ‖∇f‖ = 1.2672e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, Δt 602.0 ms: f = -6.606170021812e-01, ‖∇f‖ = 1.0507e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, Δt 400.3 ms: f = -6.608142105271e-01, ‖∇f‖ = 1.8082e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, Δt 436.9 ms: f = -6.609609282358e-01, ‖∇f‖ = 1.7516e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, Δt 400.4 ms: f = -6.610389077620e-01, ‖∇f‖ = 1.1313e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, Δt 434.7 ms: f = -6.610872587443e-01, ‖∇f‖ = 1.0263e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, Δt 406.8 ms: f = -6.611211818094e-01, ‖∇f‖ = 8.8770e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, Δt 412.0 ms: f = -6.611798476728e-01, ‖∇f‖ = 1.1488e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, Δt 884.4 ms: f = -6.612071947408e-01, ‖∇f‖ = 8.8667e-03, α = 5.31e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   35, Δt 435.3 ms: f = -6.612262742191e-01, ‖∇f‖ = 6.4670e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, Δt 471.0 ms: f = -6.612605504181e-01, ‖∇f‖ = 5.8761e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, Δt 420.2 ms: f = -6.612675211696e-01, ‖∇f‖ = 1.2105e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, Δt 439.7 ms: f = -6.612838270545e-01, ‖∇f‖ = 4.9173e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, Δt 413.4 ms: f = -6.612924067926e-01, ‖∇f‖ = 4.5919e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, Δt 452.7 ms: f = -6.613079388895e-01, ‖∇f‖ = 6.2780e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, Δt 415.8 ms: f = -6.613408208859e-01, ‖∇f‖ = 8.9068e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, Δt 442.0 ms: f = -6.614138219965e-01, ‖∇f‖ = 1.7165e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, Δt 469.8 ms: f = -6.614875628763e-01, ‖∇f‖ = 2.7430e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, Δt 484.5 ms: f = -6.616599039377e-01, ‖∇f‖ = 1.9063e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, Δt 508.3 ms: f = -6.618786281540e-01, ‖∇f‖ = 2.2182e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, Δt 482.6 ms: f = -6.619399793772e-01, ‖∇f‖ = 2.5713e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, Δt 530.0 ms: f = -6.620476596371e-01, ‖∇f‖ = 1.8967e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, Δt 532.8 ms: f = -6.621061626373e-01, ‖∇f‖ = 3.7646e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   49, Δt 484.7 ms: f = -6.622529553205e-01, ‖∇f‖ = 1.3393e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, Δt 542.6 ms: f = -6.623265267194e-01, ‖∇f‖ = 1.1674e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, Δt 507.3 ms: f = -6.623899608783e-01, ‖∇f‖ = 1.0881e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, Δt 473.0 ms: f = -6.624416718050e-01, ‖∇f‖ = 1.3579e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, Δt  3.23 s: f = -6.624656666090e-01, ‖∇f‖ = 8.4389e-03, α = 4.81e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   54, Δt 448.4 ms: f = -6.624780725324e-01, ‖∇f‖ = 5.0929e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, Δt 473.7 ms: f = -6.624880237458e-01, ‖∇f‖ = 5.2634e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, Δt 634.6 ms: f = -6.624976980387e-01, ‖∇f‖ = 3.7107e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, Δt 606.9 ms: f = -6.625008729028e-01, ‖∇f‖ = 3.0338e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, Δt 533.3 ms: f = -6.625030441100e-01, ‖∇f‖ = 2.4290e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, Δt 408.7 ms: f = -6.625070603627e-01, ‖∇f‖ = 2.1063e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, Δt 494.5 ms: f = -6.625095215489e-01, ‖∇f‖ = 2.1527e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, Δt 415.4 ms: f = -6.625099429974e-01, ‖∇f‖ = 3.4179e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   62, Δt 433.8 ms: f = -6.625114671854e-01, ‖∇f‖ = 9.0254e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, Δt 389.3 ms: f = -6.625117485785e-01, ‖∇f‖ = 7.0329e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, Δt 415.5 ms: f = -6.625121901075e-01, ‖∇f‖ = 7.7233e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, Δt 430.6 ms: f = -6.625124954826e-01, ‖∇f‖ = 1.5505e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, Δt 453.0 ms: f = -6.625128489607e-01, ‖∇f‖ = 7.2949e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, Δt 475.3 ms: f = -6.625131160497e-01, ‖∇f‖ = 5.6433e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   68, Δt 457.9 ms: f = -6.625133163141e-01, ‖∇f‖ = 6.8135e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, Δt 461.4 ms: f = -6.625134072750e-01, ‖∇f‖ = 1.8466e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, Δt 487.9 ms: f = -6.625137425655e-01, ‖∇f‖ = 5.2342e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, Δt 416.6 ms: f = -6.625138211990e-01, ‖∇f‖ = 3.5251e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   72, Δt 440.8 ms: f = -6.625138890509e-01, ‖∇f‖ = 4.0623e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, Δt 946.9 ms: f = -6.625139277394e-01, ‖∇f‖ = 6.8694e-04, α = 4.40e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   74, Δt 449.3 ms: f = -6.625139828932e-01, ‖∇f‖ = 4.5294e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   75, Δt 471.3 ms: f = -6.625140571619e-01, ‖∇f‖ = 4.6864e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, Δt 419.9 ms: f = -6.625140852255e-01, ‖∇f‖ = 7.8787e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, Δt 425.5 ms: f = -6.625141218270e-01, ‖∇f‖ = 3.9710e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, Δt 465.3 ms: f = -6.625141596871e-01, ‖∇f‖ = 2.8416e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, Δt 429.4 ms: f = -6.625141956195e-01, ‖∇f‖ = 3.7104e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, Δt 423.9 ms: f = -6.625142387935e-01, ‖∇f‖ = 3.0323e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   81, Δt 841.3 ms: f = -6.625142514540e-01, ‖∇f‖ = 3.4345e-04, α = 4.40e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   82, Δt 403.1 ms: f = -6.625142677939e-01, ‖∇f‖ = 1.1308e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   83, Δt 381.5 ms: f = -6.625142732103e-01, ‖∇f‖ = 1.0425e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   84, Δt 404.3 ms: f = -6.625142803843e-01, ‖∇f‖ = 1.4568e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   85, Δt 394.3 ms: f = -6.625142872106e-01, ‖∇f‖ = 1.4922e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   86, Δt 419.4 ms: f = -6.625142907248e-01, ‖∇f‖ = 1.2904e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 87 iterations and time  2.94 m: f = -6.625142931720e-01, ‖∇f‖ = 4.7603e-05

````

Note that `fixedpoint` returns the final optimized PEPS, the last converged environment,
the final energy estimate as well as a `NamedTuple` of diagnostics. This allows us to, e.g.,
analyze the number of cost function calls or the history of gradient norms to evaluate
the convergence rate:

````julia
@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
````

````
info_opt.fg_evaluations = 102
info_opt.gradnorms[1:10:end] = [0.9354752017288198, 0.3033034100486219, 0.017742664173948874, 0.011312957837480311, 0.006277953457599366, 0.011674249923348009, 0.0021527144858215145, 0.0005234177465405374, 0.000303234363025293]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142931720018

````

While this energy is in the right ballpark, there is still quite some deviation from the
accurate reference energy. This, however, can be attributed to the small bond dimension - an
optimization with larger bond dimension would approach this value much more closely.

A more reasonable comparison would be against another finite bond dimension PEPS simulation.
For example, Juraj Hasik's data from $J_1\text{-}J_2$
[PEPS simulations](https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat)
yields $E_{D=2,\chi=16}=-0.660231\dots$ which is more in line with what we find here.

## Compute the correlation lengths and transfer matrix spectra

In practice, in order to obtain an accurate and variational energy estimate, one would need
to compute multiple energies at different environment dimensions and extrapolate in, e.g.,
the correlation length or the second gap of the transfer matrix spectrum. For that, we would
need the [`correlation_length`](@ref) function, which computes the horizontal and vertical
correlation lengths and transfer matrix spectra for all unit cell coordinates:

````julia
ξ_h, ξ_v, λ_h, λ_v = correlation_length(peps, env)
@show ξ_h ξ_v;
````

````
ξ_h = [1.0343019611625741]
ξ_v = [1.0242271105480807]

````

## Computing observables

As a last thing, we want to see how we can compute expectation values of observables, given
the optimized PEPS and its CTMRG environment. To compute, e.g., the magnetization, we first
need to define the observable as a `TensorMap`:

````julia
σ_z = TensorMap([1.0 0.0; 0.0 -1.0], ℂ^2, ℂ^2)
````

````
2←2 TensorMap{Float64, TensorKit.ComplexSpace, 1, 1, Vector{Float64}}:
 codomain: ⊗(ℂ^2)
 domain: ⊗(ℂ^2)
 blocks: 
 * Trivial() => 2×2 reshape(view(::Vector{Float64}, 1:4), 2, 2) with eltype Float64:
 1.0   0.0
 0.0  -1.0
````

In order to be able to contract it with the PEPS and environment, we define need to define a
`LocalOperator` and specify on which physical spaces and sites the observable acts. That way,
the PEPS-environment-operator contraction gets automatically generated (also works for
multi-site operators!). See the [`LocalOperator`](@ref) docstring for more details.
The magnetization is just a single-site observable, so we have:

````julia
M = LocalOperator(fill(ℂ^2, 1, 1), (CartesianIndex(1, 1),) => σ_z)
````

````
LocalOperator{Any, TensorKit.ComplexSpace}(TensorKit.ComplexSpace[ℂ^2;;], Dict{Vector{CartesianIndex{2}}, Any}([CartesianIndex(1, 1)] => TensorMap{Float64, TensorKit.ComplexSpace, 1, 1, Vector{Float64}}([1.0, 0.0, 0.0, -1.0], ℂ^2 ← ℂ^2)))
````

Finally, to evaluate the expecation value on the `LocalOperator`, we call:

````julia
@show expectation_value(peps, M, env);
````

````
expectation_value(peps, M, env) = -0.753314967849306 + 2.983724378680108e-16im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

