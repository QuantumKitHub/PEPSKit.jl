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
the `heisenberg_XYZ` method from [MPSKitModels](https://quantumkithub.github.io/MPSKitModels.jl/dev/)
which is redefined for the `InfiniteSquare` and reexported in PEPSKit:

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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201048445e-11	time = 14.48 sec

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
[ Info: LBFGS: iter    1, Δt  4.48 s: f = -4.897965201580e-01, ‖∇f‖ = 6.0022e-01, α = 5.94e+01, m = 0, nfg = 5
[ Info: LBFGS: iter    2, Δt  2.58 s: f = -5.019846351523e-01, ‖∇f‖ = 5.3738e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, Δt 432.2 ms: f = -5.231639268895e-01, ‖∇f‖ = 3.9927e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 970.0 ms: f = -5.386543630087e-01, ‖∇f‖ = 4.1552e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, Δt  4.16 s: f = -5.498211740251e-01, ‖∇f‖ = 4.4002e-01, α = 6.90e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, Δt  1.33 s: f = -5.690169637869e-01, ‖∇f‖ = 4.8450e-01, α = 2.26e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, Δt 433.5 ms: f = -5.871277574854e-01, ‖∇f‖ = 4.1970e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 391.6 ms: f = -6.001554859580e-01, ‖∇f‖ = 2.1792e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 414.0 ms: f = -6.068836019259e-01, ‖∇f‖ = 1.9566e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 390.2 ms: f = -6.250397688236e-01, ‖∇f‖ = 3.0330e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 390.3 ms: f = -6.391660378052e-01, ‖∇f‖ = 2.3075e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 380.1 ms: f = -6.471796201327e-01, ‖∇f‖ = 2.6051e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 332.9 ms: f = -6.503370024183e-01, ‖∇f‖ = 1.6112e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 376.2 ms: f = -6.546061095291e-01, ‖∇f‖ = 7.7752e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  1.77 s: f = -6.559626479321e-01, ‖∇f‖ = 5.1323e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 329.6 ms: f = -6.570345924896e-01, ‖∇f‖ = 5.6663e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 389.1 ms: f = -6.586101543932e-01, ‖∇f‖ = 4.5249e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 493.1 ms: f = -6.594210778183e-01, ‖∇f‖ = 4.8908e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt 342.8 ms: f = -6.595829407827e-01, ‖∇f‖ = 5.7868e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, Δt 342.1 ms: f = -6.598106975843e-01, ‖∇f‖ = 1.7743e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, Δt 267.7 ms: f = -6.598737917130e-01, ‖∇f‖ = 1.4674e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, Δt 309.9 ms: f = -6.600722396803e-01, ‖∇f‖ = 1.9297e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, Δt 326.9 ms: f = -6.602319318297e-01, ‖∇f‖ = 1.7537e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, Δt 308.9 ms: f = -6.603792228360e-01, ‖∇f‖ = 2.3875e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, Δt 351.1 ms: f = -6.604618345320e-01, ‖∇f‖ = 2.3372e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, Δt 343.6 ms: f = -6.605536666268e-01, ‖∇f‖ = 1.2672e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, Δt 334.2 ms: f = -6.606170017453e-01, ‖∇f‖ = 1.0507e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, Δt 364.4 ms: f = -6.608142109566e-01, ‖∇f‖ = 1.8082e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, Δt 387.7 ms: f = -6.609609273898e-01, ‖∇f‖ = 1.7516e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, Δt  1.75 s: f = -6.610389071492e-01, ‖∇f‖ = 1.1313e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, Δt 318.4 ms: f = -6.610872576646e-01, ‖∇f‖ = 1.0263e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, Δt 420.4 ms: f = -6.611211810871e-01, ‖∇f‖ = 8.8770e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, Δt 446.3 ms: f = -6.611798459058e-01, ‖∇f‖ = 1.1488e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, Δt 656.0 ms: f = -6.612071947563e-01, ‖∇f‖ = 8.8665e-03, α = 5.31e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   35, Δt 307.7 ms: f = -6.612262740421e-01, ‖∇f‖ = 6.4669e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, Δt 358.7 ms: f = -6.612605506187e-01, ‖∇f‖ = 5.8759e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, Δt 346.8 ms: f = -6.612675198975e-01, ‖∇f‖ = 1.2106e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, Δt 324.4 ms: f = -6.612838276419e-01, ‖∇f‖ = 4.9172e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, Δt 359.6 ms: f = -6.612924061975e-01, ‖∇f‖ = 4.5917e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, Δt 354.8 ms: f = -6.613079393444e-01, ‖∇f‖ = 6.2780e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, Δt 360.7 ms: f = -6.613408203258e-01, ‖∇f‖ = 8.9066e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, Δt 371.3 ms: f = -6.614138087699e-01, ‖∇f‖ = 1.7168e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, Δt 460.5 ms: f = -6.614875451473e-01, ‖∇f‖ = 2.7425e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, Δt  1.77 s: f = -6.616598971995e-01, ‖∇f‖ = 1.9051e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, Δt 355.1 ms: f = -6.618786794855e-01, ‖∇f‖ = 2.2179e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, Δt 521.4 ms: f = -6.619401735562e-01, ‖∇f‖ = 2.5702e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, Δt 388.2 ms: f = -6.620475294793e-01, ‖∇f‖ = 1.8974e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, Δt 352.7 ms: f = -6.621066223746e-01, ‖∇f‖ = 3.7569e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   49, Δt 372.5 ms: f = -6.622531473936e-01, ‖∇f‖ = 1.3377e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, Δt 419.8 ms: f = -6.623267435765e-01, ‖∇f‖ = 1.1685e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, Δt 380.1 ms: f = -6.623901380349e-01, ‖∇f‖ = 1.0878e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, Δt 378.3 ms: f = -6.624418631454e-01, ‖∇f‖ = 1.3507e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, Δt 809.4 ms: f = -6.624657620885e-01, ‖∇f‖ = 8.4255e-03, α = 4.80e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   54, Δt 407.5 ms: f = -6.624781462654e-01, ‖∇f‖ = 5.0928e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, Δt 384.1 ms: f = -6.624881004550e-01, ‖∇f‖ = 5.2502e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, Δt 396.3 ms: f = -6.624977081423e-01, ‖∇f‖ = 3.7440e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, Δt 358.3 ms: f = -6.625008752159e-01, ‖∇f‖ = 3.0223e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, Δt  1.79 s: f = -6.625030100734e-01, ‖∇f‖ = 2.4319e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, Δt 321.3 ms: f = -6.625070447461e-01, ‖∇f‖ = 2.1035e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, Δt 432.5 ms: f = -6.625095270010e-01, ‖∇f‖ = 2.1426e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, Δt 842.1 ms: f = -6.625106812720e-01, ‖∇f‖ = 1.7899e-03, α = 5.39e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   62, Δt 369.5 ms: f = -6.625114664427e-01, ‖∇f‖ = 9.0043e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, Δt 292.0 ms: f = -6.625118759472e-01, ‖∇f‖ = 8.0076e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, Δt 330.7 ms: f = -6.625122692623e-01, ‖∇f‖ = 9.1155e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, Δt 312.2 ms: f = -6.625126277124e-01, ‖∇f‖ = 7.9980e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, Δt 345.3 ms: f = -6.625129184274e-01, ‖∇f‖ = 6.8112e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, Δt 366.9 ms: f = -6.625132414561e-01, ‖∇f‖ = 9.1523e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   68, Δt 393.9 ms: f = -6.625134613669e-01, ‖∇f‖ = 7.9805e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, Δt 372.3 ms: f = -6.625136544615e-01, ‖∇f‖ = 5.1708e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, Δt 349.5 ms: f = -6.625138313259e-01, ‖∇f‖ = 7.7793e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, Δt 711.6 ms: f = -6.625138930116e-01, ‖∇f‖ = 4.4140e-04, α = 4.92e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   72, Δt 336.0 ms: f = -6.625139309014e-01, ‖∇f‖ = 3.5569e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, Δt  1.74 s: f = -6.625140346682e-01, ‖∇f‖ = 2.7207e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, Δt 278.1 ms: f = -6.625140541155e-01, ‖∇f‖ = 1.3444e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   75, Δt 397.8 ms: f = -6.625141433986e-01, ‖∇f‖ = 4.4360e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, Δt 412.7 ms: f = -6.625141810625e-01, ‖∇f‖ = 2.1664e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, Δt 285.8 ms: f = -6.625142097443e-01, ‖∇f‖ = 2.3226e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, Δt 276.3 ms: f = -6.625142395313e-01, ‖∇f‖ = 1.7812e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, Δt 345.0 ms: f = -6.625142482836e-01, ‖∇f‖ = 3.5608e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, Δt 334.4 ms: f = -6.625142671047e-01, ‖∇f‖ = 1.1471e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 81 iterations and time 10.28 m: f = -6.625142723775e-01, ‖∇f‖ = 9.5547e-05

````

Note that `fixedpoint` returns the final optimized PEPS, the last converged environment,
the final energy estimate as well as a `NamedTuple` of diagnostics. This allows us to, e.g.,
analyze the number of cost function calls or the history of gradient norms to evaluate
the convergence rate:

````julia
@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
````

````
info_opt.fg_evaluations = 96
info_opt.gradnorms[1:10:end] = [0.9354752017288198, 0.3033034105172946, 0.017742661587312764, 0.01131275905161872, 0.00627799743470083, 0.011685290429669634, 0.0021425732842698052, 0.0007779335235130188, 0.00011470754279726877]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142723774873

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
ξ_h = [1.03450066818636]
ξ_v = [1.0244839903801328]

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
expectation_value(peps, M, env) = -0.7532814688078179 - 6.245004513516506e-17im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

