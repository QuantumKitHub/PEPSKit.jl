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
LocalOperator{Tuple{Pair{Tuple{CartesianIndex{2}, CartesianIndex{2}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}, Pair{Tuple{CartesianIndex{2}, CartesianIndex{2}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}}, TensorKit.ComplexSpace}(TensorKit.ComplexSpace[ℂ^2;;], ((CartesianIndex(1, 1), CartesianIndex(1, 2)) => TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}(ComplexF64[-0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.25 + 0.0im], (ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2)), (CartesianIndex(1, 1), CartesianIndex(2, 1)) => TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}(ComplexF64[-0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.25 + 0.0im, 0.0 + 0.0im, -0.5 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, -0.25 + 0.0im], (ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2))))
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
boundary_alg = (; tol = 1.0e-10, trunc = (; alg = :fixedspace));
````

Let us also configure the optimizer algorithm. We are going to optimize the PEPS using the
L-BFGS optimizer from [OptimKit](https://github.com/Jutho/OptimKit.jl). Again, we specify
the convergence tolerance (for the gradient norm) as well as the maximal number of iterations
and the BFGS memory size (which is used to approximate the Hessian):

````julia
optimizer_alg = (; alg = :lbfgs, tol = 1.0e-4, maxiter = 100, lbfgs_memory = 16);
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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201048445e-11	time = 0.34 sec

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
info_ctmrg.contraction_metrics = (truncation_error = 0.00080763328242187, condition_number = 1.0752351780901926e10)

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
[ Info: LBFGS: iter    1, Δt  5.62 s: f = -4.897965201565e-01, ‖∇f‖ = 6.0022e-01, α = 5.94e+01, m = 0, nfg = 5
[ Info: LBFGS: iter    2, Δt  1.04 s: f = -5.019846351509e-01, ‖∇f‖ = 5.3738e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, Δt 342.4 ms: f = -5.231639268904e-01, ‖∇f‖ = 3.9927e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 747.3 ms: f = -5.386543630134e-01, ‖∇f‖ = 4.1552e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, Δt  2.15 s: f = -5.498211740486e-01, ‖∇f‖ = 4.4002e-01, α = 6.90e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, Δt 922.3 ms: f = -5.690169637654e-01, ‖∇f‖ = 4.8450e-01, α = 2.26e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, Δt 342.6 ms: f = -5.871277574299e-01, ‖∇f‖ = 4.1970e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  1.27 s: f = -6.001554858779e-01, ‖∇f‖ = 2.1792e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 334.8 ms: f = -6.068836018580e-01, ‖∇f‖ = 1.9566e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 330.7 ms: f = -6.250397688400e-01, ‖∇f‖ = 3.0330e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 251.4 ms: f = -6.391660376636e-01, ‖∇f‖ = 2.3075e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 267.4 ms: f = -6.471796195686e-01, ‖∇f‖ = 2.6051e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 226.5 ms: f = -6.503370025252e-01, ‖∇f‖ = 1.6112e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 254.5 ms: f = -6.546061095107e-01, ‖∇f‖ = 7.7752e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 266.9 ms: f = -6.559626479096e-01, ‖∇f‖ = 5.1323e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 267.1 ms: f = -6.570345925072e-01, ‖∇f‖ = 5.6663e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 260.7 ms: f = -6.586101542992e-01, ‖∇f‖ = 4.5249e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 273.4 ms: f = -6.594210783037e-01, ‖∇f‖ = 4.8908e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt 270.6 ms: f = -6.595829407862e-01, ‖∇f‖ = 5.7868e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, Δt 261.5 ms: f = -6.598106978631e-01, ‖∇f‖ = 1.7743e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, Δt 267.3 ms: f = -6.598737919822e-01, ‖∇f‖ = 1.4674e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, Δt 256.5 ms: f = -6.600722399901e-01, ‖∇f‖ = 1.9297e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, Δt 264.3 ms: f = -6.602319320307e-01, ‖∇f‖ = 1.7537e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, Δt 264.5 ms: f = -6.603792234693e-01, ‖∇f‖ = 2.3875e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, Δt 276.0 ms: f = -6.604618342679e-01, ‖∇f‖ = 2.3372e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, Δt  1.23 s: f = -6.605536670251e-01, ‖∇f‖ = 1.2672e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, Δt 267.2 ms: f = -6.606170020178e-01, ‖∇f‖ = 1.0507e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, Δt 298.9 ms: f = -6.608142109072e-01, ‖∇f‖ = 1.8082e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, Δt 328.6 ms: f = -6.609609278098e-01, ‖∇f‖ = 1.7516e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, Δt 249.0 ms: f = -6.610389074905e-01, ‖∇f‖ = 1.1313e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, Δt 266.7 ms: f = -6.610872582358e-01, ‖∇f‖ = 1.0263e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, Δt 243.8 ms: f = -6.611211815134e-01, ‖∇f‖ = 8.8770e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, Δt 273.6 ms: f = -6.611798467656e-01, ‖∇f‖ = 1.1488e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, Δt 589.4 ms: f = -6.612071948177e-01, ‖∇f‖ = 8.8666e-03, α = 5.31e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   35, Δt 312.8 ms: f = -6.612262742065e-01, ‖∇f‖ = 6.4670e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, Δt 309.1 ms: f = -6.612605505781e-01, ‖∇f‖ = 5.8761e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, Δt 296.5 ms: f = -6.612675206044e-01, ‖∇f‖ = 1.2105e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, Δt 311.6 ms: f = -6.612838274236e-01, ‖∇f‖ = 4.9173e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, Δt 297.0 ms: f = -6.612924065028e-01, ‖∇f‖ = 4.5918e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, Δt 302.3 ms: f = -6.613079391530e-01, ‖∇f‖ = 6.2780e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, Δt 309.7 ms: f = -6.613408205522e-01, ‖∇f‖ = 8.9067e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, Δt 310.3 ms: f = -6.614138147418e-01, ‖∇f‖ = 1.7166e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, Δt 340.2 ms: f = -6.614875520889e-01, ‖∇f‖ = 2.7427e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, Δt 373.1 ms: f = -6.616598996952e-01, ‖∇f‖ = 1.9056e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, Δt 399.3 ms: f = -6.618786556444e-01, ‖∇f‖ = 2.2180e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, Δt  1.39 s: f = -6.619400903666e-01, ‖∇f‖ = 2.5707e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, Δt 319.4 ms: f = -6.620475837462e-01, ‖∇f‖ = 1.8971e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, Δt 415.2 ms: f = -6.621064283010e-01, ‖∇f‖ = 3.7601e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   49, Δt 402.7 ms: f = -6.622530651429e-01, ‖∇f‖ = 1.3384e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, Δt 335.8 ms: f = -6.623266509915e-01, ‖∇f‖ = 1.1681e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, Δt 368.6 ms: f = -6.623900625629e-01, ‖∇f‖ = 1.0879e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, Δt 400.5 ms: f = -6.624417840065e-01, ‖∇f‖ = 1.3536e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, Δt 843.4 ms: f = -6.624657232439e-01, ‖∇f‖ = 8.4312e-03, α = 4.80e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   54, Δt 417.5 ms: f = -6.624781165452e-01, ‖∇f‖ = 5.0930e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, Δt 420.6 ms: f = -6.624880687095e-01, ‖∇f‖ = 5.2557e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, Δt 407.4 ms: f = -6.624977042182e-01, ‖∇f‖ = 3.7286e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, Δt 452.2 ms: f = -6.625008746110e-01, ‖∇f‖ = 3.0276e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, Δt 407.0 ms: f = -6.625030255855e-01, ‖∇f‖ = 2.4305e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, Δt 404.5 ms: f = -6.625070516495e-01, ‖∇f‖ = 2.1048e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, Δt 416.0 ms: f = -6.625095242229e-01, ‖∇f‖ = 2.1471e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, Δt 815.1 ms: f = -6.625106856424e-01, ‖∇f‖ = 1.7835e-03, α = 5.47e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   62, Δt 375.8 ms: f = -6.625114666348e-01, ‖∇f‖ = 9.0132e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, Δt 367.3 ms: f = -6.625118764288e-01, ‖∇f‖ = 8.0255e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, Δt  1.29 s: f = -6.625122687065e-01, ‖∇f‖ = 9.0615e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, Δt 274.7 ms: f = -6.625126290506e-01, ‖∇f‖ = 8.0460e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, Δt 337.3 ms: f = -6.625129211222e-01, ‖∇f‖ = 6.8501e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, Δt 360.9 ms: f = -6.625132419353e-01, ‖∇f‖ = 9.0265e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   68, Δt 310.6 ms: f = -6.625134633026e-01, ‖∇f‖ = 7.9160e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, Δt 311.3 ms: f = -6.625136591289e-01, ‖∇f‖ = 5.0828e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, Δt 324.8 ms: f = -6.625138308035e-01, ‖∇f‖ = 7.9904e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, Δt 658.4 ms: f = -6.625138950807e-01, ‖∇f‖ = 4.3784e-04, α = 5.13e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   72, Δt 330.0 ms: f = -6.625139324941e-01, ‖∇f‖ = 3.5191e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, Δt 377.8 ms: f = -6.625140365412e-01, ‖∇f‖ = 2.7327e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, Δt 674.0 ms: f = -6.625140701012e-01, ‖∇f‖ = 7.1588e-04, α = 4.32e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   75, Δt 346.5 ms: f = -6.625141329298e-01, ‖∇f‖ = 4.5959e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, Δt 343.8 ms: f = -6.625141933640e-01, ‖∇f‖ = 2.3876e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, Δt 343.9 ms: f = -6.625142295649e-01, ‖∇f‖ = 1.7163e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, Δt 342.5 ms: f = -6.625142520304e-01, ‖∇f‖ = 1.3279e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, Δt 349.1 ms: f = -6.625142665992e-01, ‖∇f‖ = 2.8569e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, Δt 329.9 ms: f = -6.625142803082e-01, ‖∇f‖ = 1.2759e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 81 iterations and time  6.94 m: f = -6.625142855970e-01, ‖∇f‖ = 7.7033e-05

````

Note that `fixedpoint` returns the final optimized PEPS, the last converged environment,
the final energy estimate as well as a `NamedTuple` of diagnostics. This allows us to, e.g.,
analyze the number of cost function calls or the history of gradient norms to evaluate
the convergence rate:

````julia
@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
````

````
info_opt.fg_evaluations = 97
info_opt.gradnorms[1:10:end] = [0.9354752017271177, 0.3033034076782405, 0.017742662111911257, 0.011312825498690943, 0.0062779819357426676, 0.011680613608174078, 0.0021471423320920044, 0.0007990427093226254, 0.0001275947945785803]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142855969753

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
ξ_h = [1.0343404931444404]
ξ_v = [1.0242351103541054]

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
LocalOperator{Tuple{Pair{Tuple{CartesianIndex{2}}, TensorKit.TensorMap{Float64, TensorKit.ComplexSpace, 1, 1, Vector{Float64}}}}, TensorKit.ComplexSpace}(TensorKit.ComplexSpace[ℂ^2;;], ((CartesianIndex(1, 1),) => TensorMap{Float64, TensorKit.ComplexSpace, 1, 1, Vector{Float64}}([1.0, 0.0, 0.0, -1.0], ℂ^2 ← ℂ^2),))
````

Finally, to evaluate the expecation value on the `LocalOperator`, we call:

````julia
@show expectation_value(peps, M, env);
````

````
expectation_value(peps, M, env) = -0.7532893440072446 - 4.85722573273506e-17im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

