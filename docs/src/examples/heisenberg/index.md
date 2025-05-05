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
H = heisenberg_XYZ(InfiniteSquare(); Jx=-1, Jy=1, Jz=-1)
````

````
LocalOperator{Tuple{Pair{Tuple{CartesianIndex{2}, CartesianIndex{2}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}, Pair{Tuple{CartesianIndex{2}, CartesianIndex{2}}, TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 2, 2, Vector{ComplexF64}}}}, TensorKit.ComplexSpace}(TensorKit.ComplexSpace[ℂ^2;;], ((CartesianIndex(1, 1), CartesianIndex(1, 2)) => TensorMap((ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2)):
[:, :, 1, 1] =
 -0.25 + 0.0im   0.0 + 0.0im
   0.0 + 0.0im  -0.5 + 0.0im

[:, :, 2, 1] =
  0.0 + 0.0im  0.0 + 0.0im
 0.25 + 0.0im  0.0 + 0.0im

[:, :, 1, 2] =
 0.0 + 0.0im  0.25 + 0.0im
 0.0 + 0.0im   0.0 + 0.0im

[:, :, 2, 2] =
 -0.5 + 0.0im    0.0 + 0.0im
  0.0 + 0.0im  -0.25 + 0.0im
, (CartesianIndex(1, 1), CartesianIndex(2, 1)) => TensorMap((ℂ^2 ⊗ ℂ^2) ← (ℂ^2 ⊗ ℂ^2)):
[:, :, 1, 1] =
 -0.25 + 0.0im   0.0 + 0.0im
   0.0 + 0.0im  -0.5 + 0.0im

[:, :, 2, 1] =
  0.0 + 0.0im  0.0 + 0.0im
 0.25 + 0.0im  0.0 + 0.0im

[:, :, 1, 2] =
 0.0 + 0.0im  0.25 + 0.0im
 0.0 + 0.0im   0.0 + 0.0im

[:, :, 2, 2] =
 -0.5 + 0.0im    0.0 + 0.0im
  0.0 + 0.0im  -0.25 + 0.0im
))
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
boundary_alg = (; tol=1e-10, trscheme=(; alg=:fixedspace));
````

Let us also configure the optimizer algorithm. We are going to optimize the PEPS using the
L-BFGS optimizer from [OptimKit](https://github.com/Jutho/OptimKit.jl). Again, we specify
the convergence tolerance (for the gradient norm) as well as the maximal number of iterations
and the BFGS memory size (which is used to approximate the Hessian):

````julia
optimizer_alg = (; alg=:lbfgs, tol=1e-4, maxiter=100, lbfgs_memory=16);
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
peps₀ = InfinitePEPS(randn, ComplexF64, 2, Dbond)
````

````
InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}(TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}[TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')):
[:, :, 1, 1, 1] =
 0.07382174258286094 + 0.12820373667088403im   0.7897519397510839 + 0.9113654266438473im
  0.2553716885006697 - 0.4358399804354269im   -1.0272416446076236 - 0.12635062198157215im

[:, :, 2, 1, 1] =
 0.16833628450178303 - 0.10088950122180829im  -0.9702030532300809 + 0.010730752411986726im
 -1.6804460553576506 + 0.29081053879369084im   0.6844811667615024 + 0.09101537356941222im

[:, :, 1, 2, 1] =
  0.5085938050744258 + 0.3786892551842583im   1.0020057959636561 - 1.4704891009758718im
 -0.6153328223084331 + 0.10417896606055738im  0.6024931811537675 - 1.0348374874397468im

[:, :, 2, 2, 1] =
 -0.027201695938305456 + 0.5778042099380925im  0.09232089635078945 + 0.6143070126937361im
    1.0707115218777772 - 0.5747168579241235im  -0.5819741818511422 - 0.9842624134267605im

[:, :, 1, 1, 2] =
 1.2332543810053822 - 1.7783531996396438im  0.8887723728085348 + 0.7809798723615474im
 1.2251189302516847 - 0.6853683793073324im  1.5333834584675397 - 0.13856216581406375im

[:, :, 2, 1, 2] =
 0.1406381347783769 + 0.6630243440357264im   -0.7294596235434386 + 0.40327909254711103im
 0.7212056487788236 + 0.24320971945037498im   0.9991347929322827 + 0.0017902515981375842im

[:, :, 1, 2, 2] =
 0.34282910982693904 - 0.4865238029567361im   0.9380949844871762 - 0.6985342237892025im
 -0.7437083517319159 - 0.6895708849529253im  -0.8981092940164176 + 0.9720706252141459im

[:, :, 2, 2, 2] =
 -0.8897079923413616 - 0.7145412189457411im  0.07771261045117502 - 0.6400190994609709im
 -1.6099412157243007 + 0.8855200965611144im   0.7357380595021633 + 0.4626916850143416im
;;])
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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201184615e-11	time = 0.17 sec

````

Besides the converged environment, `leading_boundary` also returns a `NamedTuple` of
informational quantities such as the last maximal truncation error - that is, the SVD
approximation error incurred in the last CTMRG iteration, maximized over all spatial
directions and unit cell entries:

````julia
@show info_ctmrg.truncation_error;
````

````
info_ctmrg.truncation_error = 0.0017032153529848298

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
[ Info: LBFGS: initializing with f = 0.000601645310, ‖∇f‖ = 9.3548e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time  643.37 s: f = -0.489783740840, ‖∇f‖ = 6.0020e-01, α = 5.94e+01, m = 0, nfg = 5
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time  645.02 s: f = -0.501971411096, ‖∇f‖ = 5.3738e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  645.36 s: f = -0.523152816264, ‖∇f‖ = 3.9922e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  646.10 s: f = -0.538652145758, ‖∇f‖ = 4.1551e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, time  648.50 s: f = -0.549861364689, ‖∇f‖ = 4.4015e-01, α = 6.94e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, time  649.35 s: f = -0.568951023367, ‖∇f‖ = 4.8339e-01, α = 2.24e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, time  649.72 s: f = -0.586980871663, ‖∇f‖ = 4.2463e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  650.06 s: f = -0.599970185661, ‖∇f‖ = 2.1955e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  650.38 s: f = -0.606725496115, ‖∇f‖ = 1.9384e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  650.70 s: f = -0.624986498009, ‖∇f‖ = 2.9776e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  651.00 s: f = -0.638747320059, ‖∇f‖ = 2.3382e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  651.32 s: f = -0.645577148853, ‖∇f‖ = 2.9937e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  651.60 s: f = -0.650891062410, ‖∇f‖ = 1.4746e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  651.90 s: f = -0.654569099868, ‖∇f‖ = 7.0690e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  652.20 s: f = -0.655949603239, ‖∇f‖ = 5.0977e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  652.51 s: f = -0.657146001976, ‖∇f‖ = 5.8056e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  652.83 s: f = -0.658558478454, ‖∇f‖ = 5.0388e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  653.13 s: f = -0.659302065828, ‖∇f‖ = 4.0776e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  653.42 s: f = -0.659633838354, ‖∇f‖ = 2.2380e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, time  653.72 s: f = -0.659776177694, ‖∇f‖ = 2.1511e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, time  654.01 s: f = -0.659916031911, ‖∇f‖ = 2.0498e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, time  654.30 s: f = -0.660181523751, ‖∇f‖ = 1.7235e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, time  654.62 s: f = -0.660350536401, ‖∇f‖ = 1.8928e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, time  654.91 s: f = -0.660447076769, ‖∇f‖ = 1.0330e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, time  655.21 s: f = -0.660521574522, ‖∇f‖ = 1.0448e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, time  655.52 s: f = -0.660656071716, ‖∇f‖ = 1.8768e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, time  655.85 s: f = -0.660756412995, ‖∇f‖ = 3.2183e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, time  656.18 s: f = -0.660925447420, ‖∇f‖ = 1.3371e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, time  656.60 s: f = -0.661000634324, ‖∇f‖ = 9.8866e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, time  657.36 s: f = -0.661046316490, ‖∇f‖ = 9.1513e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, time  657.70 s: f = -0.661128304094, ‖∇f‖ = 9.6895e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, time  658.08 s: f = -0.661169144566, ‖∇f‖ = 1.3492e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, time  658.37 s: f = -0.661204525845, ‖∇f‖ = 9.6996e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, time  658.65 s: f = -0.661224003573, ‖∇f‖ = 6.2892e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   35, time  658.94 s: f = -0.661247137140, ‖∇f‖ = 4.4514e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, time  659.24 s: f = -0.661266456453, ‖∇f‖ = 5.3015e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, time  659.53 s: f = -0.661280686254, ‖∇f‖ = 9.2298e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, time  659.83 s: f = -0.661298851672, ‖∇f‖ = 5.9013e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, time  660.12 s: f = -0.661320547122, ‖∇f‖ = 6.2443e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, time  660.41 s: f = -0.661344887326, ‖∇f‖ = 9.9129e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, time  660.71 s: f = -0.661398950542, ‖∇f‖ = 1.6285e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, time  661.03 s: f = -0.661483277766, ‖∇f‖ = 1.6233e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, time  661.34 s: f = -0.661583013010, ‖∇f‖ = 2.8186e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, time  661.66 s: f = -0.661670888522, ‖∇f‖ = 3.9725e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, time  661.98 s: f = -0.661865434012, ‖∇f‖ = 1.3200e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, time  662.29 s: f = -0.661977354471, ‖∇f‖ = 1.5881e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, time  662.60 s: f = -0.662102076782, ‖∇f‖ = 2.0290e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, time  663.24 s: f = -0.662190125548, ‖∇f‖ = 2.2873e-02, α = 4.61e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   49, time  663.55 s: f = -0.662306892721, ‖∇f‖ = 1.3813e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, time  663.87 s: f = -0.662376465537, ‖∇f‖ = 1.9902e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, time  664.17 s: f = -0.662419493776, ‖∇f‖ = 1.2249e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, time  664.48 s: f = -0.662439251412, ‖∇f‖ = 7.3806e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, time  664.79 s: f = -0.662463629284, ‖∇f‖ = 5.1806e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   54, time  665.10 s: f = -0.662484473404, ‖∇f‖ = 4.6461e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, time  665.71 s: f = -0.662490501784, ‖∇f‖ = 6.3694e-03, α = 4.07e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   56, time  666.01 s: f = -0.662497687998, ‖∇f‖ = 2.9285e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, time  666.32 s: f = -0.662500949854, ‖∇f‖ = 2.1234e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, time  666.62 s: f = -0.662503723196, ‖∇f‖ = 4.1203e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, time  666.93 s: f = -0.662505780051, ‖∇f‖ = 3.0872e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, time  667.22 s: f = -0.662507116565, ‖∇f‖ = 1.9618e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, time  667.52 s: f = -0.662509290310, ‖∇f‖ = 1.5747e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   62, time  667.82 s: f = -0.662510568937, ‖∇f‖ = 1.3099e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, time  668.65 s: f = -0.662511109974, ‖∇f‖ = 2.7217e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, time  668.96 s: f = -0.662511878793, ‖∇f‖ = 1.0320e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, time  669.34 s: f = -0.662512042147, ‖∇f‖ = 5.9753e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, time  669.65 s: f = -0.662512275118, ‖∇f‖ = 6.6602e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, time  669.93 s: f = -0.662512678161, ‖∇f‖ = 9.0498e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   68, time  670.21 s: f = -0.662513114911, ‖∇f‖ = 1.8006e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, time  670.51 s: f = -0.662513454844, ‖∇f‖ = 9.5988e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, time  670.80 s: f = -0.662513639773, ‖∇f‖ = 5.2576e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, time  671.09 s: f = -0.662513713403, ‖∇f‖ = 4.0696e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   72, time  671.39 s: f = -0.662513818843, ‖∇f‖ = 4.8084e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, time  671.70 s: f = -0.662513978848, ‖∇f‖ = 6.8463e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, time  672.29 s: f = -0.662514066816, ‖∇f‖ = 5.2125e-04, α = 5.38e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   75, time  672.57 s: f = -0.662514122809, ‖∇f‖ = 3.2924e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, time  672.86 s: f = -0.662514184291, ‖∇f‖ = 2.7038e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, time  673.17 s: f = -0.662514214654, ‖∇f‖ = 4.6682e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, time  673.46 s: f = -0.662514242510, ‖∇f‖ = 2.7698e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, time  673.74 s: f = -0.662514253309, ‖∇f‖ = 1.6244e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, time  674.02 s: f = -0.662514263613, ‖∇f‖ = 1.2004e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   81, time  674.30 s: f = -0.662514271751, ‖∇f‖ = 1.4760e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   82, time  674.58 s: f = -0.662514281056, ‖∇f‖ = 1.6558e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   83, time  674.87 s: f = -0.662514283704, ‖∇f‖ = 2.1824e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 84 iterations and time 675.15 s: f = -0.662514288424, ‖∇f‖ = 5.9513e-05

````

Note that `fixedpoint` returns the final optimized PEPS, the last converged environment,
the final energy estimate as well as a `NamedTuple` of diagnostics. This allows us to, e.g.,
analyze the number of cost function calls or the history of gradient norms to evaluate
the convergence rate:

````julia
@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
````

````
info_opt.fg_evaluations = 98
info_opt.gradnorms[1:10:end] = [0.9354758925982428, 0.2977564979129917, 0.021510752143210195, 0.009151302712640632, 0.009912940904896769, 0.019901533415930574, 0.0019617677308353, 0.0005257609002981095, 0.00012003941670625574]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142884244373

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
ξ_h = [1.034117934253177]
ξ_v = [1.0240816290840877]

````

## Computing observables

As a last thing, we want to see how we can compute expectation values of observables, given
the optimized PEPS and its CTMRG environment. To compute, e.g., the magnetization, we first
need to define the observable as a `TensorMap`:

````julia
σ_z = TensorMap([1.0 0.0; 0.0 -1.0], ℂ^2, ℂ^2)
````

````
TensorMap(ℂ^2 ← ℂ^2):
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
LocalOperator{Tuple{Pair{Tuple{CartesianIndex{2}}, TensorKit.TensorMap{Float64, TensorKit.ComplexSpace, 1, 1, Vector{Float64}}}}, TensorKit.ComplexSpace}(TensorKit.ComplexSpace[ℂ^2;;], ((CartesianIndex(1, 1),) => TensorMap(ℂ^2 ← ℂ^2):
 1.0   0.0
 0.0  -1.0
,))
````

Finally, to evaluate the expecation value on the `LocalOperator`, we call:

````julia
@show expectation_value(peps, M, env);
````

````
expectation_value(peps, M, env) = -0.7554296202981441 + 2.541461125039123e-16im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

