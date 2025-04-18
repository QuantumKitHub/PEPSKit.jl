```@meta
EditURL = "../../../../examples/1.heisenberg/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//1.heisenberg/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//1.heisenberg/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//1.heisenberg)


# [Optimizing the 2D Heisenberg model](@id examples_heisenberg)

In this example we want to provide a basic rundown of PEPSKit's optimization workflow for
PEPS. To that end, we will consider the two-dimensional Heisenberg model on a square lattice

```math
H = \sum_{\langle i,j \rangle} J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j
```

Here, we want to set $J_x=J_y=J_z=1$ where the Heisenberg model is in the antiferromagnetic
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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201184615e-11	time = 0.30 sec

````

Besides the converged environment, `leading_boundary` also returns a `NamedTuple` of
informational quantities such as the last (maximal) SVD truncation error:

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
[ Info: LBFGS: initializing with f = 0.000601645310, ‖∇f‖ = 9.3547e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time    4.15 s: f = -0.489796540851, ‖∇f‖ = 6.0022e-01, α = 5.94e+01, m = 0, nfg = 5
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time    5.37 s: f = -0.501984649868, ‖∇f‖ = 5.3739e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time    5.99 s: f = -0.523163971924, ‖∇f‖ = 3.9927e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time    7.32 s: f = -0.538654390178, ‖∇f‖ = 4.1552e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, time   10.35 s: f = -0.549821445064, ‖∇f‖ = 4.4002e-01, α = 6.90e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, time   11.80 s: f = -0.569016778155, ‖∇f‖ = 4.8450e-01, α = 2.26e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, time   12.46 s: f = -0.587127261652, ‖∇f‖ = 4.1972e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time   13.09 s: f = -0.600154758006, ‖∇f‖ = 2.1793e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time   13.71 s: f = -0.606883012825, ‖∇f‖ = 1.9566e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time   14.33 s: f = -0.625040022199, ‖∇f‖ = 3.0328e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time   14.94 s: f = -0.639164743235, ‖∇f‖ = 2.3076e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time   15.55 s: f = -0.647174335216, ‖∇f‖ = 2.6065e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time   16.21 s: f = -0.650338609163, ‖∇f‖ = 1.6108e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time   16.93 s: f = -0.654606007953, ‖∇f‖ = 7.7724e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time   17.53 s: f = -0.655962567656, ‖∇f‖ = 5.1320e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time   18.14 s: f = -0.657034966533, ‖∇f‖ = 5.6668e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time   18.77 s: f = -0.658609918816, ‖∇f‖ = 4.5267e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time   19.36 s: f = -0.659421361772, ‖∇f‖ = 4.8752e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time   19.97 s: f = -0.659584257676, ‖∇f‖ = 5.7745e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, time   20.56 s: f = -0.659811195031, ‖∇f‖ = 1.7740e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, time   21.16 s: f = -0.659874427409, ‖∇f‖ = 1.4673e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, time   22.38 s: f = -0.660072570659, ‖∇f‖ = 1.9320e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, time   22.95 s: f = -0.660232141902, ‖∇f‖ = 1.7545e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, time   23.56 s: f = -0.660380080163, ‖∇f‖ = 2.3752e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, time   24.15 s: f = -0.660461052221, ‖∇f‖ = 2.3596e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, time   24.74 s: f = -0.660554016679, ‖∇f‖ = 1.2681e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, time   25.33 s: f = -0.660617092333, ‖∇f‖ = 1.0485e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, time   25.95 s: f = -0.660813477825, ‖∇f‖ = 1.7986e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, time   26.58 s: f = -0.660960969686, ‖∇f‖ = 1.7471e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, time   27.22 s: f = -0.661039077160, ‖∇f‖ = 1.1401e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, time   27.85 s: f = -0.661087806652, ‖∇f‖ = 1.0339e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, time   28.46 s: f = -0.661121452359, ‖∇f‖ = 8.8764e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, time   29.08 s: f = -0.661180968072, ‖∇f‖ = 1.0798e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, time   30.34 s: f = -0.661206863491, ‖∇f‖ = 9.1298e-03, α = 5.16e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   35, time   30.96 s: f = -0.661226335408, ‖∇f‖ = 6.6169e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, time   31.60 s: f = -0.661260012209, ‖∇f‖ = 5.9848e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, time   32.21 s: f = -0.661268668988, ‖∇f‖ = 1.0826e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, time   32.81 s: f = -0.661283178602, ‖∇f‖ = 5.0739e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, time   33.42 s: f = -0.661293239500, ‖∇f‖ = 4.8729e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, time   34.04 s: f = -0.661307958912, ‖∇f‖ = 6.2349e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, time   34.66 s: f = -0.661342970541, ‖∇f‖ = 9.2762e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, time   35.28 s: f = -0.661417237192, ‖∇f‖ = 1.7461e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, time   35.93 s: f = -0.661494994773, ‖∇f‖ = 2.7924e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, time   36.56 s: f = -0.661665032445, ‖∇f‖ = 2.1193e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, time   37.20 s: f = -0.661840699875, ‖∇f‖ = 2.3535e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, time   37.83 s: f = -0.661983211854, ‖∇f‖ = 2.0945e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, time   38.48 s: f = -0.662069016591, ‖∇f‖ = 1.9084e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, time   39.11 s: f = -0.662233515834, ‖∇f‖ = 1.7527e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   49, time   40.41 s: f = -0.662332106545, ‖∇f‖ = 1.6912e-02, α = 5.23e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   50, time   41.05 s: f = -0.662397705763, ‖∇f‖ = 1.0361e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, time   41.68 s: f = -0.662434512495, ‖∇f‖ = 8.3812e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, time   42.32 s: f = -0.662459914679, ‖∇f‖ = 5.8461e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, time   42.97 s: f = -0.662475090799, ‖∇f‖ = 1.1955e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   54, time   43.61 s: f = -0.662490551714, ‖∇f‖ = 4.3803e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, time   44.25 s: f = -0.662494796646, ‖∇f‖ = 3.0803e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, time   44.89 s: f = -0.662500144159, ‖∇f‖ = 3.3935e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, time   45.54 s: f = -0.662501922158, ‖∇f‖ = 5.3636e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, time   46.17 s: f = -0.662504711089, ‖∇f‖ = 2.3876e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, time   46.82 s: f = -0.662506778078, ‖∇f‖ = 1.8200e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, time   47.45 s: f = -0.662508721362, ‖∇f‖ = 1.9946e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, time   48.08 s: f = -0.662510450644, ‖∇f‖ = 3.2749e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   62, time   48.70 s: f = -0.662510853517, ‖∇f‖ = 2.9884e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, time   49.32 s: f = -0.662511685410, ‖∇f‖ = 8.0551e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, time   49.94 s: f = -0.662511841022, ‖∇f‖ = 7.3596e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, time   50.56 s: f = -0.662512305957, ‖∇f‖ = 8.4091e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, time   51.18 s: f = -0.662512680347, ‖∇f‖ = 7.3318e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, time   52.41 s: f = -0.662512901164, ‖∇f‖ = 1.2648e-03, α = 4.44e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   68, time   53.03 s: f = -0.662513239918, ‖∇f‖ = 5.7061e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, time   53.65 s: f = -0.662513444109, ‖∇f‖ = 6.0192e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, time   54.28 s: f = -0.662513692106, ‖∇f‖ = 7.2056e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, time   54.91 s: f = -0.662513794204, ‖∇f‖ = 1.0926e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   72, time   55.53 s: f = -0.662513946704, ‖∇f‖ = 4.1373e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, time   56.15 s: f = -0.662514020708, ‖∇f‖ = 3.2774e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, time   56.77 s: f = -0.662514074542, ‖∇f‖ = 4.1795e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   75, time   57.40 s: f = -0.662514105503, ‖∇f‖ = 9.0816e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, time   58.03 s: f = -0.662514180025, ‖∇f‖ = 2.8220e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, time   58.66 s: f = -0.662514206339, ‖∇f‖ = 2.0931e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, time   59.27 s: f = -0.662514231519, ‖∇f‖ = 2.6637e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, time   60.51 s: f = -0.662514243124, ‖∇f‖ = 2.4156e-04, α = 5.50e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   80, time   61.13 s: f = -0.662514255964, ‖∇f‖ = 1.4589e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   81, time   61.73 s: f = -0.662514264597, ‖∇f‖ = 1.6299e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   82, time   62.32 s: f = -0.662514270687, ‖∇f‖ = 2.0842e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   83, time   62.94 s: f = -0.662514276005, ‖∇f‖ = 1.1227e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   84, time   63.53 s: f = -0.662514281099, ‖∇f‖ = 1.0747e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   85, time   64.13 s: f = -0.662514284644, ‖∇f‖ = 1.0173e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 86 iterations and time 64.73 s: f = -0.662514291826, ‖∇f‖ = 8.8301e-05

````

Note that `fixedpoint` returns the final optimized PEPS, the last converged environment,
the final energy estimate as well as a `NamedTuple` of diagnostics. This allows us to, e.g.,
analyze the number of cost function calls or the history of gradient norms to evaluate
the convergence rate:

````julia
@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
````

````
info_opt.fg_evaluations = 101
info_opt.gradnorms[1:10:end] = [0.9354746780119341, 0.30327773685601944, 0.01774018270739478, 0.011400758285761247, 0.006234905315448026, 0.010361182129783082, 0.0019945520233835886, 0.0007205561049739842, 0.00014589086981901128]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142918261725

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
ξ_h = [1.0343456168037697]
ξ_v = [1.0242117798450008]

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
expectation_value(peps, M, env) = -0.7533587094249098 - 3.448382721363169e-16im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

