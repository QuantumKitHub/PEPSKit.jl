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
boundary_alg = (; tol = 1.0e-10, trscheme = (; alg = :fixedspace));
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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201215053e-11	time = 0.33 sec

````

Besides the converged environment, `leading_boundary` also returns a `NamedTuple` of
informational quantities such as the last maximal truncation error - that is, the SVD
approximation error incurred in the last CTMRG iteration, maximized over all spatial
directions and unit cell entries:

````julia
@show info_ctmrg.truncation_error;
````

````
info_ctmrg.truncation_error = 0.0008076332824218667

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
[ Info: LBFGS: iter    1, time  653.15 s: f = -0.489780515313, ‖∇f‖ = 6.0029e-01, α = 5.94e+01, m = 0, nfg = 5
[ Info: LBFGS: iter    2, time  654.16 s: f = -0.501969370077, ‖∇f‖ = 5.3739e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  654.65 s: f = -0.523150697048, ‖∇f‖ = 3.9920e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  655.75 s: f = -0.538654572541, ‖∇f‖ = 4.1550e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, time  658.50 s: f = -0.549895732386, ‖∇f‖ = 4.4023e-01, α = 6.96e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, time  659.76 s: f = -0.568903773686, ‖∇f‖ = 4.8251e-01, α = 2.23e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, time  660.31 s: f = -0.586868032201, ‖∇f‖ = 4.2837e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  660.82 s: f = -0.599838784692, ‖∇f‖ = 2.2069e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  661.31 s: f = -0.606610614248, ‖∇f‖ = 1.9251e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  662.59 s: f = -0.624864046584, ‖∇f‖ = 2.9515e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  663.09 s: f = -0.638375158459, ‖∇f‖ = 2.3675e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  663.68 s: f = -0.644407078574, ‖∇f‖ = 3.2337e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  664.08 s: f = -0.651446429820, ‖∇f‖ = 1.3169e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  664.50 s: f = -0.654528109037, ‖∇f‖ = 6.6176e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  664.95 s: f = -0.655971360877, ‖∇f‖ = 5.1875e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  665.38 s: f = -0.657229359806, ‖∇f‖ = 5.8978e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  665.83 s: f = -0.658531955909, ‖∇f‖ = 5.5554e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  666.25 s: f = -0.659295132779, ‖∇f‖ = 3.0496e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  666.67 s: f = -0.659541951057, ‖∇f‖ = 2.2298e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, time  667.08 s: f = -0.659737986345, ‖∇f‖ = 2.7588e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, time  667.50 s: f = -0.659907309276, ‖∇f‖ = 1.9371e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, time  667.92 s: f = -0.660097028509, ‖∇f‖ = 1.4424e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, time  668.36 s: f = -0.660261859113, ‖∇f‖ = 1.2401e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, time  668.81 s: f = -0.660393163797, ‖∇f‖ = 1.9193e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, time  669.24 s: f = -0.660497281892, ‖∇f‖ = 1.3339e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, time  669.67 s: f = -0.660573874658, ‖∇f‖ = 1.2488e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, time  670.11 s: f = -0.660741356363, ‖∇f‖ = 1.6202e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, time  670.59 s: f = -0.660904313975, ‖∇f‖ = 1.8663e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, time  671.07 s: f = -0.661016298846, ‖∇f‖ = 1.3962e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, time  671.57 s: f = -0.661073848989, ‖∇f‖ = 8.0058e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, time  672.01 s: f = -0.661115844316, ‖∇f‖ = 7.7806e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, time  672.47 s: f = -0.661170943776, ‖∇f‖ = 9.2021e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, time  672.96 s: f = -0.661189817452, ‖∇f‖ = 1.7374e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, time  673.41 s: f = -0.661228750949, ‖∇f‖ = 5.5231e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   35, time  673.87 s: f = -0.661241674868, ‖∇f‖ = 4.6578e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, time  674.33 s: f = -0.661255547431, ‖∇f‖ = 5.3447e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, time  674.77 s: f = -0.661267661951, ‖∇f‖ = 1.2055e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, time  675.20 s: f = -0.661283905499, ‖∇f‖ = 6.5966e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, time  675.64 s: f = -0.661292308568, ‖∇f‖ = 4.6474e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, time  676.08 s: f = -0.661305252841, ‖∇f‖ = 5.5812e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, time  676.53 s: f = -0.661333862066, ‖∇f‖ = 1.0254e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, time  677.01 s: f = -0.661388075788, ‖∇f‖ = 1.3577e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, time  677.49 s: f = -0.661466685819, ‖∇f‖ = 2.6791e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, time  677.96 s: f = -0.661613254953, ‖∇f‖ = 1.8756e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, time  678.45 s: f = -0.661816072567, ‖∇f‖ = 3.1851e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, time  678.94 s: f = -0.661914094913, ‖∇f‖ = 2.3406e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, time  679.41 s: f = -0.661987579533, ‖∇f‖ = 2.6289e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, time  679.90 s: f = -0.662122271528, ‖∇f‖ = 4.1730e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   49, time  681.13 s: f = -0.662298461024, ‖∇f‖ = 1.1879e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, time  681.68 s: f = -0.662344054505, ‖∇f‖ = 8.9729e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, time  682.28 s: f = -0.662406789067, ‖∇f‖ = 9.2057e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, time  682.77 s: f = -0.662448593102, ‖∇f‖ = 1.3971e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, time  683.24 s: f = -0.662470562406, ‖∇f‖ = 1.1117e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   54, time  683.71 s: f = -0.662486893225, ‖∇f‖ = 3.8704e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, time  684.18 s: f = -0.662492565560, ‖∇f‖ = 2.6533e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, time  684.65 s: f = -0.662499211249, ‖∇f‖ = 2.5921e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, time  685.16 s: f = -0.662501070713, ‖∇f‖ = 7.0067e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, time  685.65 s: f = -0.662505878725, ‖∇f‖ = 2.3803e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, time  686.11 s: f = -0.662507798937, ‖∇f‖ = 1.4097e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, time  686.57 s: f = -0.662509615613, ‖∇f‖ = 1.5457e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, time  687.04 s: f = -0.662510402676, ‖∇f‖ = 3.4157e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   62, time  687.51 s: f = -0.662511467817, ‖∇f‖ = 1.2986e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, time  687.95 s: f = -0.662511687943, ‖∇f‖ = 7.7089e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, time  688.39 s: f = -0.662511962285, ‖∇f‖ = 7.3725e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, time  688.82 s: f = -0.662512306742, ‖∇f‖ = 9.1907e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, time  689.33 s: f = -0.662512716968, ‖∇f‖ = 1.2363e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, time  689.79 s: f = -0.662513054490, ‖∇f‖ = 9.9891e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   68, time  690.25 s: f = -0.662513253128, ‖∇f‖ = 6.6281e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, time  690.70 s: f = -0.662513427524, ‖∇f‖ = 6.7118e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, time  691.17 s: f = -0.662513545690, ‖∇f‖ = 5.7281e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, time  691.62 s: f = -0.662513765491, ‖∇f‖ = 5.8790e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   72, time  692.07 s: f = -0.662513833730, ‖∇f‖ = 8.6229e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, time  692.52 s: f = -0.662513921748, ‖∇f‖ = 5.0238e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, time  692.97 s: f = -0.662514033473, ‖∇f‖ = 3.8872e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   75, time  693.46 s: f = -0.662514109735, ‖∇f‖ = 4.1002e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, time  693.92 s: f = -0.662514164938, ‖∇f‖ = 6.5939e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, time  694.37 s: f = -0.662514209083, ‖∇f‖ = 4.7716e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, time  694.81 s: f = -0.662514236258, ‖∇f‖ = 1.4391e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, time  695.24 s: f = -0.662514243007, ‖∇f‖ = 1.0857e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, time  695.67 s: f = -0.662514252524, ‖∇f‖ = 1.5648e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   81, time  696.10 s: f = -0.662514257885, ‖∇f‖ = 2.6735e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   82, time  696.53 s: f = -0.662514266008, ‖∇f‖ = 1.3673e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   83, time  696.96 s: f = -0.662514267342, ‖∇f‖ = 1.5962e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   84, time  697.39 s: f = -0.662514272294, ‖∇f‖ = 1.0461e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   85, time  697.82 s: f = -0.662514280046, ‖∇f‖ = 1.2844e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 86 iterations and time 698.25 s: f = -0.662514280578, ‖∇f‖ = 8.2691e-05

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
info_opt.gradnorms[1:10:end] = [0.9354698847828768, 0.2951509773910037, 0.027588479373987673, 0.008005782175537665, 0.005581239160769203, 0.008972947287595543, 0.0015457251607028555, 0.000572806562953061, 0.00015648235851587362]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142805775173

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
ξ_h = [1.0345313008323207]
ξ_v = [1.0242824078560868]

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
expectation_value(peps, M, env) = -0.7550965992105596 + 1.717376241217039e-16im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

