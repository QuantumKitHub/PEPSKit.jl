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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201331116e-11	time = 0.11 sec

````

Besides the converged environment, `leading_boundary` also returns a `NamedTuple` of
informational quantities such as the last maximal truncation error - that is, the SVD
approximation error incurred in the last CTMRG iteration, maximized over all spatial
directions and unit cell entries:

````julia
@show info_ctmrg.truncation_error;
````

````
info_ctmrg.truncation_error = 0.0008076332824218652

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
[ Info: LBFGS: iter    1, time  530.06 s: f = -0.489780515318, ‖∇f‖ = 6.0029e-01, α = 5.94e+01, m = 0, nfg = 5
[ Info: LBFGS: iter    2, time  530.54 s: f = -0.501969370083, ‖∇f‖ = 5.3739e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time  530.74 s: f = -0.523150697049, ‖∇f‖ = 3.9920e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time  531.20 s: f = -0.538654572532, ‖∇f‖ = 4.1550e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, time  532.38 s: f = -0.549895732330, ‖∇f‖ = 4.4023e-01, α = 6.96e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, time  532.92 s: f = -0.568903773751, ‖∇f‖ = 4.8251e-01, α = 2.23e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, time  533.16 s: f = -0.586868032368, ‖∇f‖ = 4.2837e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, time  533.38 s: f = -0.599838784884, ‖∇f‖ = 2.2069e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time  533.60 s: f = -0.606610614420, ‖∇f‖ = 1.9251e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time  533.84 s: f = -0.624864046816, ‖∇f‖ = 2.9515e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time  534.08 s: f = -0.638375159059, ‖∇f‖ = 2.3675e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time  534.31 s: f = -0.644407080181, ‖∇f‖ = 3.2337e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time  534.50 s: f = -0.651446429016, ‖∇f‖ = 1.3169e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time  534.72 s: f = -0.654528109114, ‖∇f‖ = 6.6176e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time  534.91 s: f = -0.655971360805, ‖∇f‖ = 5.1875e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time  535.10 s: f = -0.657229359697, ‖∇f‖ = 5.8978e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time  535.32 s: f = -0.658531955935, ‖∇f‖ = 5.5554e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time  535.52 s: f = -0.659295132854, ‖∇f‖ = 3.0496e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  536.06 s: f = -0.659541951176, ‖∇f‖ = 2.2298e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, time  536.20 s: f = -0.659737986411, ‖∇f‖ = 2.7588e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, time  536.36 s: f = -0.659907309305, ‖∇f‖ = 1.9371e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, time  536.54 s: f = -0.660097028826, ‖∇f‖ = 1.4424e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, time  536.71 s: f = -0.660261859699, ‖∇f‖ = 1.2401e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, time  536.87 s: f = -0.660393163588, ‖∇f‖ = 1.9193e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, time  537.07 s: f = -0.660497281869, ‖∇f‖ = 1.3339e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, time  537.25 s: f = -0.660573874568, ‖∇f‖ = 1.2488e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, time  537.44 s: f = -0.660741356378, ‖∇f‖ = 1.6202e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, time  537.65 s: f = -0.660904313170, ‖∇f‖ = 1.8663e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, time  537.86 s: f = -0.661016299819, ‖∇f‖ = 1.3961e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, time  538.08 s: f = -0.661073848395, ‖∇f‖ = 8.0058e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, time  538.31 s: f = -0.661115845926, ‖∇f‖ = 7.7807e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, time  538.56 s: f = -0.661170946003, ‖∇f‖ = 9.2025e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, time  538.80 s: f = -0.661189817521, ‖∇f‖ = 1.7374e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, time  539.03 s: f = -0.661228749942, ‖∇f‖ = 5.5230e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   35, time  539.24 s: f = -0.661241674095, ‖∇f‖ = 4.6578e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, time  539.47 s: f = -0.661255546757, ‖∇f‖ = 5.3449e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, time  539.65 s: f = -0.661267660051, ‖∇f‖ = 1.2056e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, time  539.85 s: f = -0.661283904430, ‖∇f‖ = 6.5960e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, time  540.02 s: f = -0.661292307599, ‖∇f‖ = 4.6473e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, time  540.23 s: f = -0.661305251122, ‖∇f‖ = 5.5816e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, time  540.44 s: f = -0.661333861848, ‖∇f‖ = 1.0256e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, time  540.63 s: f = -0.661388067313, ‖∇f‖ = 1.3577e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, time  540.83 s: f = -0.661466697789, ‖∇f‖ = 2.6785e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, time  541.03 s: f = -0.661613249837, ‖∇f‖ = 1.8758e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, time  541.23 s: f = -0.661815950052, ‖∇f‖ = 3.1866e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, time  541.44 s: f = -0.661914266112, ‖∇f‖ = 2.3377e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, time  541.66 s: f = -0.661987630419, ‖∇f‖ = 2.6251e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, time  541.89 s: f = -0.662122898853, ‖∇f‖ = 4.1624e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   49, time  542.13 s: f = -0.662298584531, ‖∇f‖ = 1.1873e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, time  542.32 s: f = -0.662344209408, ‖∇f‖ = 8.9765e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, time  542.53 s: f = -0.662406985361, ‖∇f‖ = 9.1973e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, time  542.74 s: f = -0.662448422288, ‖∇f‖ = 1.4069e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, time  542.95 s: f = -0.662470844143, ‖∇f‖ = 1.1027e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   54, time  543.18 s: f = -0.662486913078, ‖∇f‖ = 3.8679e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, time  543.81 s: f = -0.662492606250, ‖∇f‖ = 2.6466e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, time  543.99 s: f = -0.662499228449, ‖∇f‖ = 2.5451e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   57, time  544.22 s: f = -0.662501449522, ‖∇f‖ = 6.7506e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, time  544.38 s: f = -0.662506093367, ‖∇f‖ = 2.2951e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, time  544.54 s: f = -0.662508017653, ‖∇f‖ = 1.3999e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, time  544.69 s: f = -0.662509746635, ‖∇f‖ = 1.4558e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, time  544.86 s: f = -0.662510852584, ‖∇f‖ = 2.9108e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   62, time  545.02 s: f = -0.662511459247, ‖∇f‖ = 2.2460e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, time  545.19 s: f = -0.662511826357, ‖∇f‖ = 7.7832e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, time  545.35 s: f = -0.662511947523, ‖∇f‖ = 6.8985e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, time  545.52 s: f = -0.662512259234, ‖∇f‖ = 9.7159e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, time  545.70 s: f = -0.662512598991, ‖∇f‖ = 9.8695e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, time  546.08 s: f = -0.662512857448, ‖∇f‖ = 1.7013e-03, α = 5.19e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   68, time  546.27 s: f = -0.662513219985, ‖∇f‖ = 6.8818e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, time  546.46 s: f = -0.662513359115, ‖∇f‖ = 7.6621e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   70, time  546.65 s: f = -0.662513515960, ‖∇f‖ = 6.8437e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, time  546.84 s: f = -0.662513618580, ‖∇f‖ = 1.1894e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   72, time  547.02 s: f = -0.662513786456, ‖∇f‖ = 4.9346e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, time  547.18 s: f = -0.662513847396, ‖∇f‖ = 4.4509e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, time  547.34 s: f = -0.662513979836, ‖∇f‖ = 4.8820e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   75, time  547.51 s: f = -0.662514103588, ‖∇f‖ = 4.8320e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, time  547.69 s: f = -0.662514142873, ‖∇f‖ = 7.4691e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, time  547.90 s: f = -0.662514224367, ‖∇f‖ = 2.1563e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, time  548.10 s: f = -0.662514238736, ‖∇f‖ = 1.6048e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, time  548.30 s: f = -0.662514255026, ‖∇f‖ = 1.8486e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, time  548.51 s: f = -0.662514264792, ‖∇f‖ = 1.7897e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   81, time  548.70 s: f = -0.662514271376, ‖∇f‖ = 1.2336e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   82, time  548.87 s: f = -0.662514276574, ‖∇f‖ = 1.1665e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 83 iterations and time 549.04 s: f = -0.662514280757, ‖∇f‖ = 9.1781e-05

````

Note that `fixedpoint` returns the final optimized PEPS, the last converged environment,
the final energy estimate as well as a `NamedTuple` of diagnostics. This allows us to, e.g.,
analyze the number of cost function calls or the history of gradient norms to evaluate
the convergence rate:

````julia
@show info_opt.fg_evaluations info_opt.gradnorms[1:10:end];
````

````
info_opt.fg_evaluations = 95
info_opt.gradnorms[1:10:end] = [0.9354698847828358, 0.29515098058470357, 0.02758843397008457, 0.0080057855195194, 0.005581593433505604, 0.008976519236226013, 0.0014558090494201758, 0.0006843650270592786, 0.00017896547683177183]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142807571207

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
ξ_h = [1.0343595729516264]
ξ_v = [1.0242394972193403]

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
expectation_value(peps, M, env) = -0.7550757719796017 - 6.317220878853466e-16im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

