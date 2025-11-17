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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201795901e-11	time = 7.98 sec

````

Besides the converged environment, `leading_boundary` also returns a `NamedTuple` of
informational quantities such as the last maximal truncation error - that is, the SVD
approximation error incurred in the last CTMRG iteration, maximized over all spatial
directions and unit cell entries:

````julia
@show info_ctmrg.truncation_error;
````

````
info_ctmrg.truncation_error = 0.0008076332823860801

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
[ Info: LBFGS: initializing with f = 6.016453104343e-04, ‖∇f‖ = 9.3548e-01
[ Info: LBFGS: iter    1, Δt  4.52 s: f = -4.897965192207e-01, ‖∇f‖ = 6.0022e-01, α = 5.94e+01, m = 0, nfg = 5
[ Info: LBFGS: iter    2, Δt  1.55 s: f = -5.019846341173e-01, ‖∇f‖ = 5.3738e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, Δt 328.4 ms: f = -5.231639258984e-01, ‖∇f‖ = 3.9927e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 842.4 ms: f = -5.386543628560e-01, ‖∇f‖ = 4.1552e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: LBFGS: iter    5, Δt  2.49 s: f = -5.498211821771e-01, ‖∇f‖ = 4.4002e-01, α = 6.90e-02, m = 4, nfg = 4
[ Info: LBFGS: iter    6, Δt 850.0 ms: f = -5.690169518488e-01, ‖∇f‖ = 4.8450e-01, α = 2.26e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, Δt 325.7 ms: f = -5.871277303926e-01, ‖∇f‖ = 4.1970e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 319.8 ms: f = -6.001554483426e-01, ‖∇f‖ = 2.1792e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 305.8 ms: f = -6.068835686463e-01, ‖∇f‖ = 1.9566e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 276.9 ms: f = -6.250397748968e-01, ‖∇f‖ = 3.0330e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt 282.8 ms: f = -6.391659645908e-01, ‖∇f‖ = 2.3075e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt 392.4 ms: f = -6.471793388973e-01, ‖∇f‖ = 2.6051e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt 724.4 ms: f = -6.503370671505e-01, ‖∇f‖ = 1.6112e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 322.8 ms: f = -6.546061020085e-01, ‖∇f‖ = 7.7751e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt 237.3 ms: f = -6.559626422861e-01, ‖∇f‖ = 5.1323e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt 261.0 ms: f = -6.570346071528e-01, ‖∇f‖ = 5.6661e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 263.3 ms: f = -6.586101879490e-01, ‖∇f‖ = 4.5250e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 248.0 ms: f = -6.594209589994e-01, ‖∇f‖ = 4.8923e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt 263.4 ms: f = -6.595831316360e-01, ‖∇f‖ = 5.7833e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   20, Δt 268.8 ms: f = -6.598106808006e-01, ‖∇f‖ = 1.7741e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   21, Δt 256.8 ms: f = -6.598737940556e-01, ‖∇f‖ = 1.4673e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   22, Δt 264.8 ms: f = -6.600721854328e-01, ‖∇f‖ = 1.9298e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   23, Δt 254.1 ms: f = -6.602319074023e-01, ‖∇f‖ = 1.7538e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   24, Δt 273.3 ms: f = -6.603792430521e-01, ‖∇f‖ = 2.3872e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   25, Δt 263.2 ms: f = -6.604617775509e-01, ‖∇f‖ = 2.3385e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   26, Δt 258.8 ms: f = -6.605536696759e-01, ‖∇f‖ = 1.2672e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   27, Δt 257.1 ms: f = -6.606169735172e-01, ‖∇f‖ = 1.0506e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   28, Δt 285.6 ms: f = -6.608141517158e-01, ‖∇f‖ = 1.8075e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   29, Δt 305.3 ms: f = -6.609609491552e-01, ‖∇f‖ = 1.7508e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   30, Δt 796.4 ms: f = -6.610389180525e-01, ‖∇f‖ = 1.1325e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   31, Δt 328.4 ms: f = -6.610873060103e-01, ‖∇f‖ = 1.0269e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   32, Δt 254.5 ms: f = -6.611212029182e-01, ‖∇f‖ = 8.8756e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   33, Δt 281.7 ms: f = -6.611799456772e-01, ‖∇f‖ = 1.1435e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   34, Δt 553.2 ms: f = -6.612071690766e-01, ‖∇f‖ = 8.8858e-03, α = 5.30e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   35, Δt 285.6 ms: f = -6.612262681103e-01, ‖∇f‖ = 6.4795e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   36, Δt 295.0 ms: f = -6.612605080039e-01, ‖∇f‖ = 5.8879e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   37, Δt 285.7 ms: f = -6.612675983705e-01, ‖∇f‖ = 1.2007e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   38, Δt 278.1 ms: f = -6.612837466720e-01, ‖∇f‖ = 4.9256e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   39, Δt 275.6 ms: f = -6.612924155095e-01, ‖∇f‖ = 4.6090e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   40, Δt 273.9 ms: f = -6.613078556647e-01, ‖∇f‖ = 6.2724e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   41, Δt 282.3 ms: f = -6.613407946569e-01, ‖∇f‖ = 8.9187e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   42, Δt 293.4 ms: f = -6.614147324894e-01, ‖∇f‖ = 1.6921e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   43, Δt 310.8 ms: f = -6.614896605795e-01, ‖∇f‖ = 2.7701e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   44, Δt 289.7 ms: f = -6.616600115097e-01, ‖∇f‖ = 2.0288e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   45, Δt 377.8 ms: f = -6.618744649549e-01, ‖∇f‖ = 2.2589e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   46, Δt 739.3 ms: f = -6.619293873126e-01, ‖∇f‖ = 2.6087e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   47, Δt 320.0 ms: f = -6.620580680546e-01, ‖∇f‖ = 1.8561e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   48, Δt 607.9 ms: f = -6.621418094892e-01, ‖∇f‖ = 2.1694e-02, α = 5.08e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   49, Δt 299.7 ms: f = -6.622424140119e-01, ‖∇f‖ = 1.4693e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   50, Δt 317.6 ms: f = -6.623561643274e-01, ‖∇f‖ = 1.5643e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   51, Δt 325.3 ms: f = -6.624189189954e-01, ‖∇f‖ = 1.2526e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   52, Δt 322.8 ms: f = -6.624484647013e-01, ‖∇f‖ = 7.7875e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   53, Δt 302.8 ms: f = -6.624620825387e-01, ‖∇f‖ = 7.2801e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   54, Δt 297.0 ms: f = -6.624761506291e-01, ‖∇f‖ = 5.6893e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   55, Δt 302.8 ms: f = -6.624920797996e-01, ‖∇f‖ = 7.4763e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   56, Δt 618.9 ms: f = -6.624974230146e-01, ‖∇f‖ = 3.9289e-03, α = 5.46e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   57, Δt 288.3 ms: f = -6.624993982624e-01, ‖∇f‖ = 3.0506e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   58, Δt 284.2 ms: f = -6.625033254432e-01, ‖∇f‖ = 2.2061e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   59, Δt 300.0 ms: f = -6.625068780651e-01, ‖∇f‖ = 2.0862e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   60, Δt 812.0 ms: f = -6.625094238841e-01, ‖∇f‖ = 3.0782e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   61, Δt 314.7 ms: f = -6.625108313560e-01, ‖∇f‖ = 1.9160e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   62, Δt 248.5 ms: f = -6.625115537204e-01, ‖∇f‖ = 8.6226e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   63, Δt 261.4 ms: f = -6.625118952372e-01, ‖∇f‖ = 7.1198e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   64, Δt 273.4 ms: f = -6.625122637335e-01, ‖∇f‖ = 6.9776e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   65, Δt 263.1 ms: f = -6.625125878909e-01, ‖∇f‖ = 1.1757e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   66, Δt 278.5 ms: f = -6.625129012017e-01, ‖∇f‖ = 6.1076e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   67, Δt 275.1 ms: f = -6.625131620795e-01, ‖∇f‖ = 6.7509e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   68, Δt 287.4 ms: f = -6.625134659017e-01, ‖∇f‖ = 7.4076e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   69, Δt 585.2 ms: f = -6.625136173027e-01, ‖∇f‖ = 1.0650e-03, α = 4.63e-01, m = 16, nfg = 2
[ Info: LBFGS: iter   70, Δt 297.3 ms: f = -6.625138093900e-01, ‖∇f‖ = 5.3214e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   71, Δt 267.8 ms: f = -6.625138941594e-01, ‖∇f‖ = 4.9455e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   72, Δt 275.5 ms: f = -6.625139584721e-01, ‖∇f‖ = 4.6079e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   73, Δt 270.4 ms: f = -6.625140156102e-01, ‖∇f‖ = 3.5058e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   74, Δt 280.5 ms: f = -6.625140884579e-01, ‖∇f‖ = 7.7069e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   75, Δt 282.6 ms: f = -6.625141400523e-01, ‖∇f‖ = 6.5839e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   76, Δt 267.8 ms: f = -6.625141806040e-01, ‖∇f‖ = 3.1793e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   77, Δt 822.8 ms: f = -6.625142155106e-01, ‖∇f‖ = 2.0830e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   78, Δt 263.1 ms: f = -6.625142398464e-01, ‖∇f‖ = 1.8962e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   79, Δt 253.8 ms: f = -6.625142606414e-01, ‖∇f‖ = 2.6019e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   80, Δt 262.8 ms: f = -6.625142658376e-01, ‖∇f‖ = 2.4559e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: converged after 81 iterations and time  5.83 m: f = -6.625142736963e-01, ‖∇f‖ = 9.4632e-05

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
info_opt.gradnorms[1:10:end] = [0.9354752030131099, 0.3033019277899899, 0.017741463681835545, 0.01132472701425422, 0.006272379430806119, 0.015643014849711687, 0.003078240790192421, 0.000532137892512568, 0.0002455878974975024]

````

Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by
[Sandvik](@cite sandvik_computational_2011), where the energy per site was found to be
$E_{\text{ref}}=−0.6694421$. From our simple optimization we find:

````julia
@show E;
````

````
E = -0.6625142736962993

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
ξ_h = [1.0345801030391306]
ξ_v = [1.0245361358882241]

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
expectation_value(peps, M, env) = -0.7533445046064892 - 1.5265566588595902e-16im

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

