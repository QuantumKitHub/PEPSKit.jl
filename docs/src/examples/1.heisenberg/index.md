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
[ Info: CTMRG conv 27:	obj = +9.727103564786e+00	err = 2.6201184615e-11	time = 6.82 sec

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
[ Info: CTMRG init:	obj = +2.458081979447e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +2.458081979447e-01	err = 1.0702487618e-11	time = 0.01 sec
[ Info: CTMRG   2:	obj = +2.458081979447e-01	err = 3.6204359828e-12	time = 0.01 sec
[ Info: CTMRG   3:	obj = +2.458081979447e-01	err = 1.0761692541e-12	time = 0.01 sec
[ Info: CTMRG conv 4:	obj = +2.458081979447e-01	err = 3.6674259117e-13	time = 0.02 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.88e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 2.06e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.68e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 1.95e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.15e-03
[ Info: BiCGStab linsolve in iteration 3: normres = 3.44e-04
[ Info: BiCGStab linsolve in iteration 3.5: normres = 1.06e-04
[ Info: BiCGStab linsolve in iteration 4: normres = 2.94e-05
[ Info: BiCGStab linsolve in iteration 4.5: normres = 3.91e-06
[ Info: BiCGStab linsolve in iteration 5: normres = 1.17e-06
[ Info: BiCGStab linsolve in iteration 5.5: normres = 1.19e-06
┌ Info: BiCGStab linsolve converged at iteration 6:
│ * norm of residual = 3.55e-07
└ * number of operations = 14
[ Info: LBFGS: initializing with f = 0.000601645310, ‖∇f‖ = 9.3547e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: CTMRG init:	obj = +2.466747864702e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +2.467202300719e-01	err = 5.1468672546e-03	time = 0.01 sec
[ Info: CTMRG   2:	obj = +2.467177582954e-01	err = 1.0248503896e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +2.467174847646e-01	err = 7.0606560014e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +2.467175566543e-01	err = 1.7544501818e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +2.467175535135e-01	err = 7.6336938372e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +2.467175523269e-01	err = 3.2948274479e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +2.467175525681e-01	err = 6.8853494985e-05	time = 0.02 sec
[ Info: CTMRG   8:	obj = +2.467175526511e-01	err = 3.5139804357e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +2.467175526675e-01	err = 1.3695745603e-05	time = 0.01 sec
[ Info: CTMRG  10:	obj = +2.467175526653e-01	err = 3.0592132984e-06	time = 0.01 sec
[ Info: CTMRG  11:	obj = +2.467175526626e-01	err = 9.6314251491e-07	time = 0.01 sec
[ Info: CTMRG  12:	obj = +2.467175526615e-01	err = 4.7338928574e-07	time = 0.01 sec
[ Info: CTMRG  13:	obj = +2.467175526615e-01	err = 1.4856431007e-07	time = 0.01 sec
[ Info: CTMRG  14:	obj = +2.467175526616e-01	err = 3.2966310155e-08	time = 0.01 sec
[ Info: CTMRG  15:	obj = +2.467175526617e-01	err = 1.7362459887e-08	time = 0.01 sec
[ Info: CTMRG  16:	obj = +2.467175526617e-01	err = 6.3743957795e-09	time = 0.01 sec
[ Info: CTMRG  17:	obj = +2.467175526617e-01	err = 1.4125953209e-09	time = 0.01 sec
[ Info: CTMRG  18:	obj = +2.467175526617e-01	err = 5.2856033100e-10	time = 0.01 sec
[ Info: CTMRG  19:	obj = +2.467175526617e-01	err = 2.6594624172e-10	time = 0.01 sec
[ Info: CTMRG conv 20:	obj = +2.467175526617e-01	err = 6.9252688655e-11	time = 0.12 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.99e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 2.23e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.96e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 2.33e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.63e-03
[ Info: BiCGStab linsolve in iteration 3: normres = 4.84e-04
[ Info: BiCGStab linsolve in iteration 3.5: normres = 1.54e-04
[ Info: BiCGStab linsolve in iteration 4: normres = 4.61e-05
[ Info: BiCGStab linsolve in iteration 4.5: normres = 3.99e-06
[ Info: BiCGStab linsolve in iteration 5: normres = 1.06e-06
┌ Info: BiCGStab linsolve converged at iteration 5.5:
│ * norm of residual = 4.99e-07
└ * number of operations = 13
[ Info: CTMRG init:	obj = +2.504989231566e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +2.518577636537e-01	err = 2.2195315784e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +2.519084970968e-01	err = 3.5708295274e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +2.518988455424e-01	err = 3.1268882475e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +2.519002749932e-01	err = 1.3358078973e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +2.519007227276e-01	err = 4.9297438688e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +2.519007313294e-01	err = 2.0293072275e-03	time = 0.01 sec
[ Info: CTMRG   7:	obj = +2.519007274334e-01	err = 5.6475980506e-04	time = 0.01 sec
[ Info: CTMRG   8:	obj = +2.519007280100e-01	err = 1.5819533038e-04	time = 0.01 sec
[ Info: CTMRG   9:	obj = +2.519007283638e-01	err = 9.2072757112e-05	time = 0.01 sec
[ Info: CTMRG  10:	obj = +2.519007284107e-01	err = 4.0322529795e-05	time = 0.01 sec
[ Info: CTMRG  11:	obj = +2.519007284073e-01	err = 1.1668112409e-05	time = 0.01 sec
[ Info: CTMRG  12:	obj = +2.519007283947e-01	err = 3.0908008114e-06	time = 0.01 sec
[ Info: CTMRG  13:	obj = +2.519007283902e-01	err = 1.2771504159e-06	time = 0.01 sec
[ Info: CTMRG  14:	obj = +2.519007283886e-01	err = 5.2562459654e-07	time = 0.01 sec
[ Info: CTMRG  15:	obj = +2.519007283886e-01	err = 1.9526028301e-07	time = 0.01 sec
[ Info: CTMRG  16:	obj = +2.519007283887e-01	err = 4.9184396270e-08	time = 0.01 sec
[ Info: CTMRG  17:	obj = +2.519007283888e-01	err = 2.7371205699e-08	time = 0.01 sec
[ Info: CTMRG  18:	obj = +2.519007283888e-01	err = 1.0841608326e-08	time = 0.01 sec
[ Info: CTMRG  19:	obj = +2.519007283888e-01	err = 3.4538640055e-09	time = 0.01 sec
[ Info: CTMRG  20:	obj = +2.519007283888e-01	err = 1.0564151881e-09	time = 0.01 sec
[ Info: CTMRG  21:	obj = +2.519007283888e-01	err = 4.6104335122e-10	time = 0.01 sec
[ Info: CTMRG  22:	obj = +2.519007283888e-01	err = 1.8122934616e-10	time = 0.01 sec
[ Info: CTMRG conv 23:	obj = +2.519007283888e-01	err = 7.0319174446e-11	time = 0.13 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.83e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 4.09e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.84e-02
[ Info: BiCGStab linsolve in iteration 2: normres = 6.20e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 6.03e-03
[ Info: BiCGStab linsolve in iteration 3: normres = 1.67e-03
[ Info: BiCGStab linsolve in iteration 3.5: normres = 7.47e-04
[ Info: BiCGStab linsolve in iteration 4: normres = 1.80e-04
[ Info: BiCGStab linsolve in iteration 4.5: normres = 1.21e-04
[ Info: BiCGStab linsolve in iteration 5: normres = 5.27e-05
[ Info: BiCGStab linsolve in iteration 5.5: normres = 3.66e-06
[ Info: BiCGStab linsolve in iteration 6: normres = 1.22e-06
┌ Info: BiCGStab linsolve converged at iteration 6.5:
│ * norm of residual = 6.74e-08
└ * number of operations = 15
[ Info: CTMRG init:	obj = +2.773142477636e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +3.280961996908e-01	err = 1.1119405070e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +3.408862747463e-01	err = 1.1817777907e-01	time = 0.01 sec
[ Info: CTMRG   3:	obj = +3.422468241365e-01	err = 8.4221966888e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +3.423550758606e-01	err = 5.4661495617e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +3.423637658895e-01	err = 2.0305722305e-02	time = 0.01 sec
[ Info: CTMRG   6:	obj = +3.423647903142e-01	err = 5.0566947094e-03	time = 0.01 sec
[ Info: CTMRG   7:	obj = +3.423649197863e-01	err = 2.1393494537e-03	time = 0.01 sec
[ Info: CTMRG   8:	obj = +3.423649288336e-01	err = 7.1326038559e-04	time = 0.01 sec
[ Info: CTMRG   9:	obj = +3.423649290144e-01	err = 1.8057996916e-04	time = 0.01 sec
[ Info: CTMRG  10:	obj = +3.423649290717e-01	err = 9.0089521941e-05	time = 0.00 sec
[ Info: CTMRG  11:	obj = +3.423649290982e-01	err = 2.3300941438e-05	time = 0.01 sec
[ Info: CTMRG  12:	obj = +3.423649291002e-01	err = 7.8791728239e-06	time = 0.01 sec
[ Info: CTMRG  13:	obj = +3.423649290993e-01	err = 3.2204397272e-06	time = 0.00 sec
[ Info: CTMRG  14:	obj = +3.423649290996e-01	err = 8.9775415590e-07	time = 0.00 sec
[ Info: CTMRG  15:	obj = +3.423649290995e-01	err = 5.3793169894e-07	time = 0.01 sec
[ Info: CTMRG  16:	obj = +3.423649290995e-01	err = 1.5592179650e-07	time = 0.01 sec
[ Info: CTMRG  17:	obj = +3.423649290995e-01	err = 3.8856151006e-08	time = 0.00 sec
[ Info: CTMRG  18:	obj = +3.423649290995e-01	err = 1.9229756162e-08	time = 0.01 sec
[ Info: CTMRG  19:	obj = +3.423649290995e-01	err = 9.5951149575e-09	time = 0.01 sec
[ Info: CTMRG  20:	obj = +3.423649290995e-01	err = 3.6585761634e-09	time = 0.00 sec
[ Info: CTMRG  21:	obj = +3.423649290995e-01	err = 8.5616189166e-10	time = 0.01 sec
[ Info: CTMRG  22:	obj = +3.423649290995e-01	err = 3.0800030135e-10	time = 0.01 sec
[ Info: CTMRG  23:	obj = +3.423649290995e-01	err = 2.0405422214e-10	time = 0.01 sec
[ Info: CTMRG conv 24:	obj = +3.423649290995e-01	err = 7.9350588207e-11	time = 0.13 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.81e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 1.54e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 7.42e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 1.75e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 2.08e-04
[ Info: BiCGStab linsolve in iteration 3: normres = 6.29e-05
[ Info: BiCGStab linsolve in iteration 3.5: normres = 1.68e-05
[ Info: BiCGStab linsolve in iteration 4: normres = 5.40e-06
[ Info: BiCGStab linsolve in iteration 4.5: normres = 1.48e-05
[ Info: BiCGStab linsolve in iteration 5: normres = 3.78e-06
┌ Info: BiCGStab linsolve converged at iteration 5.5:
│ * norm of residual = 1.81e-08
└ * number of operations = 13
[ Info: CTMRG init:	obj = +4.355241324960e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.049671855166e-01	err = 3.4977566014e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.045100850086e-01	err = 2.0108294005e-01	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.044874876263e-01	err = 2.1035795651e-01	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.044881606001e-01	err = 1.9089865698e-01	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.044882185306e-01	err = 2.0042800366e-02	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.044882217392e-01	err = 5.5656824031e-03	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.044882218988e-01	err = 5.7048147285e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.044882219065e-01	err = 2.3007944270e-04	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.044882219068e-01	err = 3.5688381042e-05	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.044882219068e-01	err = 9.5580911901e-06	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.044882219068e-01	err = 1.8142608163e-06	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.044882219068e-01	err = 4.2890533382e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.044882219068e-01	err = 8.8210060575e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.044882219068e-01	err = 1.9750966814e-08	time = 0.01 sec
[ Info: CTMRG  15:	obj = +5.044882219068e-01	err = 4.1972035907e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.044882219068e-01	err = 9.2044965770e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.044882219068e-01	err = 1.9955644891e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +5.044882219068e-01	err = 4.2217186903e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 8.51e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.54e-03
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.48e-04
[ Info: BiCGStab linsolve in iteration 2: normres = 4.23e-05
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 8.80e-07
└ * number of operations = 7
[ Info: CTMRG init:	obj = +3.422319647713e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.059235206244e-01	err = 2.0255578684e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.112745931363e-01	err = 2.0818285390e-01	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.115117557626e-01	err = 2.3880984707e-01	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.115093847498e-01	err = 1.0893440842e-01	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.115083695228e-01	err = 1.7611497477e-02	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.115083777318e-01	err = 2.4620917957e-03	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.115083844066e-01	err = 7.4826522225e-04	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.115083844800e-01	err = 2.8878212993e-04	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.115083844399e-01	err = 4.7299635369e-05	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.115083844387e-01	err = 1.0303058937e-05	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.115083844390e-01	err = 3.2840305765e-06	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.115083844390e-01	err = 1.3182904640e-06	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.115083844390e-01	err = 3.1656887010e-07	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.115083844390e-01	err = 5.9372142563e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.115083844390e-01	err = 1.6914997756e-08	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.115083844390e-01	err = 7.6918589147e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.115083844390e-01	err = 2.1726348156e-09	time = 0.00 sec
[ Info: CTMRG  18:	obj = +5.115083844390e-01	err = 4.3095702532e-10	time = 0.01 sec
[ Info: CTMRG conv 19:	obj = +5.115083844390e-01	err = 9.0095611039e-11	time = 0.12 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.00e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.15e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 9.17e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.40e-05
[ Info: BiCGStab linsolve in iteration 2.5: normres = 2.92e-06
┌ Info: BiCGStab linsolve converged at iteration 3:
│ * norm of residual = 4.66e-07
└ * number of operations = 8
[ Info: LBFGS: iter    1, time  732.46 s: f = -0.489796540851, ‖∇f‖ = 6.0022e-01, α = 5.94e+01, m = 0, nfg = 5
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.ComplexSpace, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: CTMRG init:	obj = +5.640950274181e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.664827046284e-01	err = 8.5541943898e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.665188272848e-01	err = 9.5361394153e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.665185518575e-01	err = 4.9906223331e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.665185750436e-01	err = 3.9675306336e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.665185766900e-01	err = 9.9772842181e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.665185767074e-01	err = 1.8624251343e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.665185767053e-01	err = 2.4949133561e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.665185767052e-01	err = 3.5158447123e-06	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.665185767052e-01	err = 7.6646557758e-07	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.665185767052e-01	err = 2.6432871984e-07	time = 0.02 sec
[ Info: CTMRG  11:	obj = +5.665185767052e-01	err = 6.1799005969e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.665185767052e-01	err = 1.0719167757e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.665185767052e-01	err = 1.2598877275e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.665185767052e-01	err = 2.3249788335e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.665185767052e-01	err = 6.4079819106e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.90e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.80e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.14e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 3.71e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 8.35e-07
└ * number of operations = 7
[ Info: CTMRG init:	obj = +5.255414209869e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.258510285849e-01	err = 2.3155134355e-02	time = 0.06 sec
[ Info: CTMRG   2:	obj = +5.258515618555e-01	err = 2.5067820962e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.258515132282e-01	err = 1.2983127504e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.258515440844e-01	err = 2.2122995624e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.258515454127e-01	err = 8.3607529323e-04	time = 0.25 sec
[ Info: CTMRG   6:	obj = +5.258515453052e-01	err = 9.4699099439e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.258515452959e-01	err = 3.4372419957e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.258515452961e-01	err = 1.4651733977e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.258515452961e-01	err = 3.4728653268e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.258515452962e-01	err = 3.9807656440e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.258515452961e-01	err = 9.4139336219e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.258515452962e-01	err = 5.8812312879e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.258515452962e-01	err = 1.8138314946e-08	time = 0.01 sec
[ Info: CTMRG  14:	obj = +5.258515452962e-01	err = 3.4054119944e-09	time = 0.01 sec
[ Info: CTMRG  15:	obj = +5.258515452962e-01	err = 4.4531114588e-10	time = 0.01 sec
[ Info: CTMRG  16:	obj = +5.258515452961e-01	err = 1.6854097708e-10	time = 0.01 sec
[ Info: CTMRG conv 17:	obj = +5.258515452962e-01	err = 7.6966581908e-11	time = 0.39 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.05e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.65e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 8.86e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.40e-05
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 5.17e-07
└ * number of operations = 7
[ Info: LBFGS: iter    2, time  734.49 s: f = -0.501984649868, ‖∇f‖ = 5.3739e-01, α = 2.80e-01, m = 1, nfg = 2
[ Info: CTMRG init:	obj = +4.968273104592e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.962307463119e-01	err = 4.3932885704e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.962149486894e-01	err = 3.0548043697e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.962171150065e-01	err = 1.4274310610e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.962172915061e-01	err = 2.2606808105e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.962172843461e-01	err = 1.8231907969e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.962172829927e-01	err = 4.1053122318e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.962172829934e-01	err = 4.4027754764e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.962172830022e-01	err = 1.8670932046e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.962172830025e-01	err = 7.2998707694e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.962172830024e-01	err = 1.7299831046e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.962172830024e-01	err = 3.1059078477e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.962172830024e-01	err = 9.7390964221e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.962172830024e-01	err = 4.7426002166e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.962172830024e-01	err = 1.4007557897e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.962172830024e-01	err = 2.5859635083e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.962172830024e-01	err = 5.3995702982e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.962172830024e-01	err = 2.3838227266e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.962172830024e-01	err = 9.3742778827e-11	time = 0.11 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.58e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.39e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.04e-04
[ Info: BiCGStab linsolve in iteration 2: normres = 3.85e-05
[ Info: BiCGStab linsolve in iteration 2.5: normres = 2.38e-06
┌ Info: BiCGStab linsolve converged at iteration 3:
│ * norm of residual = 4.05e-07
└ * number of operations = 8
[ Info: LBFGS: iter    3, time  734.72 s: f = -0.523163971924, ‖∇f‖ = 3.9927e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: CTMRG init:	obj = +3.562598080445e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +3.596831790076e-01	err = 2.1723962944e-01	time = 0.00 sec
[ Info: CTMRG   2:	obj = +3.595275896187e-01	err = 1.3970935615e-01	time = 0.00 sec
[ Info: CTMRG   3:	obj = +3.595245983237e-01	err = 6.7542524698e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +3.595238759998e-01	err = 2.1141802791e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +3.595232562709e-01	err = 7.4741289874e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +3.595231453278e-01	err = 1.6565241775e-03	time = 0.00 sec
[ Info: CTMRG   7:	obj = +3.595231340288e-01	err = 5.2796105893e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +3.595231328369e-01	err = 2.0585129130e-04	time = 0.00 sec
[ Info: CTMRG   9:	obj = +3.595231326390e-01	err = 6.8342124066e-05	time = 0.00 sec
[ Info: CTMRG  10:	obj = +3.595231326074e-01	err = 2.5627513069e-05	time = 0.00 sec
[ Info: CTMRG  11:	obj = +3.595231326030e-01	err = 9.3034807248e-06	time = 0.00 sec
[ Info: CTMRG  12:	obj = +3.595231326025e-01	err = 3.5835141132e-06	time = 0.00 sec
[ Info: CTMRG  13:	obj = +3.595231326024e-01	err = 1.3899078677e-06	time = 0.00 sec
[ Info: CTMRG  14:	obj = +3.595231326024e-01	err = 5.1505027780e-07	time = 0.00 sec
[ Info: CTMRG  15:	obj = +3.595231326024e-01	err = 1.9089200667e-07	time = 0.00 sec
[ Info: CTMRG  16:	obj = +3.595231326024e-01	err = 7.0182635216e-08	time = 0.00 sec
[ Info: CTMRG  17:	obj = +3.595231326024e-01	err = 2.5505438905e-08	time = 0.00 sec
[ Info: CTMRG  18:	obj = +3.595231326024e-01	err = 9.3197226471e-09	time = 0.02 sec
[ Info: CTMRG  19:	obj = +3.595231326024e-01	err = 3.4246060196e-09	time = 0.01 sec
[ Info: CTMRG  20:	obj = +3.595231326024e-01	err = 1.2717678278e-09	time = 0.01 sec
[ Info: CTMRG  21:	obj = +3.595231326024e-01	err = 4.7430899057e-10	time = 0.01 sec
[ Info: CTMRG  22:	obj = +3.595231326024e-01	err = 1.7723079135e-10	time = 0.01 sec
[ Info: CTMRG conv 23:	obj = +3.595231326024e-01	err = 6.5988037074e-11	time = 0.13 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.55e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 1.22e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.73e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 4.32e-04
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.56e-04
[ Info: BiCGStab linsolve in iteration 3: normres = 2.89e-05
[ Info: BiCGStab linsolve in iteration 3.5: normres = 6.28e-03
[ Info: BiCGStab linsolve in iteration 4: normres = 1.28e-03
┌ Info: BiCGStab linsolve converged at iteration 4.5:
│ * norm of residual = 5.31e-08
└ * number of operations = 11
[ Info: CTMRG init:	obj = +4.680044618365e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.678878042342e-01	err = 4.0067386290e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.678751770234e-01	err = 3.1844803872e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.678757002284e-01	err = 2.3390575547e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.678758088696e-01	err = 2.3333366616e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.678758087099e-01	err = 8.7270862010e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.678758078482e-01	err = 2.0206487957e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.678758078169e-01	err = 4.1103913248e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.678758078226e-01	err = 1.0481808577e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.678758078231e-01	err = 4.2077137369e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.678758078231e-01	err = 1.4649754992e-06	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.678758078231e-01	err = 4.1264370851e-07	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.678758078231e-01	err = 1.0850345284e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.678758078231e-01	err = 2.7381579388e-08	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.678758078231e-01	err = 1.0813919781e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.678758078231e-01	err = 3.4304424306e-09	time = 0.01 sec
[ Info: CTMRG  16:	obj = +4.678758078231e-01	err = 9.2798340115e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.678758078231e-01	err = 2.3475738746e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.678758078231e-01	err = 7.2973345454e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.61e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.57e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.11e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 2.36e-04
[ Info: BiCGStab linsolve in iteration 2.5: normres = 5.22e-06
[ Info: BiCGStab linsolve in iteration 3: normres = 1.07e-06
┌ Info: BiCGStab linsolve converged at iteration 3.5:
│ * norm of residual = 1.07e-07
└ * number of operations = 9
[ Info: LBFGS: iter    4, time  735.25 s: f = -0.538654390178, ‖∇f‖ = 4.1552e-01, α = 2.29e-01, m = 3, nfg = 2
[ Info: CTMRG init:	obj = +1.361090548391e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +2.492681304638e-01	err = 2.7633751803e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +2.729788177171e-01	err = 3.3254602721e-01	time = 0.01 sec
[ Info: CTMRG   3:	obj = +2.771967161921e-01	err = 1.2871942679e-01	time = 0.01 sec
[ Info: CTMRG   4:	obj = +2.779581748526e-01	err = 8.4067728014e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +2.780581629514e-01	err = 6.0831204990e-02	time = 0.01 sec
[ Info: CTMRG   6:	obj = +2.780474606214e-01	err = 5.6932247083e-02	time = 0.01 sec
[ Info: CTMRG   7:	obj = +2.780302531384e-01	err = 3.4222154914e-02	time = 0.01 sec
[ Info: CTMRG   8:	obj = +2.780200353034e-01	err = 3.3736099111e-02	time = 0.01 sec
[ Info: CTMRG   9:	obj = +2.780150195975e-01	err = 2.1491380928e-02	time = 0.01 sec
[ Info: CTMRG  10:	obj = +2.780127868273e-01	err = 1.6640500825e-02	time = 0.01 sec
[ Info: CTMRG  11:	obj = +2.780118071149e-01	err = 1.0801060671e-02	time = 0.01 sec
[ Info: CTMRG  12:	obj = +2.780114025117e-01	err = 7.3818436743e-03	time = 0.01 sec
[ Info: CTMRG  13:	obj = +2.780112272568e-01	err = 4.7841431573e-03	time = 0.01 sec
[ Info: CTMRG  14:	obj = +2.780111585913e-01	err = 3.1469231082e-03	time = 0.01 sec
[ Info: CTMRG  15:	obj = +2.780111276329e-01	err = 2.0310417700e-03	time = 0.01 sec
[ Info: CTMRG  16:	obj = +2.780111165486e-01	err = 1.3193954295e-03	time = 0.01 sec
[ Info: CTMRG  17:	obj = +2.780111109125e-01	err = 8.4931855133e-04	time = 0.01 sec
[ Info: CTMRG  18:	obj = +2.780111092872e-01	err = 5.4832384428e-04	time = 0.01 sec
[ Info: CTMRG  19:	obj = +2.780111081752e-01	err = 3.5236207484e-04	time = 0.01 sec
[ Info: CTMRG  20:	obj = +2.780111080036e-01	err = 2.2671813449e-04	time = 0.01 sec
[ Info: CTMRG  21:	obj = +2.780111077502e-01	err = 1.4552348057e-04	time = 0.01 sec
[ Info: CTMRG  22:	obj = +2.780111077635e-01	err = 9.3454119058e-05	time = 0.01 sec
[ Info: CTMRG  23:	obj = +2.780111076937e-01	err = 5.9938526146e-05	time = 0.01 sec
[ Info: CTMRG  24:	obj = +2.780111077133e-01	err = 3.8449675674e-05	time = 0.01 sec
[ Info: CTMRG  25:	obj = +2.780111076904e-01	err = 2.4647795488e-05	time = 0.01 sec
[ Info: CTMRG  26:	obj = +2.780111077008e-01	err = 1.5801059838e-05	time = 0.01 sec
[ Info: CTMRG  27:	obj = +2.780111076924e-01	err = 1.0125795229e-05	time = 0.01 sec
[ Info: CTMRG  28:	obj = +2.780111076971e-01	err = 6.4889439047e-06	time = 0.01 sec
[ Info: CTMRG  29:	obj = +2.780111076938e-01	err = 4.1574499013e-06	time = 0.01 sec
[ Info: CTMRG  30:	obj = +2.780111076958e-01	err = 2.6636405300e-06	time = 0.01 sec
[ Info: CTMRG  31:	obj = +2.780111076945e-01	err = 1.7063672352e-06	time = 0.01 sec
[ Info: CTMRG  32:	obj = +2.780111076953e-01	err = 1.0931104129e-06	time = 0.01 sec
[ Info: CTMRG  33:	obj = +2.780111076947e-01	err = 7.0020655988e-07	time = 0.01 sec
[ Info: CTMRG  34:	obj = +2.780111076951e-01	err = 4.4852201369e-07	time = 0.01 sec
[ Info: CTMRG  35:	obj = +2.780111076949e-01	err = 2.8729264916e-07	time = 0.01 sec
[ Info: CTMRG  36:	obj = +2.780111076950e-01	err = 1.8401865610e-07	time = 0.01 sec
[ Info: CTMRG  37:	obj = +2.780111076949e-01	err = 1.1786625616e-07	time = 0.01 sec
[ Info: CTMRG  38:	obj = +2.780111076950e-01	err = 7.5494400898e-08	time = 0.01 sec
[ Info: CTMRG  39:	obj = +2.780111076949e-01	err = 4.8354217077e-08	time = 0.01 sec
[ Info: CTMRG  40:	obj = +2.780111076950e-01	err = 3.0970795192e-08	time = 0.01 sec
[ Info: CTMRG  41:	obj = +2.780111076949e-01	err = 1.9836592332e-08	time = 0.01 sec
[ Info: CTMRG  42:	obj = +2.780111076950e-01	err = 1.2705178147e-08	time = 0.01 sec
[ Info: CTMRG  43:	obj = +2.780111076949e-01	err = 8.1375289079e-09	time = 0.01 sec
[ Info: CTMRG  44:	obj = +2.780111076949e-01	err = 5.2119907250e-09	time = 0.01 sec
[ Info: CTMRG  45:	obj = +2.780111076949e-01	err = 3.3382079689e-09	time = 0.01 sec
[ Info: CTMRG  46:	obj = +2.780111076949e-01	err = 2.1380735358e-09	time = 0.01 sec
[ Info: CTMRG  47:	obj = +2.780111076949e-01	err = 1.3694050211e-09	time = 0.01 sec
[ Info: CTMRG  48:	obj = +2.780111076949e-01	err = 8.7708492359e-10	time = 0.01 sec
[ Info: CTMRG  49:	obj = +2.780111076949e-01	err = 5.6175862626e-10	time = 0.01 sec
[ Info: CTMRG  50:	obj = +2.780111076949e-01	err = 3.5979756862e-10	time = 0.01 sec
[ Info: CTMRG  51:	obj = +2.780111076949e-01	err = 2.3044300143e-10	time = 0.01 sec
[ Info: CTMRG  52:	obj = +2.780111076949e-01	err = 1.4759372169e-10	time = 0.01 sec
[ Info: CTMRG conv 53:	obj = +2.780111076949e-01	err = 9.4533846806e-11	time = 0.29 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.29e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 5.56e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.21e-02
[ Info: BiCGStab linsolve in iteration 2: normres = 5.92e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 3.69e-03
[ Info: BiCGStab linsolve in iteration 3: normres = 1.54e-03
[ Info: BiCGStab linsolve in iteration 3.5: normres = 1.47e-04
[ Info: BiCGStab linsolve in iteration 4: normres = 3.29e-05
[ Info: BiCGStab linsolve in iteration 4.5: normres = 1.35e-05
[ Info: BiCGStab linsolve in iteration 5: normres = 2.53e-06
┌ Info: BiCGStab linsolve converged at iteration 5.5:
│ * norm of residual = 5.01e-07
└ * number of operations = 13
[ Info: CTMRG init:	obj = +1.747462633461e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +2.718493922832e-01	err = 2.7890398206e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +2.796442514830e-01	err = 3.4014977169e-01	time = 0.01 sec
[ Info: CTMRG   3:	obj = +2.803869164174e-01	err = 9.5922304517e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +2.803706435718e-01	err = 6.6406375999e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +2.803183420351e-01	err = 4.4063125080e-02	time = 0.01 sec
[ Info: CTMRG   6:	obj = +2.802922722169e-01	err = 4.0155182717e-02	time = 0.01 sec
[ Info: CTMRG   7:	obj = +2.802817958119e-01	err = 2.1359492680e-02	time = 0.01 sec
[ Info: CTMRG   8:	obj = +2.802779179079e-01	err = 1.3997059564e-02	time = 0.01 sec
[ Info: CTMRG   9:	obj = +2.802765110153e-01	err = 8.0592501297e-03	time = 0.01 sec
[ Info: CTMRG  10:	obj = +2.802760168664e-01	err = 5.0941094049e-03	time = 0.01 sec
[ Info: CTMRG  11:	obj = +2.802758392735e-01	err = 3.0805114137e-03	time = 0.00 sec
[ Info: CTMRG  12:	obj = +2.802757787064e-01	err = 1.8814415892e-03	time = 0.01 sec
[ Info: CTMRG  13:	obj = +2.802757563217e-01	err = 1.1265596873e-03	time = 0.01 sec
[ Info: CTMRG  14:	obj = +2.802757491228e-01	err = 6.7670929086e-04	time = 0.01 sec
[ Info: CTMRG  15:	obj = +2.802757462098e-01	err = 4.0309421683e-04	time = 0.01 sec
[ Info: CTMRG  16:	obj = +2.802757454171e-01	err = 2.4028400007e-04	time = 0.01 sec
[ Info: CTMRG  17:	obj = +2.802757450036e-01	err = 1.4273046611e-04	time = 0.01 sec
[ Info: CTMRG  18:	obj = +2.802757449388e-01	err = 8.4783470625e-05	time = 0.01 sec
[ Info: CTMRG  19:	obj = +2.802757448686e-01	err = 5.0287710298e-05	time = 0.01 sec
[ Info: CTMRG  20:	obj = +2.802757448720e-01	err = 2.9823338887e-05	time = 0.00 sec
[ Info: CTMRG  21:	obj = +2.802757448566e-01	err = 1.7675631746e-05	time = 0.01 sec
[ Info: CTMRG  22:	obj = +2.802757448611e-01	err = 1.0474783786e-05	time = 0.01 sec
[ Info: CTMRG  23:	obj = +2.802757448568e-01	err = 6.2057598499e-06	time = 0.00 sec
[ Info: CTMRG  24:	obj = +2.802757448588e-01	err = 3.6763227068e-06	time = 0.01 sec
[ Info: CTMRG  25:	obj = +2.802757448574e-01	err = 2.1776040456e-06	time = 0.01 sec
[ Info: CTMRG  26:	obj = +2.802757448581e-01	err = 1.2898116978e-06	time = 0.01 sec
[ Info: CTMRG  27:	obj = +2.802757448577e-01	err = 7.6392306184e-07	time = 0.00 sec
[ Info: CTMRG  28:	obj = +2.802757448579e-01	err = 4.5244238301e-07	time = 0.01 sec
[ Info: CTMRG  29:	obj = +2.802757448578e-01	err = 2.6795749960e-07	time = 0.01 sec
[ Info: CTMRG  30:	obj = +2.802757448579e-01	err = 1.5869510452e-07	time = 0.01 sec
[ Info: CTMRG  31:	obj = +2.802757448578e-01	err = 9.3984457232e-08	time = 0.01 sec
[ Info: CTMRG  32:	obj = +2.802757448579e-01	err = 5.5660358279e-08	time = 0.01 sec
[ Info: CTMRG  33:	obj = +2.802757448578e-01	err = 3.2963519525e-08	time = 0.01 sec
[ Info: CTMRG  34:	obj = +2.802757448578e-01	err = 1.9521800691e-08	time = 0.01 sec
[ Info: CTMRG  35:	obj = +2.802757448578e-01	err = 1.1561256755e-08	time = 0.00 sec
[ Info: CTMRG  36:	obj = +2.802757448578e-01	err = 6.8468312133e-09	time = 0.01 sec
[ Info: CTMRG  37:	obj = +2.802757448578e-01	err = 4.0548386307e-09	time = 0.01 sec
[ Info: CTMRG  38:	obj = +2.802757448578e-01	err = 2.4013588735e-09	time = 0.01 sec
[ Info: CTMRG  39:	obj = +2.802757448578e-01	err = 1.4221332948e-09	time = 0.00 sec
[ Info: CTMRG  40:	obj = +2.802757448578e-01	err = 8.4221677250e-10	time = 0.01 sec
[ Info: CTMRG  41:	obj = +2.802757448578e-01	err = 4.9877801141e-10	time = 0.01 sec
[ Info: CTMRG  42:	obj = +2.802757448578e-01	err = 2.9538585025e-10	time = 0.01 sec
[ Info: CTMRG  43:	obj = +2.802757448578e-01	err = 1.7493268954e-10	time = 0.01 sec
[ Info: CTMRG  44:	obj = +2.802757448578e-01	err = 1.0359885552e-10	time = 0.01 sec
[ Info: CTMRG conv 45:	obj = +2.802757448578e-01	err = 6.1354058626e-11	time = 0.25 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.35e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 4.03e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.41e-02
[ Info: BiCGStab linsolve in iteration 2: normres = 2.72e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.92e-03
[ Info: BiCGStab linsolve in iteration 3: normres = 5.49e-04
[ Info: BiCGStab linsolve in iteration 3.5: normres = 1.52e-04
[ Info: BiCGStab linsolve in iteration 4: normres = 3.33e-05
[ Info: BiCGStab linsolve in iteration 4.5: normres = 2.37e-06
┌ Info: BiCGStab linsolve converged at iteration 5:
│ * norm of residual = 4.80e-07
└ * number of operations = 12
[ Info: CTMRG init:	obj = +3.218181656181e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +3.347408492966e-01	err = 2.2748631899e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +3.349727572571e-01	err = 1.7705654425e-01	time = 0.01 sec
[ Info: CTMRG   3:	obj = +3.349557854112e-01	err = 5.9986367263e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +3.349497623768e-01	err = 2.3135597061e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +3.349481474864e-01	err = 7.5570497872e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +3.349478065136e-01	err = 1.9538199085e-03	time = 0.01 sec
[ Info: CTMRG   7:	obj = +3.349477423423e-01	err = 1.2340473771e-03	time = 0.01 sec
[ Info: CTMRG   8:	obj = +3.349477304763e-01	err = 4.3525791110e-04	time = 0.01 sec
[ Info: CTMRG   9:	obj = +3.349477282227e-01	err = 2.0650966963e-04	time = 0.01 sec
[ Info: CTMRG  10:	obj = +3.349477278074e-01	err = 8.3822899760e-05	time = 0.00 sec
[ Info: CTMRG  11:	obj = +3.349477277259e-01	err = 3.7732886311e-05	time = 0.00 sec
[ Info: CTMRG  12:	obj = +3.349477277123e-01	err = 1.6144491689e-05	time = 0.01 sec
[ Info: CTMRG  13:	obj = +3.349477277090e-01	err = 7.1473298752e-06	time = 0.01 sec
[ Info: CTMRG  14:	obj = +3.349477277087e-01	err = 3.0866539934e-06	time = 0.00 sec
[ Info: CTMRG  15:	obj = +3.349477277085e-01	err = 1.3443256803e-06	time = 0.00 sec
[ Info: CTMRG  16:	obj = +3.349477277085e-01	err = 5.7924833331e-07	time = 0.01 sec
[ Info: CTMRG  17:	obj = +3.349477277085e-01	err = 2.5062959720e-07	time = 0.00 sec
[ Info: CTMRG  18:	obj = +3.349477277085e-01	err = 1.0812525472e-07	time = 0.01 sec
[ Info: CTMRG  19:	obj = +3.349477277085e-01	err = 4.6777995734e-08	time = 0.00 sec
[ Info: CTMRG  20:	obj = +3.349477277085e-01	err = 2.0222471517e-08	time = 0.01 sec
[ Info: CTMRG  21:	obj = +3.349477277085e-01	err = 8.7525869465e-09	time = 0.01 sec
[ Info: CTMRG  22:	obj = +3.349477277085e-01	err = 3.7861132156e-09	time = 0.00 sec
[ Info: CTMRG  23:	obj = +3.349477277085e-01	err = 1.6381695143e-09	time = 0.01 sec
[ Info: CTMRG  24:	obj = +3.349477277085e-01	err = 7.0852749498e-10	time = 0.01 sec
[ Info: CTMRG  25:	obj = +3.349477277085e-01	err = 3.0646711677e-10	time = 0.00 sec
[ Info: CTMRG  26:	obj = +3.349477277085e-01	err = 1.3253887089e-10	time = 0.01 sec
[ Info: CTMRG conv 27:	obj = +3.349477277085e-01	err = 5.7327038894e-11	time = 0.15 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.34e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 2.00e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.98e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 7.77e-04
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.75e-04
[ Info: BiCGStab linsolve in iteration 3: normres = 4.04e-05
[ Info: BiCGStab linsolve in iteration 3.5: normres = 2.83e-05
[ Info: BiCGStab linsolve in iteration 4: normres = 5.78e-06
┌ Info: BiCGStab linsolve converged at iteration 4.5:
│ * norm of residual = 1.64e-07
└ * number of operations = 11
[ Info: CTMRG init:	obj = +4.465844626156e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.466478643504e-01	err = 3.0531464291e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.466423237538e-01	err = 2.6259975717e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.466423269680e-01	err = 2.2698419728e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.466423753841e-01	err = 1.5737800890e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.466423771165e-01	err = 8.0106757224e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.466423767483e-01	err = 1.4995743009e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.466423767172e-01	err = 4.6336099280e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.466423767192e-01	err = 7.4052685550e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.466423767195e-01	err = 2.7312234384e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.466423767195e-01	err = 1.2667504865e-06	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.466423767195e-01	err = 4.6783595574e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.466423767195e-01	err = 1.3108458777e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.466423767195e-01	err = 2.8100062526e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.466423767195e-01	err = 8.4836640681e-09	time = 0.01 sec
[ Info: CTMRG  15:	obj = +4.466423767195e-01	err = 3.6015760753e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.466423767195e-01	err = 1.2269490759e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.466423767195e-01	err = 3.1040872266e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.466423767195e-01	err = 6.8059332535e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.73e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.43e-03
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.12e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 4.18e-04
[ Info: BiCGStab linsolve in iteration 2.5: normres = 9.07e-06
[ Info: BiCGStab linsolve in iteration 3: normres = 2.09e-06
┌ Info: BiCGStab linsolve converged at iteration 3.5:
│ * norm of residual = 2.92e-07
└ * number of operations = 9
[ Info: LBFGS: iter    5, time  736.72 s: f = -0.549821445064, ‖∇f‖ = 4.4002e-01, α = 6.90e-02, m = 4, nfg = 4
[ Info: CTMRG init:	obj = +2.412790896317e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +2.983930870846e-01	err = 2.5141343896e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +3.004376846111e-01	err = 2.7968754892e-01	time = 0.01 sec
[ Info: CTMRG   3:	obj = +3.004971439000e-01	err = 1.3426217604e-01	time = 0.01 sec
[ Info: CTMRG   4:	obj = +3.004553813954e-01	err = 5.8970983213e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +3.004376693784e-01	err = 3.0280891908e-02	time = 0.01 sec
[ Info: CTMRG   6:	obj = +3.004319417580e-01	err = 1.5672899027e-02	time = 0.01 sec
[ Info: CTMRG   7:	obj = +3.004302027936e-01	err = 5.8028874433e-03	time = 0.01 sec
[ Info: CTMRG   8:	obj = +3.004296867220e-01	err = 2.5948112491e-03	time = 0.01 sec
[ Info: CTMRG   9:	obj = +3.004295340477e-01	err = 1.1447200167e-03	time = 0.01 sec
[ Info: CTMRG  10:	obj = +3.004294892962e-01	err = 6.6058094381e-04	time = 0.01 sec
[ Info: CTMRG  11:	obj = +3.004294760322e-01	err = 3.4254230926e-04	time = 0.01 sec
[ Info: CTMRG  12:	obj = +3.004294721864e-01	err = 1.9141797831e-04	time = 0.01 sec
[ Info: CTMRG  13:	obj = +3.004294710268e-01	err = 1.0281286744e-04	time = 0.01 sec
[ Info: CTMRG  14:	obj = +3.004294707015e-01	err = 5.6363800062e-05	time = 0.01 sec
[ Info: CTMRG  15:	obj = +3.004294705975e-01	err = 3.0499158004e-05	time = 0.00 sec
[ Info: CTMRG  16:	obj = +3.004294705714e-01	err = 1.6597746883e-05	time = 0.01 sec
[ Info: CTMRG  17:	obj = +3.004294705613e-01	err = 8.9959318478e-06	time = 0.01 sec
[ Info: CTMRG  18:	obj = +3.004294705597e-01	err = 4.8833705049e-06	time = 0.01 sec
[ Info: CTMRG  19:	obj = +3.004294705585e-01	err = 2.6476507833e-06	time = 0.01 sec
[ Info: CTMRG  20:	obj = +3.004294705585e-01	err = 1.4360833121e-06	time = 0.01 sec
[ Info: CTMRG  21:	obj = +3.004294705583e-01	err = 7.7864696454e-07	time = 0.01 sec
[ Info: CTMRG  22:	obj = +3.004294705584e-01	err = 4.2222672142e-07	time = 0.00 sec
[ Info: CTMRG  23:	obj = +3.004294705583e-01	err = 2.2893083353e-07	time = 0.00 sec
[ Info: CTMRG  24:	obj = +3.004294705583e-01	err = 1.2412889552e-07	time = 0.00 sec
[ Info: CTMRG  25:	obj = +3.004294705583e-01	err = 6.7301966372e-08	time = 0.01 sec
[ Info: CTMRG  26:	obj = +3.004294705583e-01	err = 3.6490897114e-08	time = 0.01 sec
[ Info: CTMRG  27:	obj = +3.004294705583e-01	err = 1.9785054528e-08	time = 0.01 sec
[ Info: CTMRG  28:	obj = +3.004294705583e-01	err = 1.0727293977e-08	time = 0.00 sec
[ Info: CTMRG  29:	obj = +3.004294705583e-01	err = 5.8162335389e-09	time = 0.01 sec
[ Info: CTMRG  30:	obj = +3.004294705583e-01	err = 3.1535036279e-09	time = 0.01 sec
[ Info: CTMRG  31:	obj = +3.004294705583e-01	err = 1.7097974506e-09	time = 0.00 sec
[ Info: CTMRG  32:	obj = +3.004294705583e-01	err = 9.2703482284e-10	time = 0.00 sec
[ Info: CTMRG  33:	obj = +3.004294705583e-01	err = 5.0262776853e-10	time = 0.00 sec
[ Info: CTMRG  34:	obj = +3.004294705583e-01	err = 2.7251995940e-10	time = 0.01 sec
[ Info: CTMRG  35:	obj = +3.004294705583e-01	err = 1.4775870840e-10	time = 0.00 sec
[ Info: CTMRG conv 36:	obj = +3.004294705583e-01	err = 8.0112964589e-11	time = 0.20 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.18e-01
[ Info: BiCGStab linsolve in iteration 1: normres = 2.39e-02
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.06e-03
[ Info: BiCGStab linsolve in iteration 2: normres = 1.51e-03
[ Info: BiCGStab linsolve in iteration 2.5: normres = 2.98e-04
[ Info: BiCGStab linsolve in iteration 3: normres = 7.41e-05
[ Info: BiCGStab linsolve in iteration 3.5: normres = 1.98e-05
[ Info: BiCGStab linsolve in iteration 4: normres = 2.51e-06
[ Info: BiCGStab linsolve in iteration 4.5: normres = 6.09e-06
[ Info: BiCGStab linsolve in iteration 5: normres = 1.08e-06
┌ Info: BiCGStab linsolve converged at iteration 5.5:
│ * norm of residual = 2.17e-08
└ * number of operations = 13
[ Info: CTMRG init:	obj = +4.029197970323e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.040265024968e-01	err = 8.3906634603e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.040213992499e-01	err = 5.3339298008e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.040200587979e-01	err = 1.8853025793e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.040201322910e-01	err = 4.1227587370e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.040201455082e-01	err = 1.0596262918e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.040201448203e-01	err = 2.8068145851e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.040201445920e-01	err = 1.0381716561e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.040201445821e-01	err = 2.5617878885e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.040201445837e-01	err = 6.0269535884e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.040201445838e-01	err = 2.5252478073e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.040201445838e-01	err = 9.0508673606e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.040201445838e-01	err = 2.6859837947e-07	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.040201445838e-01	err = 6.3087542319e-08	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.040201445838e-01	err = 2.0694143817e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.040201445838e-01	err = 8.5466921256e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.040201445838e-01	err = 2.9874902093e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.040201445838e-01	err = 8.3859571372e-10	time = 0.01 sec
[ Info: CTMRG  18:	obj = +4.040201445838e-01	err = 2.1393527162e-10	time = 0.01 sec
[ Info: CTMRG conv 19:	obj = +4.040201445838e-01	err = 7.1518605014e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 7.79e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.31e-03
[ Info: BiCGStab linsolve in iteration 1.5: normres = 9.63e-04
[ Info: BiCGStab linsolve in iteration 2: normres = 1.54e-04
[ Info: BiCGStab linsolve in iteration 2.5: normres = 3.10e-05
[ Info: BiCGStab linsolve in iteration 3: normres = 7.89e-06
[ Info: BiCGStab linsolve in iteration 3.5: normres = 2.64e-06
┌ Info: BiCGStab linsolve converged at iteration 4:
│ * norm of residual = 5.38e-07
└ * number of operations = 10
[ Info: LBFGS: iter    6, time  737.65 s: f = -0.569016778155, ‖∇f‖ = 4.8450e-01, α = 2.26e-01, m = 5, nfg = 2
[ Info: CTMRG init:	obj = +4.035207942335e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.042584410425e-01	err = 5.2437278550e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.042836556690e-01	err = 4.9527012746e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.042852299400e-01	err = 2.4699437399e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.042851646966e-01	err = 9.1509951326e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.042851450449e-01	err = 9.8690854240e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.042851437275e-01	err = 2.8473471054e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.042851437211e-01	err = 1.0573289997e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.042851437238e-01	err = 3.7637932675e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.042851437234e-01	err = 1.3583952250e-05	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.042851437233e-01	err = 2.9014142659e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.042851437233e-01	err = 4.9034564121e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.042851437233e-01	err = 1.1013089712e-07	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.042851437233e-01	err = 5.8263635065e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.042851437233e-01	err = 2.0927785712e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.042851437233e-01	err = 5.3087796933e-09	time = 0.01 sec
[ Info: CTMRG  16:	obj = +4.042851437233e-01	err = 1.7248442558e-09	time = 0.01 sec
[ Info: CTMRG  17:	obj = +4.042851437233e-01	err = 5.1762627037e-10	time = 0.01 sec
[ Info: CTMRG  18:	obj = +4.042851437233e-01	err = 1.6893888947e-10	time = 0.01 sec
[ Info: CTMRG conv 19:	obj = +4.042851437233e-01	err = 5.9271458341e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 8.95e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.48e-03
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.04e-04
[ Info: BiCGStab linsolve in iteration 2: normres = 1.08e-04
[ Info: BiCGStab linsolve in iteration 2.5: normres = 6.61e-04
[ Info: BiCGStab linsolve in iteration 3: normres = 1.19e-04
[ Info: BiCGStab linsolve in iteration 3.5: normres = 7.44e-06
[ Info: BiCGStab linsolve in iteration 4: normres = 1.36e-06
┌ Info: BiCGStab linsolve converged at iteration 4.5:
│ * norm of residual = 4.36e-09
└ * number of operations = 11
[ Info: LBFGS: iter    7, time  737.95 s: f = -0.587127261652, ‖∇f‖ = 4.1972e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: CTMRG init:	obj = +4.494454290295e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.488017112917e-01	err = 7.5042395244e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.487977728719e-01	err = 4.2518831105e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.487981795686e-01	err = 3.4979894660e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.487981689220e-01	err = 6.1684772135e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.487981640997e-01	err = 1.6222504331e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.487981636483e-01	err = 2.4136515871e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.487981636206e-01	err = 5.7927862064e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.487981636191e-01	err = 1.7706002698e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.487981636190e-01	err = 5.3192324036e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.487981636190e-01	err = 1.4283361356e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.487981636190e-01	err = 3.5279836033e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.487981636190e-01	err = 8.3983532294e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.487981636190e-01	err = 2.0592313254e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.487981636190e-01	err = 5.3861869203e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.487981636190e-01	err = 1.4742300511e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.487981636190e-01	err = 4.1020401276e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.487981636190e-01	err = 1.1476630177e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.487981636190e-01	err = 3.2198542704e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 5.08e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.54e-03
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.64e-04
[ Info: BiCGStab linsolve in iteration 2: normres = 2.50e-05
[ Info: BiCGStab linsolve in iteration 2.5: normres = 3.70e-06
┌ Info: BiCGStab linsolve converged at iteration 3:
│ * norm of residual = 6.68e-07
└ * number of operations = 8
[ Info: LBFGS: iter    8, time  738.14 s: f = -0.600154758006, ‖∇f‖ = 2.1793e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: CTMRG init:	obj = +4.569285721684e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.572522604787e-01	err = 2.1243836175e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.572538951113e-01	err = 2.4214513898e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.572535377757e-01	err = 2.1692309351e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.572535033057e-01	err = 4.4260078016e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.572535009899e-01	err = 6.0942657421e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.572535008401e-01	err = 1.5510681652e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.572535008298e-01	err = 4.0477781029e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.572535008290e-01	err = 1.0691965424e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.572535008289e-01	err = 2.7462527254e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.572535008289e-01	err = 7.0843815967e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.572535008289e-01	err = 1.8449048386e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.572535008289e-01	err = 4.9036176537e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.572535008289e-01	err = 1.3215402770e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.572535008289e-01	err = 3.5995433118e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.572535008289e-01	err = 9.8473999376e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.572535008289e-01	err = 2.6975564134e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.572535008289e-01	err = 7.3811792327e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 4.62e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.35e-03
[ Info: BiCGStab linsolve in iteration 1.5: normres = 9.64e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.39e-05
[ Info: BiCGStab linsolve in iteration 2.5: normres = 4.87e-06
┌ Info: BiCGStab linsolve converged at iteration 3:
│ * norm of residual = 4.89e-07
└ * number of operations = 8
[ Info: LBFGS: iter    9, time  738.33 s: f = -0.606883012825, ‖∇f‖ = 1.9566e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: CTMRG init:	obj = +4.751460494098e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.901950895617e-01	err = 1.2581520933e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.904532787071e-01	err = 8.1457829973e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.904572215992e-01	err = 7.1511426296e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.904572639531e-01	err = 3.0996938051e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.904572634247e-01	err = 1.5577559375e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.904572633539e-01	err = 5.0453343455e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.904572633498e-01	err = 6.1297117723e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.904572633495e-01	err = 1.6222827133e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.904572633495e-01	err = 2.9532921887e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.904572633495e-01	err = 7.1162783161e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.904572633495e-01	err = 1.4094107604e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.904572633495e-01	err = 3.2153200798e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.904572633495e-01	err = 6.6490374456e-09	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.904572633495e-01	err = 1.4745212902e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.904572633495e-01	err = 3.1167312715e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.904572633495e-01	err = 6.8196952087e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.15e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.96e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.71e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 4.82e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 3.70e-07
└ * number of operations = 7
[ Info: LBFGS: iter   10, time  738.51 s: f = -0.625040022199, ‖∇f‖ = 3.0328e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: CTMRG init:	obj = +4.903047313693e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.905863636392e-01	err = 4.4650850752e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.905683240376e-01	err = 1.0062185721e-01	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.905676217881e-01	err = 3.1563747458e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.905675942509e-01	err = 1.1871810596e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.905675931490e-01	err = 1.4226558041e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.905675931043e-01	err = 1.1094633837e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.905675931024e-01	err = 5.1903839202e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.905675931023e-01	err = 4.2256440222e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.905675931023e-01	err = 1.9093152851e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.905675931023e-01	err = 1.8676028333e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.905675931023e-01	err = 7.2246028958e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.905675931023e-01	err = 8.5276931973e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.905675931023e-01	err = 2.8057797515e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.905675931023e-01	err = 3.8537126993e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.905675931023e-01	err = 1.1128050445e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.905675931023e-01	err = 1.7180873355e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.09e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.71e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.20e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 5.02e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.82e-07
└ * number of operations = 7
[ Info: LBFGS: iter   11, time  738.70 s: f = -0.639164743235, ‖∇f‖ = 2.3076e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: CTMRG init:	obj = +4.925811754316e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.988304585060e-01	err = 1.0008347269e-01	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.989519899158e-01	err = 1.0647292714e-01	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.989536236066e-01	err = 6.7400475230e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.989536001200e-01	err = 4.0031937321e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.989535965234e-01	err = 3.0676161215e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.989535962958e-01	err = 1.0579611877e-03	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.989535962839e-01	err = 1.2763776966e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.989535962833e-01	err = 3.9852634218e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.989535962833e-01	err = 5.2674200426e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.989535962833e-01	err = 1.6276173936e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.989535962833e-01	err = 2.5791215365e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.989535962833e-01	err = 6.8224384629e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.989535962833e-01	err = 1.2062177443e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.989535962833e-01	err = 2.9135955763e-09	time = 0.01 sec
[ Info: CTMRG  15:	obj = +4.989535962833e-01	err = 5.5044387237e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.989535962833e-01	err = 1.2569373368e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.989535962833e-01	err = 2.4561163224e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 3.18e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.57e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.46e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 6.39e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.55e-07
└ * number of operations = 7
[ Info: LBFGS: iter   12, time  738.89 s: f = -0.647174335216, ‖∇f‖ = 2.6065e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: CTMRG init:	obj = +5.405492682453e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.427801286802e-01	err = 7.6717163902e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.427830011126e-01	err = 7.1399340684e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.427828772067e-01	err = 7.5634327916e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.427828725696e-01	err = 6.2868584438e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.427828724428e-01	err = 7.1550333180e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.427828724396e-01	err = 1.6070816270e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.427828724395e-01	err = 1.4321836621e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.427828724395e-01	err = 3.5694309329e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.427828724395e-01	err = 3.1885117918e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.427828724395e-01	err = 7.9543581307e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.427828724395e-01	err = 7.7248576221e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.427828724395e-01	err = 1.8249489862e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.427828724395e-01	err = 2.0721673430e-10	time = 0.00 sec
[ Info: CTMRG conv 14:	obj = +5.427828724395e-01	err = 4.4087775497e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.97e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.53e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.77e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 5.09e-07
└ * number of operations = 6
[ Info: LBFGS: iter   13, time  739.05 s: f = -0.650338609163, ‖∇f‖ = 1.6108e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: CTMRG init:	obj = +5.305347582917e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.322477179888e-01	err = 6.1823996560e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.322937108615e-01	err = 5.5926282645e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.322949585294e-01	err = 3.3514574593e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.322949936813e-01	err = 2.2919824445e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.322949947127e-01	err = 5.6177783996e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.322949947440e-01	err = 8.2180099169e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.322949947449e-01	err = 1.3884475228e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.322949947450e-01	err = 2.2272647969e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.322949947450e-01	err = 3.7176359415e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.322949947450e-01	err = 6.1570733363e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.322949947450e-01	err = 1.0490306818e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.322949947450e-01	err = 1.7817148203e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.322949947450e-01	err = 3.0873477182e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.322949947450e-01	err = 5.3520909835e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.322949947450e-01	err = 9.3629966180e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.61e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.17e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.92e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.60e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.43e-07
└ * number of operations = 7
[ Info: LBFGS: iter   14, time  739.23 s: f = -0.654606007953, ‖∇f‖ = 7.7724e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: CTMRG init:	obj = +5.318345276045e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.319484700515e-01	err = 6.8827250218e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.319489723280e-01	err = 5.6866171593e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.319489976527e-01	err = 2.4965770639e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.319489987328e-01	err = 7.5580296888e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.319489987762e-01	err = 5.1322009892e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.319489987779e-01	err = 9.1237575731e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.319489987780e-01	err = 2.1496489750e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.319489987780e-01	err = 1.9716342242e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.319489987780e-01	err = 6.8170806486e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.319489987780e-01	err = 5.1868602306e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.319489987780e-01	err = 2.1741363096e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.319489987780e-01	err = 2.1151357768e-09	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.319489987780e-01	err = 7.1604650072e-10	time = 0.00 sec
[ Info: CTMRG conv 14:	obj = +5.319489987780e-01	err = 8.5421124578e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.71e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.24e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.35e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 3.51e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 3.69e-07
└ * number of operations = 7
[ Info: LBFGS: iter   15, time  739.41 s: f = -0.655962567656, ‖∇f‖ = 5.1320e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: CTMRG init:	obj = +5.384478376744e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.384328572163e-01	err = 6.9413634806e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.384325013378e-01	err = 7.4981499690e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.384325200982e-01	err = 4.5539057960e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.384325216386e-01	err = 9.2162673624e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.384325217260e-01	err = 1.3513302582e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.384325217306e-01	err = 7.7229500390e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.384325217308e-01	err = 1.0818900907e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.384325217308e-01	err = 1.5084499532e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.384325217308e-01	err = 2.7373991237e-07	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.384325217308e-01	err = 6.1929993834e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.384325217308e-01	err = 1.1950369765e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.384325217308e-01	err = 2.7338526117e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.384325217308e-01	err = 5.5024907332e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.384325217308e-01	err = 1.2171496601e-10	time = 0.01 sec
[ Info: CTMRG conv 15:	obj = +5.384325217308e-01	err = 2.4618079413e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.67e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.71e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.44e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.22e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 3.58e-07
└ * number of operations = 7
[ Info: LBFGS: iter   16, time  739.59 s: f = -0.657034966533, ‖∇f‖ = 5.6668e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: CTMRG init:	obj = +5.526295024707e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.527997174411e-01	err = 5.1915113688e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.528102602561e-01	err = 6.8532082643e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.528108731603e-01	err = 7.2437586700e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.528109064990e-01	err = 1.5912881053e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.528109082867e-01	err = 1.7152051521e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.528109083823e-01	err = 2.1923062362e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.528109083874e-01	err = 2.8342417225e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.528109083877e-01	err = 5.4527028189e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.528109083877e-01	err = 1.2890298973e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.528109083877e-01	err = 2.9985702873e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.528109083877e-01	err = 6.8949352819e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.528109083877e-01	err = 1.5887781202e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.528109083877e-01	err = 3.6530439817e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.528109083877e-01	err = 8.4131526813e-10	time = 0.01 sec
[ Info: CTMRG  15:	obj = +5.528109083877e-01	err = 1.9323742432e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +5.528109083877e-01	err = 4.3794080897e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.55e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.73e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.21e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.02e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 8.45e-07
└ * number of operations = 7
[ Info: LBFGS: iter   17, time  739.81 s: f = -0.658609918816, ‖∇f‖ = 4.5267e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.677351553934e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.676540650188e-01	err = 8.1121950466e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.676557801648e-01	err = 9.8508548940e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.676559337146e-01	err = 4.9071808894e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.676559435273e-01	err = 1.6289688930e-02	time = 0.24 sec
[ Info: CTMRG   5:	obj = +5.676559441271e-01	err = 1.6822148326e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.676559441634e-01	err = 2.2379838580e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.676559441656e-01	err = 2.5108531982e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.676559441658e-01	err = 2.5779331098e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.676559441658e-01	err = 3.1808574283e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.676559441658e-01	err = 8.9034900278e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.676559441658e-01	err = 2.2773476137e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.676559441658e-01	err = 5.5973679220e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.676559441658e-01	err = 1.3584034249e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.676559441658e-01	err = 3.3035634048e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.676559441658e-01	err = 7.9739519733e-11	time = 0.31 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.58e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.40e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.94e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 9.89e-07
└ * number of operations = 6
[ Info: LBFGS: iter   18, time  740.22 s: f = -0.659421361772, ‖∇f‖ = 4.8752e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.737978897057e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.739222975857e-01	err = 4.6101476264e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.739349062170e-01	err = 4.2215111228e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.739357219420e-01	err = 1.9564899301e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.739357726444e-01	err = 2.8765811504e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.739357757773e-01	err = 8.7017770956e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.739357759709e-01	err = 1.0215005459e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.739357759829e-01	err = 1.7793037040e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.739357759836e-01	err = 3.8457298884e-06	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.739357759837e-01	err = 1.0299620705e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.739357759837e-01	err = 2.7605838464e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.739357759837e-01	err = 7.0562738324e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.739357759837e-01	err = 1.7746967135e-08	time = 0.02 sec
[ Info: CTMRG  13:	obj = +5.739357759837e-01	err = 4.4366077410e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.739357759837e-01	err = 1.1064387907e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.739357759837e-01	err = 2.8054101760e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +5.739357759837e-01	err = 6.6982893607e-11	time = 0.12 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.62e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.43e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.36e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.62e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 9.75e-08
└ * number of operations = 7
[ Info: LBFGS: iter   19, time  740.43 s: f = -0.659584257676, ‖∇f‖ = 5.7745e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.745076718628e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.745504437966e-01	err = 2.6756870940e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.745540194585e-01	err = 2.8944679002e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.745542424652e-01	err = 7.7206198448e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.745542558356e-01	err = 2.7277946047e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.745542566313e-01	err = 4.4629488603e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.745542566786e-01	err = 5.3026651135e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.745542566814e-01	err = 6.1208775576e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.745542566816e-01	err = 1.7032705487e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.745542566816e-01	err = 4.7740572637e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.745542566816e-01	err = 1.2154636031e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.745542566816e-01	err = 3.0057748435e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.745542566816e-01	err = 7.3383713037e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.745542566816e-01	err = 1.8075494674e-09	time = 0.02 sec
[ Info: CTMRG  14:	obj = +5.745542566816e-01	err = 4.3690387741e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.745542566816e-01	err = 9.8198371912e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.54e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.13e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.56e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 7.53e-07
└ * number of operations = 6
[ Info: LBFGS: iter   20, time  740.59 s: f = -0.659811195031, ‖∇f‖ = 1.7740e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.752674236133e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.752694036899e-01	err = 2.1218296584e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.752695499956e-01	err = 1.3704207111e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.752695590184e-01	err = 4.5807953561e-03	time = 0.02 sec
[ Info: CTMRG   4:	obj = +5.752695595597e-01	err = 3.4238528365e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.752695595920e-01	err = 2.0224562220e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.752695595939e-01	err = 3.9865308042e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.752695595940e-01	err = 5.6760672361e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.752695595940e-01	err = 7.6538316722e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.752695595940e-01	err = 1.2463425262e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.752695595940e-01	err = 2.5216574089e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.752695595940e-01	err = 5.7812688632e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.752695595940e-01	err = 1.4258756369e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.752695595940e-01	err = 3.4716330364e-10	time = 0.00 sec
[ Info: CTMRG conv 14:	obj = +5.752695595940e-01	err = 8.6928184333e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.53e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.01e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.51e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 7.16e-07
└ * number of operations = 6
[ Info: LBFGS: iter   21, time  740.77 s: f = -0.659874427409, ‖∇f‖ = 1.4673e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.774023192899e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.774064958137e-01	err = 5.2667649352e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.774069388430e-01	err = 5.2960609639e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.774069652092e-01	err = 3.4923403633e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.774069667823e-01	err = 2.8750346775e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.774069668767e-01	err = 6.2186150563e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.774069668824e-01	err = 8.6136884852e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.774069668827e-01	err = 9.0441803580e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.774069668827e-01	err = 9.0103356074e-07	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.774069668827e-01	err = 1.1384324661e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.774069668827e-01	err = 2.8411595872e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.774069668827e-01	err = 7.0092464313e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.774069668827e-01	err = 1.7193878645e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.774069668827e-01	err = 4.2148593500e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.774069668827e-01	err = 1.0304832851e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.774069668827e-01	err = 2.5103947776e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.54e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.61e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.12e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.38e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 9.74e-08
└ * number of operations = 7
[ Info: LBFGS: iter   22, time  740.97 s: f = -0.660072570659, ‖∇f‖ = 1.9320e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.783110309028e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.783108255746e-01	err = 7.2443994591e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.783111423198e-01	err = 2.8860814323e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.783111595653e-01	err = 2.2443381270e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.783111605521e-01	err = 4.6360446117e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.783111606098e-01	err = 1.0052495730e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.783111606132e-01	err = 1.0433521403e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.783111606134e-01	err = 9.1319735408e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.783111606134e-01	err = 8.8028573318e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.783111606134e-01	err = 1.3151474683e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.783111606134e-01	err = 3.2396316084e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.783111606134e-01	err = 8.2149969187e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.783111606134e-01	err = 2.0034110301e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.783111606134e-01	err = 4.8663471437e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.783111606134e-01	err = 1.2033947986e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.783111606134e-01	err = 3.3750013034e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.58e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.37e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.44e-04
[ Info: BiCGStab linsolve in iteration 2: normres = 1.89e-05
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.48e-07
└ * number of operations = 7
[ Info: LBFGS: iter   23, time  741.18 s: f = -0.660232141902, ‖∇f‖ = 1.7545e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.774957758920e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.774964260001e-01	err = 6.6910362684e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.774976182560e-01	err = 2.5319186985e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.774976876573e-01	err = 2.1875537787e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.774976917736e-01	err = 5.9041750499e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.774976920214e-01	err = 1.7972192344e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.774976920364e-01	err = 2.2391149907e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.774976920373e-01	err = 2.4041293211e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.774976920373e-01	err = 2.6421033580e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.774976920373e-01	err = 5.2152551435e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.774976920373e-01	err = 1.3588440773e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.774976920373e-01	err = 3.2646794172e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.774976920373e-01	err = 8.0499742470e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.774976920373e-01	err = 1.9496547196e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.774976920373e-01	err = 4.8236662536e-10	time = 0.01 sec
[ Info: CTMRG  15:	obj = +5.774976920373e-01	err = 1.1222137722e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +5.774976920373e-01	err = 2.9079293043e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.71e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.87e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.83e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.03e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 5.73e-08
└ * number of operations = 7
[ Info: LBFGS: iter   24, time  741.38 s: f = -0.660380080163, ‖∇f‖ = 2.3752e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.749926971670e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.750450811480e-01	err = 7.7270285488e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.750491452719e-01	err = 3.8135605681e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.750493887424e-01	err = 1.2445130794e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.750494030845e-01	err = 4.8086169382e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.750494039262e-01	err = 1.3132102922e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.750494039755e-01	err = 2.5036781202e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.750494039784e-01	err = 2.1182037659e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.750494039785e-01	err = 7.5321688127e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.750494039786e-01	err = 8.6005660195e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.750494039786e-01	err = 3.1830412707e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.750494039786e-01	err = 5.2918435787e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.750494039786e-01	err = 1.6446625880e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.750494039786e-01	err = 3.3168386738e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.750494039786e-01	err = 9.1175729121e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.750494039786e-01	err = 2.0195940187e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +5.750494039786e-01	err = 5.1821331571e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.78e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 7.41e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.75e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 7.69e-07
└ * number of operations = 6
[ Info: LBFGS: iter   25, time  741.58 s: f = -0.660461052221, ‖∇f‖ = 2.3596e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.765747440600e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.765788384867e-01	err = 2.2853927761e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.765791823038e-01	err = 1.3335130319e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.765792030577e-01	err = 9.8433894609e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.765792042909e-01	err = 2.1037476909e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.765792043640e-01	err = 5.2126556905e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.765792043684e-01	err = 8.6168268881e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.765792043686e-01	err = 7.0162547794e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.765792043686e-01	err = 1.5688460809e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.765792043686e-01	err = 2.4306451964e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.765792043686e-01	err = 6.9216033883e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.765792043686e-01	err = 1.3025643496e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.765792043686e-01	err = 3.8027422852e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.765792043686e-01	err = 8.2564302200e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.765792043686e-01	err = 2.1918734256e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +5.765792043686e-01	err = 5.0773904127e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.74e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.82e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.90e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.10e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 4.40e-08
└ * number of operations = 7
[ Info: LBFGS: iter   26, time  741.79 s: f = -0.660554016679, ‖∇f‖ = 1.2681e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.757322983460e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.757343721418e-01	err = 2.1079992809e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.757345387839e-01	err = 1.0166458464e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.757345495986e-01	err = 7.3820816488e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.757345502703e-01	err = 8.9909591579e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.757345503116e-01	err = 2.7671963248e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.757345503142e-01	err = 3.6140774931e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.757345503143e-01	err = 6.3768792424e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.757345503143e-01	err = 9.4625711612e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.757345503143e-01	err = 2.5237583786e-07	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.757345503143e-01	err = 4.5947464379e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.757345503143e-01	err = 1.3110474418e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.757345503143e-01	err = 2.6577531679e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.757345503143e-01	err = 7.3094753444e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.757345503143e-01	err = 1.5998759171e-10	time = 0.01 sec
[ Info: CTMRG conv 15:	obj = +5.757345503143e-01	err = 4.2336499920e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.76e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.21e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.65e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.22e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 6.24e-08
└ * number of operations = 7
[ Info: LBFGS: iter   27, time  741.99 s: f = -0.660617092333, ‖∇f‖ = 1.0485e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.691631586134e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.691313758312e-01	err = 1.1127714797e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.691306598475e-01	err = 8.3633507268e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.691306081406e-01	err = 2.2760694213e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.691306044675e-01	err = 3.7238536282e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.691306042103e-01	err = 1.2331756614e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.691306041923e-01	err = 1.7901320419e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.691306041911e-01	err = 4.6453716163e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.691306041910e-01	err = 1.1374558993e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.691306041910e-01	err = 3.9978366268e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.691306041910e-01	err = 9.4661004369e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.691306041910e-01	err = 3.1688051985e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.691306041910e-01	err = 7.4572919635e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.691306041910e-01	err = 2.4809135180e-08	time = 0.01 sec
[ Info: CTMRG  14:	obj = +5.691306041910e-01	err = 5.8399514545e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.691306041910e-01	err = 1.9392756033e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.691306041910e-01	err = 4.5672920180e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.691306041910e-01	err = 1.5144960849e-10	time = 0.01 sec
[ Info: CTMRG conv 18:	obj = +5.691306041910e-01	err = 3.5696979483e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.93e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 7.80e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.71e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 9.51e-07
└ * number of operations = 6
[ Info: LBFGS: iter   28, time  742.20 s: f = -0.660813477825, ‖∇f‖ = 1.7986e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.591990954908e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.591610169535e-01	err = 5.4879295071e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.591609825010e-01	err = 2.6625147412e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.591609804415e-01	err = 1.4017124921e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.591609802496e-01	err = 1.2598553282e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.591609802342e-01	err = 1.8820040724e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.591609802330e-01	err = 2.7096671897e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.591609802329e-01	err = 5.7667718937e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.591609802329e-01	err = 1.0559269066e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.591609802329e-01	err = 4.2181941011e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.591609802329e-01	err = 1.0502109284e-06	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.591609802329e-01	err = 3.5469854819e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.591609802329e-01	err = 9.5877101239e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.591609802329e-01	err = 3.1194933559e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.591609802329e-01	err = 8.6658950684e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.591609802329e-01	err = 2.7756072257e-09	time = 0.01 sec
[ Info: CTMRG  16:	obj = +5.591609802329e-01	err = 7.8340669087e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.591609802329e-01	err = 2.4783678024e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +5.591609802329e-01	err = 7.0849178395e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.95e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.52e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.29e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.58e-06
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.60e-06
┌ Info: BiCGStab linsolve converged at iteration 3:
│ * norm of residual = 2.83e-07
└ * number of operations = 8
[ Info: LBFGS: iter   29, time  742.71 s: f = -0.660960969686, ‖∇f‖ = 1.7471e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.539558878304e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.539752091369e-01	err = 4.1470294142e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.539792510442e-01	err = 4.7847540257e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.539796111926e-01	err = 1.1555918689e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.539796417725e-01	err = 4.1461570076e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.539796443371e-01	err = 1.2084325931e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.539796445513e-01	err = 1.9632555807e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.539796445691e-01	err = 3.6393072328e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.539796445706e-01	err = 6.6099892800e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.539796445708e-01	err = 1.7934472006e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.539796445708e-01	err = 4.7538083772e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.539796445708e-01	err = 1.3049298279e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.539796445708e-01	err = 3.9479295853e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.539796445708e-01	err = 1.0139677357e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.539796445708e-01	err = 3.3404786037e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.539796445708e-01	err = 7.9259193557e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.539796445708e-01	err = 2.8292805329e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +5.539796445708e-01	err = 6.2037879550e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.90e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.41e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.06e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.03e-06
[ Info: BiCGStab linsolve in iteration 2.5: normres = 1.11e-06
┌ Info: BiCGStab linsolve converged at iteration 3:
│ * norm of residual = 1.12e-08
└ * number of operations = 8
[ Info: LBFGS: iter   30, time  742.95 s: f = -0.661039077160, ‖∇f‖ = 1.1401e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.543832099958e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.543792183643e-01	err = 3.7883656995e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.543790438941e-01	err = 2.6898809478e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.543790313905e-01	err = 1.0538275064e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.543790303854e-01	err = 2.3573598711e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.543790303045e-01	err = 4.1273066505e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.543790302980e-01	err = 7.7637788890e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.543790302975e-01	err = 2.4626010034e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.543790302975e-01	err = 6.1769076461e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.543790302975e-01	err = 2.2693899191e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.543790302975e-01	err = 5.6638817605e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.543790302975e-01	err = 2.0553265137e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.543790302975e-01	err = 5.2998319426e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.543790302975e-01	err = 1.9025272420e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.543790302975e-01	err = 5.0219617993e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.543790302975e-01	err = 1.7921504633e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.543790302975e-01	err = 4.8030813529e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.543790302975e-01	err = 1.7096037135e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +5.543790302975e-01	err = 4.6283162910e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.90e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.10e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 7.90e-06
[ Info: BiCGStab linsolve in iteration 2: normres = 1.30e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 4.48e-07
└ * number of operations = 7
[ Info: LBFGS: iter   31, time  743.15 s: f = -0.661087806652, ‖∇f‖ = 1.0339e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.519543985810e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.519518927726e-01	err = 1.3395954187e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.519518865043e-01	err = 1.3634648983e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.519518868851e-01	err = 6.5563096965e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.519518869269e-01	err = 2.1108600340e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.519518869307e-01	err = 5.2480763606e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.519518869310e-01	err = 7.0692167650e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.519518869310e-01	err = 1.3219974578e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.519518869310e-01	err = 1.9470385022e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.519518869310e-01	err = 5.5788700872e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.519518869310e-01	err = 1.9802755960e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.519518869310e-01	err = 5.0593860126e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.519518869310e-01	err = 1.9270770547e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.519518869310e-01	err = 5.0349407772e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.519518869310e-01	err = 1.8801447845e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.519518869310e-01	err = 5.0040090660e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.519518869310e-01	err = 1.8410365012e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +5.519518869310e-01	err = 4.9569737079e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.91e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.83e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 9.73e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 5.90e-07
└ * number of operations = 6
[ Info: LBFGS: iter   32, time  743.35 s: f = -0.661121452359, ‖∇f‖ = 8.8764e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.435912005696e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.436007882889e-01	err = 3.0325945688e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.436033225325e-01	err = 4.0530949330e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.436035780366e-01	err = 1.1462831550e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.436036025205e-01	err = 8.7060100692e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.436036048348e-01	err = 2.7786670383e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.436036050524e-01	err = 4.9328772754e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.436036050728e-01	err = 6.6855650971e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.436036050747e-01	err = 9.6809505522e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.436036050749e-01	err = 1.5748130724e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.436036050749e-01	err = 4.1842173186e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.436036050749e-01	err = 1.2902906913e-07	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.436036050749e-01	err = 3.7976395576e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.436036050749e-01	err = 1.2130088618e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.436036050749e-01	err = 3.3624163166e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.436036050749e-01	err = 1.1552460156e-09	time = 0.01 sec
[ Info: CTMRG  16:	obj = +5.436036050749e-01	err = 3.1073672437e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.436036050749e-01	err = 1.0997143024e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +5.436036050749e-01	err = 2.9784875217e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.94e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.35e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.28e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 3.19e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 6.22e-08
└ * number of operations = 7
[ Info: LBFGS: iter   33, time  743.57 s: f = -0.661180968072, ‖∇f‖ = 1.0798e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.439225121131e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.439193276537e-01	err = 3.8824424609e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.439202291941e-01	err = 1.6593452375e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.439203158928e-01	err = 5.3340807460e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.439203241746e-01	err = 2.7838671876e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.439203249619e-01	err = 7.2572765319e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.439203250366e-01	err = 1.6885253072e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.439203250437e-01	err = 3.2900532985e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.439203250443e-01	err = 6.5666193756e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.439203250444e-01	err = 2.3797750506e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.439203250444e-01	err = 6.2761973516e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.439203250444e-01	err = 2.3097295090e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.439203250444e-01	err = 5.7123356250e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.439203250444e-01	err = 2.0798630020e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.439203250444e-01	err = 5.0929076578e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.439203250444e-01	err = 1.8485408848e-09	time = 0.01 sec
[ Info: CTMRG  16:	obj = +5.439203250444e-01	err = 4.5054917151e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.439203250444e-01	err = 1.6333204943e-10	time = 0.01 sec
[ Info: CTMRG conv 18:	obj = +5.439203250444e-01	err = 3.9696609234e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.95e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.12e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.85e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.57e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 5.37e-08
└ * number of operations = 7
[ Info: CTMRG init:	obj = +5.438718833136e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.438710974166e-01	err = 2.6162492611e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.438713361071e-01	err = 8.4657179009e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.438713587572e-01	err = 3.0378244946e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.438713608992e-01	err = 1.2098309665e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.438713611009e-01	err = 5.5922136679e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.438713611198e-01	err = 8.6609309395e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.438713611216e-01	err = 1.7685720841e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.438713611218e-01	err = 4.4876625855e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.438713611218e-01	err = 1.2542553505e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.438713611218e-01	err = 3.8070769294e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.438713611218e-01	err = 1.0921906499e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.438713611218e-01	err = 3.3983804551e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.438713611218e-01	err = 9.6871498319e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.438713611218e-01	err = 3.0340630051e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.438713611218e-01	err = 8.5878984189e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.438713611218e-01	err = 2.7010186529e-10	time = 0.01 sec
[ Info: CTMRG conv 17:	obj = +5.438713611218e-01	err = 7.5993244513e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.94e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.76e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.93e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.81e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 4.28e-08
└ * number of operations = 7
[ Info: LBFGS: iter   34, time  744.01 s: f = -0.661206863491, ‖∇f‖ = 9.1298e-03, α = 5.16e-01, m = 16, nfg = 2
[ Info: CTMRG init:	obj = +5.441731369005e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.441718411263e-01	err = 6.7685026096e-03	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.441718309851e-01	err = 3.8979572503e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.441718304264e-01	err = 2.7832993020e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.441718303786e-01	err = 3.5203612978e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.441718303742e-01	err = 1.1460341716e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.441718303738e-01	err = 3.6608408369e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.441718303738e-01	err = 5.5583878613e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.441718303738e-01	err = 1.9626085459e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.441718303738e-01	err = 5.2070037428e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.441718303738e-01	err = 1.6404286084e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.441718303738e-01	err = 4.8993730928e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.441718303738e-01	err = 1.4795669704e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.441718303738e-01	err = 4.4733946164e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.441718303738e-01	err = 1.3563707019e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.441718303738e-01	err = 4.0974489918e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.441718303738e-01	err = 1.2511107466e-10	time = 0.01 sec
[ Info: CTMRG conv 17:	obj = +5.441718303738e-01	err = 3.7661518747e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.95e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.45e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.90e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.89e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 7.40e-08
└ * number of operations = 7
[ Info: LBFGS: iter   35, time  744.23 s: f = -0.661226335408, ‖∇f‖ = 6.6169e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.402058126435e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.402058667336e-01	err = 1.3736911282e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.402056323671e-01	err = 1.9367344047e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.402056082882e-01	err = 6.9372113248e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.402056059275e-01	err = 4.1933668480e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.402056056985e-01	err = 1.2150256164e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.402056056764e-01	err = 2.6340454902e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.402056056743e-01	err = 3.6495057334e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.402056056741e-01	err = 8.1080371742e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.402056056741e-01	err = 1.9598109735e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.402056056741e-01	err = 5.9412479992e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.402056056741e-01	err = 1.6809402337e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.402056056741e-01	err = 5.0133967112e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +5.402056056741e-01	err = 1.4566593242e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.402056056741e-01	err = 4.3159068215e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.402056056741e-01	err = 1.2684921746e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.402056056741e-01	err = 3.7436597894e-10	time = 0.01 sec
[ Info: CTMRG  17:	obj = +5.402056056741e-01	err = 1.1109468648e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +5.402056056741e-01	err = 3.2694659242e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.97e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.52e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.60e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 8.52e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.36e-07
└ * number of operations = 7
[ Info: LBFGS: iter   36, time  744.46 s: f = -0.661260012209, ‖∇f‖ = 5.9848e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.401725794020e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.401714569933e-01	err = 1.5858490766e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.401713229617e-01	err = 1.1887922679e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.401713082865e-01	err = 5.8941340539e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.401713067984e-01	err = 2.3609052771e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.401713066517e-01	err = 4.5579772643e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.401713066375e-01	err = 7.4411336242e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.401713066361e-01	err = 2.9212141435e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.401713066359e-01	err = 4.7712606710e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.401713066359e-01	err = 1.3660234248e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.401713066359e-01	err = 3.8297770309e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.401713066359e-01	err = 9.8707294969e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.401713066359e-01	err = 3.2189981170e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.401713066359e-01	err = 8.6089099021e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.401713066359e-01	err = 2.7400151227e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.401713066359e-01	err = 7.4412132300e-10	time = 0.01 sec
[ Info: CTMRG  16:	obj = +5.401713066359e-01	err = 2.3514539335e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +5.401713066359e-01	err = 6.4308071272e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.96e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.28e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.22e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 6.84e-07
└ * number of operations = 6
[ Info: LBFGS: iter   37, time  744.67 s: f = -0.661268668988, ‖∇f‖ = 1.0826e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.387272115400e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.387280402148e-01	err = 6.8278088737e-03	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.387281861458e-01	err = 5.4261176243e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.387282009746e-01	err = 3.5477831519e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.387282024107e-01	err = 1.2815961301e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.387282025478e-01	err = 4.1247266285e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.387282025609e-01	err = 7.8780023012e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.387282025621e-01	err = 1.5191343931e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.387282025622e-01	err = 1.6849366862e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.387282025622e-01	err = 5.8112364784e-07	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.387282025622e-01	err = 1.0548979422e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.387282025622e-01	err = 3.6985392527e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.387282025622e-01	err = 8.1728765595e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.387282025622e-01	err = 2.8468727637e-09	time = 0.01 sec
[ Info: CTMRG  14:	obj = +5.387282025622e-01	err = 6.5380497724e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.387282025622e-01	err = 2.2799144818e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +5.387282025622e-01	err = 5.3067244057e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.97e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.40e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 7.40e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 3.26e-07
└ * number of operations = 6
[ Info: LBFGS: iter   38, time  744.88 s: f = -0.661283178602, ‖∇f‖ = 5.0739e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.369964811744e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.369958426469e-01	err = 9.4665859289e-03	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.369958232411e-01	err = 7.2919910161e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.369958212074e-01	err = 2.6813496126e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.369958209935e-01	err = 1.5600120967e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.369958209719e-01	err = 4.8430330270e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.369958209697e-01	err = 9.5587029525e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.369958209695e-01	err = 2.1059613530e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.369958209695e-01	err = 2.4345077718e-06	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.369958209695e-01	err = 6.8888816326e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.369958209694e-01	err = 1.3341897127e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.369958209694e-01	err = 3.6052046390e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.369958209694e-01	err = 9.6155249588e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.369958209694e-01	err = 2.7046972936e-09	time = 0.01 sec
[ Info: CTMRG  14:	obj = +5.369958209694e-01	err = 7.2109040450e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.369958209694e-01	err = 2.1438212277e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +5.369958209694e-01	err = 5.5378076280e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 1.98e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.34e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 9.34e-06
[ Info: BiCGStab linsolve in iteration 2: normres = 1.10e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 3.93e-07
└ * number of operations = 7
[ Info: LBFGS: iter   39, time  745.09 s: f = -0.661293239500, ‖∇f‖ = 4.8729e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.354449621486e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.354428703136e-01	err = 1.3758626843e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.354426286569e-01	err = 9.4531308920e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.354426026719e-01	err = 4.1268435550e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.354426000294e-01	err = 1.6904169178e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.354425997668e-01	err = 5.6495328902e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.354425997410e-01	err = 1.2406061781e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.354425997385e-01	err = 3.2896972597e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.354425997382e-01	err = 4.5159923684e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.354425997382e-01	err = 9.4300150796e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.354425997382e-01	err = 1.5743296513e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.354425997382e-01	err = 3.6347007617e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.354425997382e-01	err = 9.5386751694e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.354425997382e-01	err = 2.3105075215e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.354425997382e-01	err = 6.7328852738e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.354425997382e-01	err = 1.7390418021e-10	time = 0.01 sec
[ Info: CTMRG conv 16:	obj = +5.354425997382e-01	err = 5.1797540885e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.00e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.60e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.03e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.64e-07
└ * number of operations = 6
[ Info: LBFGS: iter   40, time  745.30 s: f = -0.661307958912, ‖∇f‖ = 6.2349e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.313300515273e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.313227231700e-01	err = 4.4955824128e-02	time = 0.05 sec
[ Info: CTMRG   2:	obj = +5.313214982545e-01	err = 3.5216830098e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.313213594837e-01	err = 1.0313978214e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.313213449633e-01	err = 4.6653180367e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.313213434831e-01	err = 2.7519312188e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.313213433338e-01	err = 6.2471105040e-04	time = 0.24 sec
[ Info: CTMRG   7:	obj = +5.313213433188e-01	err = 8.9044402819e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.313213433173e-01	err = 1.3219042603e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.313213433172e-01	err = 1.3218961535e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.313213433171e-01	err = 2.0252425310e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.313213433171e-01	err = 3.0592872243e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.313213433171e-01	err = 1.0430527544e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.313213433171e-01	err = 3.2804517865e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.313213433171e-01	err = 1.0599478835e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.313213433171e-01	err = 3.3601729140e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.313213433171e-01	err = 1.0719310492e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +5.313213433171e-01	err = 3.3958179369e-11	time = 0.36 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.02e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.87e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.17e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 8.28e-07
└ * number of operations = 6
[ Info: LBFGS: iter   41, time  745.78 s: f = -0.661342970541, ‖∇f‖ = 9.2762e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.211957382243e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.211762625113e-01	err = 1.1020646844e-01	time = 0.01 sec
[ Info: CTMRG   2:	obj = +5.211699611963e-01	err = 7.4481850072e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.211691766127e-01	err = 2.8333498724e-02	time = 0.01 sec
[ Info: CTMRG   4:	obj = +5.211690886423e-01	err = 1.3446085947e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +5.211690790472e-01	err = 5.3309638064e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +5.211690780121e-01	err = 1.4211663296e-03	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.211690779010e-01	err = 2.7551066064e-04	time = 0.01 sec
[ Info: CTMRG   8:	obj = +5.211690778891e-01	err = 4.2886745899e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.211690778878e-01	err = 6.7351499237e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.211690778877e-01	err = 1.1402719805e-06	time = 0.01 sec
[ Info: CTMRG  11:	obj = +5.211690778877e-01	err = 2.4387962316e-07	time = 0.02 sec
[ Info: CTMRG  12:	obj = +5.211690778877e-01	err = 6.3206588382e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.211690778877e-01	err = 2.0797435605e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.211690778877e-01	err = 6.8305846215e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.211690778877e-01	err = 2.2373264449e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.211690778877e-01	err = 7.3217787642e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.211690778877e-01	err = 2.3951036344e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +5.211690778877e-01	err = 7.8329627217e-11	time = 0.12 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.15e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.65e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.51e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 3.07e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 8.91e-07
└ * number of operations = 7
[ Info: LBFGS: iter   42, time  746.01 s: f = -0.661417237192, ‖∇f‖ = 1.7461e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.043151425320e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.043555440692e-01	err = 2.0606605946e-01	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.043465249374e-01	err = 8.1521389483e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.043451858294e-01	err = 2.1121628078e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.043450181480e-01	err = 2.0283629507e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.043449977818e-01	err = 7.2215239307e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.043449953361e-01	err = 2.1237200881e-03	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.043449950439e-01	err = 3.9825944576e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.043449950090e-01	err = 6.9297647607e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +5.043449950049e-01	err = 1.1901554409e-05	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.043449950044e-01	err = 2.2630675674e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.043449950043e-01	err = 5.6906075100e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.043449950043e-01	err = 2.0419831879e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.043449950043e-01	err = 7.1495393819e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.043449950043e-01	err = 2.4787306425e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.043449950043e-01	err = 8.5680098104e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.043449950043e-01	err = 2.9580859866e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.043449950043e-01	err = 1.0207665562e-09	time = 0.00 sec
[ Info: CTMRG  18:	obj = +5.043449950043e-01	err = 3.5215445851e-10	time = 0.00 sec
[ Info: CTMRG  19:	obj = +5.043449950043e-01	err = 1.2149427622e-10	time = 0.00 sec
[ Info: CTMRG conv 20:	obj = +5.043449950043e-01	err = 4.1898392519e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.25e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.54e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.78e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 4.41e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.85e-07
└ * number of operations = 7
[ Info: LBFGS: iter   43, time  746.22 s: f = -0.661494994773, ‖∇f‖ = 2.7924e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.027450069913e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.027378701297e-01	err = 5.1094487540e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.027370044925e-01	err = 2.0610784853e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +5.027369010069e-01	err = 8.9218148869e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.027368888989e-01	err = 5.8962347694e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.027368874749e-01	err = 1.7424506238e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.027368873071e-01	err = 3.6499390894e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +5.027368872873e-01	err = 5.9104469886e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.027368872850e-01	err = 1.0134500875e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.027368872847e-01	err = 2.6327575699e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +5.027368872847e-01	err = 7.7928412920e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.027368872847e-01	err = 2.5620774270e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +5.027368872847e-01	err = 8.6761876916e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.027368872847e-01	err = 2.9808928092e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.027368872847e-01	err = 1.0273657402e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.027368872847e-01	err = 3.5438348184e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +5.027368872847e-01	err = 1.2221827880e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.027368872847e-01	err = 4.2139137129e-10	time = 0.00 sec
[ Info: CTMRG  18:	obj = +5.027368872847e-01	err = 1.4519064354e-10	time = 0.00 sec
[ Info: CTMRG conv 19:	obj = +5.027368872847e-01	err = 5.0033079164e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.27e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 9.03e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.65e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 3.29e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.45e-07
└ * number of operations = 7
[ Info: LBFGS: iter   44, time  746.42 s: f = -0.661665032445, ‖∇f‖ = 2.1193e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.997259388265e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.997434091165e-01	err = 2.1319379898e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.997424394894e-01	err = 3.1440794778e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.997423047248e-01	err = 8.9566756650e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.997422872813e-01	err = 7.8488341655e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.997422850616e-01	err = 1.5740316196e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.997422847814e-01	err = 1.7287364053e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.997422847462e-01	err = 3.5399981148e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.997422847418e-01	err = 7.5354269222e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.997422847413e-01	err = 1.7472653580e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.997422847412e-01	err = 4.7902544415e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.997422847412e-01	err = 1.5276794453e-07	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.997422847412e-01	err = 5.3831290570e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.997422847412e-01	err = 1.9059392235e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.997422847412e-01	err = 6.7681376868e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.997422847412e-01	err = 2.4084368673e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.997422847412e-01	err = 8.5816732731e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.997422847412e-01	err = 3.0600090431e-10	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.997422847412e-01	err = 1.0917235381e-10	time = 0.01 sec
[ Info: CTMRG conv 19:	obj = +4.997422847412e-01	err = 3.8984024490e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.22e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 7.73e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.33e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.04e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.93e-07
└ * number of operations = 7
[ Info: LBFGS: iter   45, time  746.63 s: f = -0.661840699875, ‖∇f‖ = 2.3535e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.912350354529e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.912801591534e-01	err = 4.9201443171e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.912817703942e-01	err = 9.4018965645e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.912819658265e-01	err = 1.3882058341e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.912819946837e-01	err = 2.2253187793e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.912819989460e-01	err = 5.2173976273e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.912819995745e-01	err = 8.4145188946e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.912819996671e-01	err = 1.1678436587e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.912819996808e-01	err = 2.1299738318e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.912819996829e-01	err = 3.8322897521e-06	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.912819996832e-01	err = 8.3512527203e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.912819996832e-01	err = 2.2574105482e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.912819996832e-01	err = 7.7895221879e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.912819996832e-01	err = 2.9696038140e-08	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.912819996832e-01	err = 1.1434886267e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.912819996832e-01	err = 4.4168598480e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.912819996832e-01	err = 1.7079986063e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.912819996832e-01	err = 6.6070850458e-10	time = 0.01 sec
[ Info: CTMRG  18:	obj = +4.912819996832e-01	err = 2.5565601745e-10	time = 0.00 sec
[ Info: CTMRG conv 19:	obj = +4.912819996832e-01	err = 9.8917416004e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.25e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.46e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.16e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.47e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.85e-07
└ * number of operations = 7
[ Info: LBFGS: iter   46, time  746.84 s: f = -0.661983211854, ‖∇f‖ = 2.0945e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +5.025330516247e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +5.025013161656e-01	err = 3.7262956909e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +5.024982636995e-01	err = 5.9104360401e-02	time = 0.01 sec
[ Info: CTMRG   3:	obj = +5.024978015183e-01	err = 1.9898674105e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +5.024977361778e-01	err = 2.0977057198e-02	time = 0.00 sec
[ Info: CTMRG   5:	obj = +5.024977271904e-01	err = 4.8084494692e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +5.024977259681e-01	err = 5.5766592047e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +5.024977258026e-01	err = 6.5443767084e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +5.024977257802e-01	err = 1.0298289364e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +5.024977257772e-01	err = 2.5804937220e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +5.024977257768e-01	err = 8.1137457236e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +5.024977257768e-01	err = 2.8719120710e-07	time = 0.01 sec
[ Info: CTMRG  12:	obj = +5.024977257767e-01	err = 1.0504451781e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +5.024977257767e-01	err = 3.8941462508e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +5.024977257767e-01	err = 1.4491179912e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +5.024977257767e-01	err = 5.4009794604e-09	time = 0.01 sec
[ Info: CTMRG  16:	obj = +5.024977257767e-01	err = 2.0140266561e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +5.024977257767e-01	err = 7.5124019201e-10	time = 0.00 sec
[ Info: CTMRG  18:	obj = +5.024977257767e-01	err = 2.8023658102e-10	time = 0.00 sec
[ Info: CTMRG  19:	obj = +5.024977257767e-01	err = 1.0453510150e-10	time = 0.01 sec
[ Info: CTMRG conv 20:	obj = +5.024977257767e-01	err = 3.9037482248e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.40e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 9.88e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.57e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.17e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.83e-07
└ * number of operations = 7
[ Info: LBFGS: iter   47, time  747.06 s: f = -0.662069016591, ‖∇f‖ = 1.9084e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.864801008385e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.866290742665e-01	err = 4.9250010918e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.866281147451e-01	err = 6.7595039472e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.866277796217e-01	err = 2.2923353155e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.866277230508e-01	err = 2.6443022022e-02	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.866277139855e-01	err = 7.3747801331e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.866277125523e-01	err = 1.4227474364e-03	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.866277123270e-01	err = 2.2988468978e-04	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.866277122917e-01	err = 3.7643953857e-05	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.866277122861e-01	err = 6.5965927946e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.866277122853e-01	err = 1.2085745198e-06	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.866277122851e-01	err = 2.8540650146e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.866277122851e-01	err = 1.2829494096e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.866277122851e-01	err = 5.4456178407e-08	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.866277122851e-01	err = 2.2285903244e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.866277122851e-01	err = 8.9850902997e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.866277122851e-01	err = 3.6010095035e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.866277122851e-01	err = 1.4399030956e-09	time = 0.01 sec
[ Info: CTMRG  18:	obj = +4.866277122851e-01	err = 5.7521663851e-10	time = 0.00 sec
[ Info: CTMRG  19:	obj = +4.866277122851e-01	err = 2.2974368228e-10	time = 0.00 sec
[ Info: CTMRG conv 20:	obj = +4.866277122851e-01	err = 9.1731499957e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.43e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.03e-04
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.02e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.13e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 3.30e-07
└ * number of operations = 7
[ Info: LBFGS: iter   48, time  747.27 s: f = -0.662233515834, ‖∇f‖ = 1.7527e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.865762026415e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.865552131075e-01	err = 2.5099336574e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.865543433734e-01	err = 1.6187632893e-02	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.865541812568e-01	err = 1.2454578674e-02	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.865541532339e-01	err = 2.6180836383e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.865541485675e-01	err = 1.3770069427e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.865541478018e-01	err = 3.0527243762e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.865541476768e-01	err = 6.5580721151e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.865541476564e-01	err = 1.2631547800e-05	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.865541476531e-01	err = 2.4491119457e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.865541476526e-01	err = 6.0792872105e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.865541476525e-01	err = 2.4500096860e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.865541476525e-01	err = 1.0074365221e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.865541476525e-01	err = 4.1130823135e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.865541476525e-01	err = 1.6694836785e-08	time = 0.01 sec
[ Info: CTMRG  15:	obj = +4.865541476525e-01	err = 6.7564919172e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.865541476525e-01	err = 2.7308124124e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.865541476525e-01	err = 1.1031092630e-09	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.865541476525e-01	err = 4.4549134178e-10	time = 0.01 sec
[ Info: CTMRG  19:	obj = +4.865541476525e-01	err = 1.7989576062e-10	time = 0.00 sec
[ Info: CTMRG conv 20:	obj = +4.865541476525e-01	err = 7.2639870177e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.39e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.75e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.32e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.39e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.83e-07
└ * number of operations = 7
[ Info: CTMRG init:	obj = +4.867581560856e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.867519001252e-01	err = 1.2951453645e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.867516101978e-01	err = 9.1993381187e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.867515568795e-01	err = 6.2073431202e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.867515477783e-01	err = 1.3390206492e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.867515462749e-01	err = 6.3459200659e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.867515460297e-01	err = 1.2048535350e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.867515459899e-01	err = 2.6264484414e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.867515459835e-01	err = 4.9481206357e-06	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.867515459824e-01	err = 1.0106694922e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.867515459823e-01	err = 3.1666024681e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.867515459822e-01	err = 1.3614558734e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.867515459822e-01	err = 5.6526890923e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.867515459822e-01	err = 2.3047740445e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.867515459822e-01	err = 9.3186500948e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.867515459822e-01	err = 3.7542292065e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.867515459822e-01	err = 1.5101684939e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.867515459822e-01	err = 6.0707054414e-10	time = 0.01 sec
[ Info: CTMRG  18:	obj = +4.867515459822e-01	err = 2.4398323899e-10	time = 0.00 sec
[ Info: CTMRG conv 19:	obj = +4.867515459822e-01	err = 9.8042421643e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.40e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.02e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.72e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.78e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.56e-07
└ * number of operations = 7
[ Info: LBFGS: iter   49, time  747.70 s: f = -0.662332106545, ‖∇f‖ = 1.6912e-02, α = 5.23e-01, m = 16, nfg = 2
[ Info: CTMRG init:	obj = +4.862950588272e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.862840325625e-01	err = 9.1042291527e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.862839806779e-01	err = 6.6410828656e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.862839797666e-01	err = 6.8678084765e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.862839797727e-01	err = 2.6672908191e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.862839797791e-01	err = 7.2026531035e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.862839797805e-01	err = 1.3225164977e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.862839797808e-01	err = 2.0312511868e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.862839797808e-01	err = 3.5372990652e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.862839797808e-01	err = 5.5125618545e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.862839797808e-01	err = 2.1581964431e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.862839797808e-01	err = 9.4072818763e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.862839797808e-01	err = 3.9343276160e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.862839797808e-01	err = 1.6104078487e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.862839797808e-01	err = 6.5346722834e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.862839797808e-01	err = 2.6423533830e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.862839797808e-01	err = 1.0668463153e-09	time = 0.01 sec
[ Info: CTMRG  17:	obj = +4.862839797808e-01	err = 4.3042610028e-10	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.862839797808e-01	err = 1.7360600259e-10	time = 0.00 sec
[ Info: CTMRG conv 19:	obj = +4.862839797808e-01	err = 7.0003540153e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.42e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.10e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 4.19e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.30e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.76e-07
└ * number of operations = 7
[ Info: LBFGS: iter   50, time  747.91 s: f = -0.662397705763, ‖∇f‖ = 1.0361e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.893824468396e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.893695986099e-01	err = 1.1390621318e-02	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.893690571249e-01	err = 1.3380894274e-02	time = 0.05 sec
[ Info: CTMRG   3:	obj = +4.893689684468e-01	err = 5.4894625530e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.893689539309e-01	err = 7.8963541007e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.893689515958e-01	err = 1.7736394097e-03	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.893689512234e-01	err = 3.0241836955e-04	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.893689511642e-01	err = 4.8275441414e-05	time = 0.24 sec
[ Info: CTMRG   8:	obj = +4.893689511548e-01	err = 8.2763977483e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.893689511533e-01	err = 1.2655865883e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.893689511530e-01	err = 1.7864090952e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.893689511530e-01	err = 4.9868465607e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.893689511530e-01	err = 1.9670269269e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.893689511530e-01	err = 8.0856259813e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.893689511530e-01	err = 3.2837613747e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.893689511530e-01	err = 1.3220813178e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.893689511530e-01	err = 5.3021149954e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.893689511530e-01	err = 2.1230558278e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.893689511530e-01	err = 8.4959096968e-11	time = 0.36 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.80e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.78e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.31e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 2.42e-07
└ * number of operations = 7
[ Info: LBFGS: iter   51, time  748.41 s: f = -0.662434512495, ‖∇f‖ = 8.3812e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.873344619303e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.873334243454e-01	err = 1.1099105959e-02	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.873335983033e-01	err = 8.6096569291e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.873336272395e-01	err = 2.6157767972e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.873336320352e-01	err = 4.1745171164e-03	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.873336328165e-01	err = 1.1163254230e-03	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.873336329429e-01	err = 2.0388090381e-04	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.873336329633e-01	err = 3.6754193314e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.873336329666e-01	err = 6.8538024408e-06	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.873336329672e-01	err = 1.1628953574e-06	time = 0.02 sec
[ Info: CTMRG  10:	obj = +4.873336329672e-01	err = 2.1627590876e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.873336329673e-01	err = 7.3713651650e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.873336329673e-01	err = 3.0177412450e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.873336329673e-01	err = 1.2273757232e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.873336329673e-01	err = 4.9615562975e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.873336329673e-01	err = 1.9987296371e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.873336329673e-01	err = 8.0378253946e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.873336329673e-01	err = 3.2293571004e-10	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.873336329673e-01	err = 1.2967864134e-10	time = 0.00 sec
[ Info: CTMRG conv 19:	obj = +4.873336329673e-01	err = 5.2056103107e-11	time = 0.12 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.46e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 6.12e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.56e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 4.11e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.75e-07
└ * number of operations = 7
[ Info: LBFGS: iter   52, time  748.64 s: f = -0.662459914679, ‖∇f‖ = 5.8461e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.881417718274e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.881483950633e-01	err = 7.7310558650e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.881497700741e-01	err = 4.1319082935e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.881500114941e-01	err = 4.5310278701e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.881500510368e-01	err = 3.0184843576e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.881500573751e-01	err = 9.4074464583e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.881500583830e-01	err = 2.0584326057e-04	time = 0.02 sec
[ Info: CTMRG   7:	obj = +4.881500585428e-01	err = 4.1745001617e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.881500585681e-01	err = 9.1099662777e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.881500585721e-01	err = 2.4020935540e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.881500585727e-01	err = 7.6979464331e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.881500585728e-01	err = 2.9823699018e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.881500585728e-01	err = 1.1924474760e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.881500585728e-01	err = 4.7569922080e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.881500585728e-01	err = 1.8950455035e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.881500585728e-01	err = 7.5428339398e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.881500585728e-01	err = 3.0006434320e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.881500585728e-01	err = 1.1932906845e-09	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.881500585728e-01	err = 4.7443931848e-10	time = 0.00 sec
[ Info: CTMRG  19:	obj = +4.881500585728e-01	err = 1.8861717720e-10	time = 0.00 sec
[ Info: CTMRG conv 20:	obj = +4.881500585728e-01	err = 7.4967997746e-11	time = 0.10 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.46e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.29e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.23e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.14e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 6.94e-08
└ * number of operations = 7
[ Info: LBFGS: iter   53, time  748.85 s: f = -0.662475090799, ‖∇f‖ = 1.1955e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.877652814782e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.877728053729e-01	err = 3.8203207689e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.877744730022e-01	err = 2.7511456268e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.877747604877e-01	err = 8.8932653684e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.877748072897e-01	err = 4.0671907071e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.877748147618e-01	err = 9.5452965796e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.877748159459e-01	err = 2.5790260975e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.877748161331e-01	err = 9.5666705065e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.877748161626e-01	err = 3.9937545292e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.877748161673e-01	err = 1.6485960408e-06	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.877748161680e-01	err = 6.6740022927e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.877748161681e-01	err = 2.6831715970e-07	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.877748161681e-01	err = 1.0727982803e-07	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.877748161681e-01	err = 4.2774024155e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.877748161681e-01	err = 1.7022782053e-08	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.877748161681e-01	err = 6.7671325669e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.877748161681e-01	err = 2.6882700423e-09	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.877748161681e-01	err = 1.0674561471e-09	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.877748161681e-01	err = 4.2374832168e-10	time = 0.00 sec
[ Info: CTMRG  19:	obj = +4.877748161681e-01	err = 1.6818826336e-10	time = 0.00 sec
[ Info: CTMRG conv 20:	obj = +4.877748161681e-01	err = 6.6746769170e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.46e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.00e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 7.46e-06
[ Info: BiCGStab linsolve in iteration 2: normres = 1.49e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 6.68e-08
└ * number of operations = 7
[ Info: LBFGS: iter   54, time  749.07 s: f = -0.662490551714, ‖∇f‖ = 4.3803e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870317675865e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870319253809e-01	err = 1.6664736676e-03	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.870319134703e-01	err = 2.0776644147e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870319111716e-01	err = 1.1544245835e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870319107945e-01	err = 1.6366022073e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870319107340e-01	err = 4.3842632216e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.870319107244e-01	err = 9.2273610132e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870319107229e-01	err = 1.6440142298e-05	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870319107226e-01	err = 2.7580834334e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870319107226e-01	err = 4.5313147574e-07	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.870319107226e-01	err = 7.3039979726e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.870319107226e-01	err = 1.3531950605e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870319107226e-01	err = 3.9906609488e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.870319107226e-01	err = 1.5633451124e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870319107226e-01	err = 6.3301560351e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.870319107226e-01	err = 2.5470583934e-10	time = 0.01 sec
[ Info: CTMRG  16:	obj = +4.870319107226e-01	err = 1.0193232141e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.870319107226e-01	err = 4.0689835072e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.30e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.28e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 1.14e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 8.02e-08
└ * number of operations = 7
[ Info: LBFGS: iter   55, time  749.27 s: f = -0.662494796646, ‖∇f‖ = 3.0803e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.867774155272e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.867774505566e-01	err = 2.4129670839e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.867774604484e-01	err = 8.2999637010e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.867774624891e-01	err = 1.2052675009e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.867774628317e-01	err = 7.8406209004e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.867774628872e-01	err = 2.2233661270e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.867774628961e-01	err = 4.7774794494e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.867774628975e-01	err = 9.2935061620e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.867774628978e-01	err = 1.6530791481e-06	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.867774628978e-01	err = 2.9460621382e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.867774628978e-01	err = 9.1416010636e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.867774628978e-01	err = 3.4838553222e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.867774628978e-01	err = 1.3854341983e-08	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.867774628978e-01	err = 5.5451742816e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.867774628978e-01	err = 2.2193757470e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.867774628978e-01	err = 8.8782169049e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.867774628978e-01	err = 3.5499098017e-10	time = 0.01 sec
[ Info: CTMRG  17:	obj = +4.867774628978e-01	err = 1.4189740703e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.867774628978e-01	err = 5.6705989414e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.18e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.72e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 2.62e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 1.12e-07
└ * number of operations = 7
[ Info: LBFGS: iter   56, time  749.48 s: f = -0.662500144159, ‖∇f‖ = 3.3935e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.860002353379e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.860013431488e-01	err = 3.3115184730e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.860014757934e-01	err = 4.6320537559e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.860014988886e-01	err = 1.0460184063e-03	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.860015027168e-01	err = 1.8618600125e-03	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.860015033390e-01	err = 3.9521060768e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.860015034393e-01	err = 6.9796470271e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.860015034555e-01	err = 1.4529615635e-05	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.860015034581e-01	err = 2.6751335073e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.860015034585e-01	err = 6.7074458469e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.860015034586e-01	err = 2.6579258020e-07	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.860015034586e-01	err = 1.0816693846e-07	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.860015034586e-01	err = 4.3801274739e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.860015034586e-01	err = 1.7649287882e-08	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.860015034586e-01	err = 7.0924442397e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.860015034586e-01	err = 2.8461462794e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.860015034586e-01	err = 1.1412987761e-09	time = 0.01 sec
[ Info: CTMRG  17:	obj = +4.860015034586e-01	err = 4.5747320396e-10	time = 0.00 sec
[ Info: CTMRG  18:	obj = +4.860015034586e-01	err = 1.8333045843e-10	time = 0.00 sec
[ Info: CTMRG conv 19:	obj = +4.860015034586e-01	err = 7.3458567112e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.71e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.65e-05
[ Info: BiCGStab linsolve in iteration 2: normres = 8.38e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 6.50e-08
└ * number of operations = 7
[ Info: LBFGS: iter   57, time  749.69 s: f = -0.662501922158, ‖∇f‖ = 5.3636e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.863578737185e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.863579909469e-01	err = 9.5313618234e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.863580289693e-01	err = 1.3994846381e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.863580355929e-01	err = 4.1410928667e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.863580366873e-01	err = 7.3460619254e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.863580368647e-01	err = 1.7260360784e-04	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.863580368932e-01	err = 3.4080997535e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.863580368978e-01	err = 7.2299169005e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.863580368985e-01	err = 1.3644740944e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.863580368986e-01	err = 3.1155296829e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.863580368986e-01	err = 1.2542758283e-07	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.863580368986e-01	err = 5.1015432158e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.863580368986e-01	err = 2.0633187411e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.863580368986e-01	err = 8.3082140012e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.863580368986e-01	err = 3.3371382732e-09	time = 0.01 sec
[ Info: CTMRG  15:	obj = +4.863580368986e-01	err = 1.3385872957e-09	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.863580368986e-01	err = 5.3652763580e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.863580368986e-01	err = 2.1494620807e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.863580368986e-01	err = 8.6095563925e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.21e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.23e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 4.60e-07
└ * number of operations = 6
[ Info: LBFGS: iter   58, time  749.89 s: f = -0.662504711089, ‖∇f‖ = 2.3876e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.866212669635e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.866213613100e-01	err = 1.4759657582e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.866213747809e-01	err = 7.9289524373e-04	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.866213771274e-01	err = 5.1306401193e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.866213775143e-01	err = 4.2065930893e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.866213775769e-01	err = 1.0562284232e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.866213775869e-01	err = 2.4072274243e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.866213775886e-01	err = 5.1736460954e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.866213775888e-01	err = 9.8532084572e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.866213775889e-01	err = 1.9730125915e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.866213775889e-01	err = 7.1167411583e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.866213775889e-01	err = 2.8948314389e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.866213775889e-01	err = 1.1741049954e-08	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.866213775889e-01	err = 4.7348494686e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.866213775889e-01	err = 1.9029961332e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.866213775889e-01	err = 7.6342973253e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.866213775889e-01	err = 3.0595469313e-10	time = 0.01 sec
[ Info: CTMRG  17:	obj = +4.866213775889e-01	err = 1.2254344743e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.866213775889e-01	err = 4.9069377092e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.55e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 8.70e-06
[ Info: BiCGStab linsolve in iteration 2: normres = 1.81e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 8.12e-08
└ * number of operations = 7
[ Info: LBFGS: iter   59, time  750.10 s: f = -0.662506778078, ‖∇f‖ = 1.8200e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.867458482880e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.867459827181e-01	err = 1.9349503462e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.867459822846e-01	err = 5.3929265647e-04	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.867459822054e-01	err = 7.3569257296e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.867459821906e-01	err = 2.0179568136e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.867459821881e-01	err = 6.4892812865e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.867459821877e-01	err = 1.3236857494e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.867459821876e-01	err = 2.2794381970e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.867459821876e-01	err = 3.7412876605e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.867459821876e-01	err = 7.6207185477e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.867459821876e-01	err = 2.5410811418e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.867459821876e-01	err = 1.0282660307e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.867459821876e-01	err = 4.1984756534e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.867459821876e-01	err = 1.6973518368e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.867459821876e-01	err = 6.8235565697e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.867459821876e-01	err = 2.7355442340e-10	time = 0.01 sec
[ Info: CTMRG  16:	obj = +4.867459821876e-01	err = 1.0951984365e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.867459821876e-01	err = 4.3822209879e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.97e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 7.14e-06
[ Info: BiCGStab linsolve in iteration 2: normres = 1.54e-06
┌ Info: BiCGStab linsolve converged at iteration 2.5:
│ * norm of residual = 6.11e-08
└ * number of operations = 7
[ Info: LBFGS: iter   60, time  750.31 s: f = -0.662508721362, ‖∇f‖ = 1.9946e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.869287389745e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.869286597747e-01	err = 1.9940602121e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.869286601384e-01	err = 1.3841587518e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.869286603601e-01	err = 1.4840542536e-03	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.869286603954e-01	err = 3.1301968780e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.869286604010e-01	err = 1.2092804596e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.869286604018e-01	err = 2.8689931889e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.869286604020e-01	err = 5.3905012256e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.869286604020e-01	err = 9.2092243926e-07	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.869286604020e-01	err = 1.7574894868e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.869286604020e-01	err = 4.6388651259e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.869286604020e-01	err = 1.6794589504e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.869286604020e-01	err = 6.6199291963e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.869286604020e-01	err = 2.6456375835e-09	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.869286604020e-01	err = 1.0581734137e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.869286604020e-01	err = 4.2307623696e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.869286604020e-01	err = 1.6908030361e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.869286604020e-01	err = 6.7554981369e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.10e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 5.78e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 6.48e-07
└ * number of operations = 6
[ Info: LBFGS: iter   61, time  750.51 s: f = -0.662510450644, ‖∇f‖ = 3.2749e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.872775787444e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.872777894551e-01	err = 2.3434221429e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.872777841895e-01	err = 1.8813696812e-03	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.872777833132e-01	err = 6.1073374493e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.872777831708e-01	err = 6.8243763429e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.872777831478e-01	err = 1.5065789801e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.872777831441e-01	err = 2.6638126130e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.872777831435e-01	err = 4.2841700114e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.872777831434e-01	err = 7.6306045732e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.872777831434e-01	err = 1.8614599385e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.872777831434e-01	err = 6.3967311729e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.872777831434e-01	err = 2.4744609316e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.872777831434e-01	err = 9.8243799941e-09	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.872777831434e-01	err = 3.9171440203e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.872777831434e-01	err = 1.5629020137e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.872777831434e-01	err = 6.2344782858e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.872777831434e-01	err = 2.4863067289e-10	time = 0.01 sec
[ Info: CTMRG conv 17:	obj = +4.872777831434e-01	err = 9.9130046892e-11	time = 0.09 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.49e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.08e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.17e-05
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 8.75e-07
└ * number of operations = 6
[ Info: LBFGS: iter   62, time  750.71 s: f = -0.662510853517, ‖∇f‖ = 2.9884e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870813040506e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870813615032e-01	err = 1.2913179703e-03	time = 0.05 sec
[ Info: CTMRG   2:	obj = +4.870813597568e-01	err = 9.8483687344e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870813594482e-01	err = 3.1885953822e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870813593976e-01	err = 3.8106446287e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870813593894e-01	err = 8.5842328305e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.870813593881e-01	err = 1.4678094852e-05	time = 0.24 sec
[ Info: CTMRG   7:	obj = +4.870813593879e-01	err = 2.9454756324e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870813593879e-01	err = 5.6025443041e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870813593879e-01	err = 9.5459992712e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870813593879e-01	err = 1.6099638575e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.870813593879e-01	err = 5.7201685986e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870813593879e-01	err = 2.2454381617e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.870813593879e-01	err = 8.9491949405e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870813593879e-01	err = 3.5715708112e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.870813593879e-01	err = 1.4253943748e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.870813593879e-01	err = 5.6877686023e-11	time = 0.35 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 2.04e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 7.36e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 2.05e-07
└ * number of operations = 6
[ Info: LBFGS: iter   63, time  751.17 s: f = -0.662511685410, ‖∇f‖ = 8.0551e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870349859529e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870349859999e-01	err = 2.0991342629e-04	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.870349862326e-01	err = 1.7244951606e-04	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.870349862713e-01	err = 1.2109232875e-04	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.870349862776e-01	err = 9.9158329777e-05	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.870349862786e-01	err = 2.3228423778e-05	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.870349862788e-01	err = 4.6330789052e-06	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.870349862788e-01	err = 1.0140210755e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.870349862788e-01	err = 1.9433795295e-07	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.870349862788e-01	err = 3.5422610687e-08	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.870349862788e-01	err = 7.4279377322e-09	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.870349862788e-01	err = 2.4014400012e-09	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.870349862788e-01	err = 9.7471500618e-10	time = 0.02 sec
[ Info: CTMRG  13:	obj = +4.870349862788e-01	err = 3.9332033591e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870349862788e-01	err = 1.5796076576e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +4.870349862788e-01	err = 6.3284236934e-11	time = 0.12 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.70e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 6.21e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 2.15e-07
└ * number of operations = 6
[ Info: LBFGS: iter   64, time  751.39 s: f = -0.662511841022, ‖∇f‖ = 7.3596e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.869966911865e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.869966411373e-01	err = 5.2013445409e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.869966429809e-01	err = 2.4833601508e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.869966433178e-01	err = 3.2690554348e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.869966433733e-01	err = 1.3976489212e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.869966433823e-01	err = 3.4992166977e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.869966433837e-01	err = 6.3118717958e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.869966433839e-01	err = 1.4864271828e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.869966433840e-01	err = 2.9818848923e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.869966433840e-01	err = 5.9144279246e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.869966433840e-01	err = 1.4288396759e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.869966433840e-01	err = 5.5280856478e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.869966433840e-01	err = 2.2403524184e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.869966433840e-01	err = 9.0152133171e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.869966433840e-01	err = 3.6153983618e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.869966433840e-01	err = 1.4472500077e-10	time = 0.01 sec
[ Info: CTMRG conv 16:	obj = +4.869966433840e-01	err = 5.7875664824e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 7.03e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.11e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.59e-07
└ * number of operations = 6
[ Info: LBFGS: iter   65, time  751.57 s: f = -0.662512305957, ‖∇f‖ = 8.4091e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.869623538089e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.869623228070e-01	err = 6.3829661358e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.869623230353e-01	err = 3.8613461334e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.869623231053e-01	err = 3.5505201312e-04	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.869623231176e-01	err = 1.4057040963e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.869623231197e-01	err = 3.5843110600e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.869623231200e-01	err = 5.8504693981e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.869623231200e-01	err = 1.3106255496e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.869623231200e-01	err = 2.5814406020e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.869623231200e-01	err = 5.1426243502e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.869623231200e-01	err = 1.8065260945e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.869623231200e-01	err = 7.3136567715e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.869623231200e-01	err = 2.9477386272e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.869623231200e-01	err = 1.1825459408e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.869623231200e-01	err = 4.7310794397e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.869623231200e-01	err = 1.8903348735e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.869623231200e-01	err = 7.5478963908e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.49e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.42e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.54e-07
└ * number of operations = 6
[ Info: LBFGS: iter   66, time  751.76 s: f = -0.662512680347, ‖∇f‖ = 7.3318e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.873820030782e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.873820420917e-01	err = 2.2143221946e-03	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.873820372269e-01	err = 2.0481779639e-03	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.873820364679e-01	err = 5.3037962248e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.873820363473e-01	err = 7.5089565140e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.873820363281e-01	err = 1.7017852072e-04	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.873820363251e-01	err = 3.0094963158e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.873820363246e-01	err = 5.8035667300e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.873820363245e-01	err = 1.0447594979e-06	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.873820363245e-01	err = 1.7658119259e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.873820363245e-01	err = 3.0977924964e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.873820363245e-01	err = 7.3032903032e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.873820363245e-01	err = 2.5111531456e-09	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.873820363245e-01	err = 9.8851018512e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.873820363245e-01	err = 3.9522131739e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.873820363245e-01	err = 1.5797195478e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.873820363245e-01	err = 6.3028635303e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.22e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.08e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 3.14e-07
└ * number of operations = 6
[ Info: CTMRG init:	obj = +4.871502229196e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871502306089e-01	err = 9.8755224133e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871502296477e-01	err = 9.0529225624e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871502294976e-01	err = 2.3382784352e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871502294737e-01	err = 3.3204031357e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871502294699e-01	err = 7.5449854396e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871502294693e-01	err = 1.3361981270e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.871502294692e-01	err = 2.5528415707e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871502294692e-01	err = 4.5938787008e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871502294692e-01	err = 7.7466797946e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871502294692e-01	err = 1.3531039075e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871502294692e-01	err = 3.2009851233e-09	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.871502294692e-01	err = 1.1117563012e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871502294692e-01	err = 4.4013941366e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871502294692e-01	err = 1.7643050257e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +4.871502294692e-01	err = 7.0619373596e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 7.67e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.99e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.65e-07
└ * number of operations = 6
[ Info: LBFGS: iter   67, time  752.14 s: f = -0.662512901164, ‖∇f‖ = 1.2648e-03, α = 4.44e-01, m = 16, nfg = 2
[ Info: CTMRG init:	obj = +4.871582525206e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871582439217e-01	err = 7.0361757325e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871582415394e-01	err = 4.6307068763e-04	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.871582411462e-01	err = 1.9343006639e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871582410819e-01	err = 5.7876533775e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871582410716e-01	err = 1.2977387141e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871582410699e-01	err = 3.0450495296e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871582410697e-01	err = 6.7359692099e-07	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.871582410696e-01	err = 1.3578179313e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871582410696e-01	err = 5.2222628038e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871582410696e-01	err = 2.0864136204e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871582410696e-01	err = 8.3510027031e-09	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.871582410696e-01	err = 3.3335645189e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871582410696e-01	err = 1.3299483108e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871582410696e-01	err = 5.3024874607e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.871582410696e-01	err = 2.1135886950e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.871582410696e-01	err = 8.4230456350e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.10e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.20e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.86e-07
└ * number of operations = 6
[ Info: LBFGS: iter   68, time  752.34 s: f = -0.662513239918, ‖∇f‖ = 5.7061e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871483260462e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871483438638e-01	err = 3.2602681294e-04	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.871483432773e-01	err = 3.0035296932e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871483431681e-01	err = 1.6100407782e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871483431499e-01	err = 1.1686577884e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871483431469e-01	err = 3.1380876014e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871483431464e-01	err = 5.5772827525e-06	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.871483431463e-01	err = 9.5246453347e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871483431463e-01	err = 1.5007471777e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871483431463e-01	err = 5.2589587229e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871483431463e-01	err = 2.0682875473e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.871483431463e-01	err = 8.2471209582e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871483431463e-01	err = 3.2886125745e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871483431463e-01	err = 1.3117782091e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871483431463e-01	err = 5.2302126729e-10	time = 0.01 sec
[ Info: CTMRG  15:	obj = +4.871483431463e-01	err = 2.0848754569e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.871483431463e-01	err = 8.3092252273e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 1.02e-05
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.05e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.60e-07
└ * number of operations = 6
[ Info: LBFGS: iter   69, time  752.53 s: f = -0.662513444109, ‖∇f‖ = 6.0192e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871055125305e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871055599359e-01	err = 3.3634784872e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871055603125e-01	err = 2.5982992996e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871055603637e-01	err = 4.1946092031e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871055603714e-01	err = 2.8063187517e-04	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.871055603726e-01	err = 8.7874307642e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871055603728e-01	err = 1.7319042429e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871055603728e-01	err = 3.1412622469e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871055603728e-01	err = 5.0099116614e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871055603728e-01	err = 8.5149232397e-08	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.871055603728e-01	err = 3.3763644435e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871055603728e-01	err = 1.3518044681e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871055603728e-01	err = 5.4068605594e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871055603728e-01	err = 2.1615959132e-09	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.871055603728e-01	err = 8.6336725923e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.871055603728e-01	err = 3.4466105637e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.871055603728e-01	err = 1.3753924361e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.871055603728e-01	err = 5.4873812654e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 8.99e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.90e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.40e-07
└ * number of operations = 6
[ Info: LBFGS: iter   70, time  752.73 s: f = -0.662513692106, ‖∇f‖ = 7.2056e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871552878423e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871553915379e-01	err = 8.9202731825e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871553973321e-01	err = 7.3548423914e-04	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.871553983327e-01	err = 8.3595905074e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871553984966e-01	err = 1.6546816571e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871553985229e-01	err = 7.6151164930e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871553985271e-01	err = 1.7566067674e-05	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871553985278e-01	err = 3.5257855639e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.871553985279e-01	err = 6.8726358933e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871553985279e-01	err = 1.7765534459e-07	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871553985279e-01	err = 6.1463487748e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871553985279e-01	err = 2.3987614869e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871553985279e-01	err = 9.5606539508e-09	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.871553985280e-01	err = 3.8234189622e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871553985279e-01	err = 1.5284842653e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.871553985280e-01	err = 6.1071515684e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.871553985280e-01	err = 2.4389413250e-10	time = 0.01 sec
[ Info: CTMRG conv 17:	obj = +4.871553985279e-01	err = 9.7367767317e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.47e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 9.85e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 3.86e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 2.30e-07
└ * number of operations = 6
[ Info: LBFGS: iter   71, time  752.93 s: f = -0.662513794204, ‖∇f‖ = 1.0926e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870955159804e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870955242317e-01	err = 4.7869063845e-04	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.870955246517e-01	err = 2.6992177903e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870955247220e-01	err = 1.0961449254e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870955247336e-01	err = 1.4925804044e-04	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870955247355e-01	err = 4.8672011520e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.870955247358e-01	err = 1.0918883684e-05	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.870955247358e-01	err = 2.0619789672e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870955247358e-01	err = 3.6034236211e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870955247358e-01	err = 6.0273330273e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870955247358e-01	err = 1.3728687924e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.870955247358e-01	err = 4.9082233078e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870955247358e-01	err = 1.9226655236e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.870955247358e-01	err = 7.6680180638e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870955247358e-01	err = 3.0656189433e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.870955247358e-01	err = 1.2255553875e-10	time = 0.01 sec
[ Info: CTMRG conv 16:	obj = +4.870955247358e-01	err = 4.8977976333e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 7.75e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 2.85e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 2.06e-07
└ * number of operations = 6
[ Info: LBFGS: iter   72, time  753.13 s: f = -0.662513946704, ‖∇f‖ = 4.1373e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870752269852e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870752286197e-01	err = 3.9932652563e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.870752288596e-01	err = 2.6466484168e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870752289037e-01	err = 1.1433401528e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870752289111e-01	err = 6.2273963197e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870752289123e-01	err = 2.4705984548e-05	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.870752289125e-01	err = 5.7977075461e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870752289125e-01	err = 1.1276763650e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870752289125e-01	err = 2.0656333962e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870752289125e-01	err = 3.9040725698e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870752289125e-01	err = 1.1485978827e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.870752289125e-01	err = 4.6100960558e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870752289125e-01	err = 1.8568373167e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.870752289125e-01	err = 7.4561344742e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870752289125e-01	err = 2.9872031965e-10	time = 0.01 sec
[ Info: CTMRG  15:	obj = +4.870752289125e-01	err = 1.1950654362e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.870752289125e-01	err = 4.7763719917e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.47e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.71e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.69e-07
└ * number of operations = 6
[ Info: LBFGS: iter   73, time  753.36 s: f = -0.662514020708, ‖∇f‖ = 3.2774e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870813432406e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870813453148e-01	err = 1.8834962320e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.870813452681e-01	err = 1.2686730260e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870813452625e-01	err = 1.3145315247e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870813452617e-01	err = 3.1816344042e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870813452615e-01	err = 1.0342581760e-05	time = 0.24 sec
[ Info: CTMRG   6:	obj = +4.870813452615e-01	err = 2.0854868107e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870813452615e-01	err = 3.6519336561e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870813452615e-01	err = 6.4424421015e-08	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870813452615e-01	err = 1.3843181511e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870813452615e-01	err = 4.3185362476e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.870813452615e-01	err = 1.6324287363e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870813452615e-01	err = 6.4480857565e-10	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.870813452615e-01	err = 2.5663103239e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870813452615e-01	err = 1.0229134609e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +4.870813452615e-01	err = 4.0787669868e-11	time = 0.30 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.50e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.14e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.35e-07
└ * number of operations = 6
[ Info: LBFGS: iter   74, time  753.78 s: f = -0.662514074542, ‖∇f‖ = 4.1795e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870722344844e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870723291553e-01	err = 2.5941061633e-04	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.870723464056e-01	err = 1.1084784840e-04	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.870723494464e-01	err = 9.3758282099e-05	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.870723499475e-01	err = 9.4297775847e-05	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.870723500284e-01	err = 2.5561657323e-05	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.870723500413e-01	err = 4.6248610944e-06	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.870723500434e-01	err = 1.1436931887e-06	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.870723500437e-01	err = 4.3630391336e-07	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.870723500438e-01	err = 1.7801537182e-07	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.870723500438e-01	err = 7.2554332340e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.870723500438e-01	err = 2.9358076362e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.870723500438e-01	err = 1.1810976534e-08	time = 0.02 sec
[ Info: CTMRG  13:	obj = +4.870723500438e-01	err = 4.7359284405e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.870723500438e-01	err = 1.8951827955e-09	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.870723500438e-01	err = 7.5749821459e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.870723500438e-01	err = 3.0254736910e-10	time = 0.00 sec
[ Info: CTMRG  17:	obj = +4.870723500438e-01	err = 1.2078791119e-10	time = 0.00 sec
[ Info: CTMRG conv 18:	obj = +4.870723500438e-01	err = 4.8208653950e-11	time = 0.13 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.69e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.36e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.06e-07
└ * number of operations = 6
[ Info: LBFGS: iter   75, time  754.01 s: f = -0.662514105503, ‖∇f‖ = 9.0816e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871059996010e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871060260824e-01	err = 2.3212975558e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871060313226e-01	err = 6.0366707495e-05	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871060322339e-01	err = 4.2781800076e-05	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871060323836e-01	err = 9.7161530041e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871060324077e-01	err = 2.6294097185e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871060324116e-01	err = 5.1769823968e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871060324122e-01	err = 9.8695437433e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871060324123e-01	err = 2.4648367451e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871060324123e-01	err = 8.9984899112e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871060324123e-01	err = 3.6220571920e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871060324123e-01	err = 1.4683676590e-08	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.871060324123e-01	err = 5.9202167119e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871060324123e-01	err = 2.3770160384e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871060324123e-01	err = 9.5188830755e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.871060324123e-01	err = 3.8060341252e-10	time = 0.00 sec
[ Info: CTMRG  16:	obj = +4.871060324123e-01	err = 1.5203855885e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.871060324123e-01	err = 6.0699425286e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.28e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.24e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.03e-07
└ * number of operations = 6
[ Info: LBFGS: iter   76, time  754.21 s: f = -0.662514180025, ‖∇f‖ = 2.8220e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871258251721e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871258259952e-01	err = 1.7293852066e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871258260174e-01	err = 1.1346642231e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871258260216e-01	err = 1.0233945068e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871258260223e-01	err = 6.0570493275e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871258260224e-01	err = 2.0896520923e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871258260224e-01	err = 4.5655925705e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871258260224e-01	err = 8.5956434648e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871258260224e-01	err = 1.5327118351e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871258260224e-01	err = 2.8484127698e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871258260224e-01	err = 6.4590159028e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871258260224e-01	err = 2.5185838514e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871258260224e-01	err = 1.0135315637e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871258260224e-01	err = 4.0681619704e-10	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871258260224e-01	err = 1.6292460926e-10	time = 0.00 sec
[ Info: CTMRG conv 15:	obj = +4.871258260224e-01	err = 6.5156869516e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.86e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.03e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.03e-07
└ * number of operations = 6
[ Info: LBFGS: iter   77, time  754.39 s: f = -0.662514206339, ‖∇f‖ = 2.0931e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871297192460e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871297195782e-01	err = 2.1211911635e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871297194628e-01	err = 1.4320092176e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871297194429e-01	err = 1.9926352688e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871297194397e-01	err = 3.2429111048e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871297194392e-01	err = 1.4663659258e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871297194391e-01	err = 3.5857813250e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871297194391e-01	err = 7.0178191046e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871297194391e-01	err = 1.2462503873e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871297194391e-01	err = 2.1787859117e-08	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.871297194391e-01	err = 4.0553879959e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871297194391e-01	err = 1.1212126372e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871297194391e-01	err = 4.2962250755e-10	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871297194391e-01	err = 1.6958944823e-10	time = 0.00 sec
[ Info: CTMRG conv 14:	obj = +4.871297194391e-01	err = 6.7374698962e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 5.31e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.27e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.07e-07
└ * number of operations = 6
[ Info: LBFGS: iter   78, time  754.58 s: f = -0.662514231519, ‖∇f‖ = 2.6637e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871403384801e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871403574921e-01	err = 2.5568178996e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871403605819e-01	err = 2.0089994629e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871403611211e-01	err = 2.9754469798e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871403612097e-01	err = 5.9599373811e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871403612240e-01	err = 2.6363572735e-05	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.871403612263e-01	err = 5.9238042400e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871403612266e-01	err = 1.1086621617e-06	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871403612267e-01	err = 2.0262400771e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871403612267e-01	err = 7.5692988916e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871403612267e-01	err = 3.0759849754e-08	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.871403612267e-01	err = 1.2466815181e-08	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871403612267e-01	err = 5.0206957768e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871403612267e-01	err = 2.0135864110e-09	time = 0.00 sec
[ Info: CTMRG  14:	obj = +4.871403612267e-01	err = 8.0565433886e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.871403612267e-01	err = 3.2190725787e-10	time = 0.01 sec
[ Info: CTMRG  16:	obj = +4.871403612267e-01	err = 1.2851399791e-10	time = 0.00 sec
[ Info: CTMRG conv 17:	obj = +4.871403612267e-01	err = 5.1280083750e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.79e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.16e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.06e-07
└ * number of operations = 6
[ Info: CTMRG init:	obj = +4.871355908543e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871355966086e-01	err = 1.4046316207e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871355975438e-01	err = 1.1049674349e-04	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871355977071e-01	err = 1.6372858084e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871355977339e-01	err = 3.2766704582e-05	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.871355977382e-01	err = 1.4493026864e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871355977389e-01	err = 3.2535786842e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871355977390e-01	err = 6.0746278835e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871355977390e-01	err = 1.1048152143e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871355977390e-01	err = 4.1498305040e-08	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.871355977390e-01	err = 1.6854548372e-08	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871355977390e-01	err = 6.8302721326e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871355977390e-01	err = 2.7507577208e-09	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871355977390e-01	err = 1.1032683620e-09	time = 0.01 sec
[ Info: CTMRG  14:	obj = +4.871355977390e-01	err = 4.4145680170e-10	time = 0.00 sec
[ Info: CTMRG  15:	obj = +4.871355977390e-01	err = 1.7639427491e-10	time = 0.00 sec
[ Info: CTMRG conv 16:	obj = +4.871355977390e-01	err = 7.0428609682e-11	time = 0.08 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.84e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.12e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.03e-07
└ * number of operations = 6
[ Info: LBFGS: iter   79, time  754.98 s: f = -0.662514243124, ‖∇f‖ = 2.4156e-04, α = 5.50e-01, m = 16, nfg = 2
[ Info: CTMRG init:	obj = +4.871179793379e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871179801697e-01	err = 1.5631360249e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871179801833e-01	err = 5.5606384689e-05	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.871179801852e-01	err = 1.4161299902e-04	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.871179801854e-01	err = 6.4294663733e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871179801855e-01	err = 1.9586641733e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871179801855e-01	err = 4.0133955465e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871179801855e-01	err = 7.2141125870e-07	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.871179801855e-01	err = 1.2115678362e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.871179801855e-01	err = 2.0142931321e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871179801855e-01	err = 3.6807478447e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871179801855e-01	err = 9.3804583294e-10	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871179801855e-01	err = 3.2939576562e-10	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.871179801855e-01	err = 1.2840573337e-10	time = 0.00 sec
[ Info: CTMRG conv 14:	obj = +4.871179801855e-01	err = 5.1027997092e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.83e-06
[ Info: BiCGStab linsolve in iteration 1.5: normres = 1.17e-06
┌ Info: BiCGStab linsolve converged at iteration 2:
│ * norm of residual = 1.06e-07
└ * number of operations = 6
[ Info: LBFGS: iter   80, time  755.17 s: f = -0.662514255964, ‖∇f‖ = 1.4589e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.871073990611e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.871073992573e-01	err = 1.2520311614e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.871073992474e-01	err = 3.9887951457e-05	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.871073992458e-01	err = 7.5472399281e-05	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.871073992455e-01	err = 4.0831399145e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.871073992454e-01	err = 1.1983311291e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.871073992454e-01	err = 2.4288228106e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.871073992454e-01	err = 4.3266287791e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.871073992454e-01	err = 7.3700330899e-08	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.871073992454e-01	err = 1.3367391679e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.871073992454e-01	err = 3.2342099574e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.871073992454e-01	err = 1.1050350908e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.871073992454e-01	err = 4.2985840603e-10	time = 0.00 sec
[ Info: CTMRG  13:	obj = +4.871073992454e-01	err = 1.7132274257e-10	time = 0.01 sec
[ Info: CTMRG conv 14:	obj = +4.871073992454e-01	err = 6.8479019651e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 4.30e-06
┌ Info: BiCGStab linsolve converged at iteration 1.5:
│ * norm of residual = 9.20e-07
└ * number of operations = 5
[ Info: LBFGS: iter   81, time  755.35 s: f = -0.662514264597, ‖∇f‖ = 1.6299e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870802265100e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870802270952e-01	err = 1.5637536758e-04	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.870802270994e-01	err = 6.5932274507e-05	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.870802270998e-01	err = 8.4743946162e-05	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870802270999e-01	err = 6.3767011841e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870802270999e-01	err = 1.8007628475e-05	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.870802270999e-01	err = 3.6762667212e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870802270999e-01	err = 6.8319004154e-07	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.870802270999e-01	err = 1.2815259166e-07	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870802270999e-01	err = 2.2222081395e-08	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870802270999e-01	err = 3.8771168412e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.870802270999e-01	err = 1.1224318440e-09	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870802270999e-01	err = 4.1649202324e-10	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.870802270999e-01	err = 1.6462481867e-10	time = 0.00 sec
[ Info: CTMRG conv 14:	obj = +4.870802270999e-01	err = 6.5733103902e-11	time = 0.07 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.75e-06
┌ Info: BiCGStab linsolve converged at iteration 1.5:
│ * norm of residual = 6.29e-07
└ * number of operations = 5
[ Info: LBFGS: iter   82, time  755.53 s: f = -0.662514270687, ‖∇f‖ = 2.0842e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870861801606e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870861801546e-01	err = 3.0026862223e-05	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.870861801521e-01	err = 1.0052445643e-05	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870861801518e-01	err = 3.5172205110e-05	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870861801518e-01	err = 1.7511339538e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870861801518e-01	err = 5.2661006646e-06	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.870861801518e-01	err = 1.0545807128e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870861801518e-01	err = 1.8889459938e-07	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870861801518e-01	err = 3.1188385446e-08	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870861801518e-01	err = 4.9674504469e-09	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870861801518e-01	err = 7.6165287749e-10	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.870861801518e-01	err = 1.2997626631e-10	time = 0.00 sec
[ Info: CTMRG conv 12:	obj = +4.870861801518e-01	err = 3.5947094377e-11	time = 0.06 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.77e-06
┌ Info: BiCGStab linsolve converged at iteration 1.5:
│ * norm of residual = 6.21e-07
└ * number of operations = 5
[ Info: LBFGS: iter   83, time  755.70 s: f = -0.662514276005, ‖∇f‖ = 1.1227e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870924286027e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870924285770e-01	err = 5.1111427041e-05	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.870924285712e-01	err = 1.6995613204e-05	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.870924285707e-01	err = 4.5583186554e-05	time = 0.00 sec
[ Info: CTMRG   4:	obj = +4.870924285706e-01	err = 2.2187857241e-05	time = 0.00 sec
[ Info: CTMRG   5:	obj = +4.870924285706e-01	err = 6.7063783159e-06	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.870924285706e-01	err = 1.3203376049e-06	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870924285706e-01	err = 2.3417684274e-07	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.870924285706e-01	err = 3.8634842497e-08	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870924285706e-01	err = 6.2564255420e-09	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870924285706e-01	err = 1.0591197139e-09	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.870924285706e-01	err = 2.4363386038e-10	time = 0.00 sec
[ Info: CTMRG conv 12:	obj = +4.870924285706e-01	err = 8.3649103606e-11	time = 0.06 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.92e-06
┌ Info: BiCGStab linsolve converged at iteration 1.5:
│ * norm of residual = 7.05e-07
└ * number of operations = 5
[ Info: LBFGS: iter   84, time  755.91 s: f = -0.662514281099, ‖∇f‖ = 1.0747e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870928381111e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870928379488e-01	err = 2.8222048730e-05	time = 0.00 sec
[ Info: CTMRG   2:	obj = +4.870928379327e-01	err = 1.9291761999e-05	time = 0.00 sec
[ Info: CTMRG   3:	obj = +4.870928379304e-01	err = 1.9206061441e-05	time = 0.24 sec
[ Info: CTMRG   4:	obj = +4.870928379300e-01	err = 4.1153580589e-06	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.870928379299e-01	err = 1.5444182106e-06	time = 0.00 sec
[ Info: CTMRG   6:	obj = +4.870928379299e-01	err = 3.2729482757e-07	time = 0.00 sec
[ Info: CTMRG   7:	obj = +4.870928379299e-01	err = 6.1772540606e-08	time = 0.00 sec
[ Info: CTMRG   8:	obj = +4.870928379299e-01	err = 1.2017085572e-08	time = 0.00 sec
[ Info: CTMRG   9:	obj = +4.870928379299e-01	err = 2.8873364166e-09	time = 0.00 sec
[ Info: CTMRG  10:	obj = +4.870928379299e-01	err = 9.2398739389e-10	time = 0.00 sec
[ Info: CTMRG  11:	obj = +4.870928379299e-01	err = 3.4237484988e-10	time = 0.00 sec
[ Info: CTMRG  12:	obj = +4.870928379299e-01	err = 1.3303805908e-10	time = 0.00 sec
[ Info: CTMRG conv 13:	obj = +4.870928379299e-01	err = 5.2490247254e-11	time = 0.30 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.98e-06
┌ Info: BiCGStab linsolve converged at iteration 1.5:
│ * norm of residual = 7.06e-07
└ * number of operations = 5
[ Info: LBFGS: iter   85, time  756.33 s: f = -0.662514284644, ‖∇f‖ = 1.0173e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +4.870856616218e-01	err = 1.0000e+00
[ Info: CTMRG   1:	obj = +4.870856604044e-01	err = 6.5977582010e-05	time = 0.01 sec
[ Info: CTMRG   2:	obj = +4.870856602664e-01	err = 4.1250268645e-05	time = 0.01 sec
[ Info: CTMRG   3:	obj = +4.870856602447e-01	err = 5.3496895553e-05	time = 0.01 sec
[ Info: CTMRG   4:	obj = +4.870856602412e-01	err = 2.8889294878e-05	time = 0.01 sec
[ Info: CTMRG   5:	obj = +4.870856602406e-01	err = 8.5942434633e-06	time = 0.01 sec
[ Info: CTMRG   6:	obj = +4.870856602405e-01	err = 1.7822285215e-06	time = 0.01 sec
[ Info: CTMRG   7:	obj = +4.870856602405e-01	err = 3.0751246919e-07	time = 0.01 sec
[ Info: CTMRG   8:	obj = +4.870856602405e-01	err = 5.1833408219e-08	time = 0.01 sec
[ Info: CTMRG   9:	obj = +4.870856602405e-01	err = 9.2340459614e-09	time = 0.01 sec
[ Info: CTMRG  10:	obj = +4.870856602405e-01	err = 2.6588387991e-09	time = 0.01 sec
[ Info: CTMRG  11:	obj = +4.870856602405e-01	err = 1.0354992143e-09	time = 0.01 sec
[ Info: CTMRG  12:	obj = +4.870856602405e-01	err = 4.0936881781e-10	time = 0.01 sec
[ Info: CTMRG  13:	obj = +4.870856602405e-01	err = 1.6260405180e-10	time = 0.01 sec
[ Info: CTMRG conv 14:	obj = +4.870856602405e-01	err = 6.4747169783e-11	time = 0.11 sec
[ Info: BiCGStab linsolve starts with norm of residual = 2.48e-02
[ Info: BiCGStab linsolve in iteration 1: normres = 3.90e-06
┌ Info: BiCGStab linsolve converged at iteration 1.5:
│ * norm of residual = 5.43e-07
└ * number of operations = 5
[ Info: LBFGS: converged after 86 iterations and time 756.56 s: f = -0.662514291826, ‖∇f‖ = 8.8301e-05

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

