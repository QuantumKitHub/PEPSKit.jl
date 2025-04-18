```@meta
EditURL = "../../../../examples/2.boundary_mps/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//2.boundary_mps/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//2.boundary_mps/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//2.boundary_mps)


# Boundary MPS contractions using VUMPS and PEPOs

Instead of using CTMRG to contract an infinite PEPS, one can also use an boundary MPSs
ansatz to contract the infinite network. In particular, we will here use VUMPS to do so.

Before we start, we'll fix the random seed for reproducability:

````julia
using Random
Random.seed!(29384293742893);
````

Besides `TensorKit` and `PEPSKit`, we here also need [`MPSKit`](https://quantumkithub.github.io/MPSKit.jl/stable/)
which implements the VUMPS algorithm as well as the required MPS operations:

````julia
using TensorKit, PEPSKit, MPSKit
````

## Computing a PEPS norm

We start by initializing a random initial infinite PEPS:

````julia
peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
````

````
InfinitePEPS{TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}}(TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}[TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')):
[:, :, 1, 1, 1] =
 -0.5524390176345264 - 0.07357188568178248im  0.34014501646081047 - 0.7552574870030472im
 -0.5455245317233405 + 0.8946618856309984im     1.249282911658007 + 0.45352274131986825im

[:, :, 2, 1, 1] =
    0.33621043661988675 + 0.4400876608299719im   -0.9866664087107284 - 0.28688827761325675im
 -0.0077250067072679235 + 1.7380910495900947im  -0.19071062901939098 - 1.1367500834118434im

[:, :, 1, 2, 1] =
 -0.09149850722392933 + 0.3560942836258964im      1.6255618447281441 - 0.5689426732891244im
 -0.19309251474097275 - 0.32363899914302613im  -0.025356816648697236 + 0.5632279168368712im

[:, :, 2, 2, 1] =
 0.07675114584269166 - 0.011479824536308164im  -0.17779977372973318 + 1.1379201927122535im
 -1.0116302866282385 - 0.9253070687198848im      1.1649047337212566 + 0.9936369101208083im

[:, :, 1, 1, 2] =
  0.2510676919806213 - 0.182052326055189im   -0.5792402993550532 - 0.4309109406268341im
 0.04501645227038913 - 0.8140971172854408im  -0.5608346802110794 + 0.21262550530307248im

[:, :, 2, 1, 2] =
  1.5061767210554262 + 0.17190948125245623im  -0.8001234458239143 + 0.6764943808639017im
 -0.8176938467062373 - 0.40919675695722396im  -0.6692181340575689 + 0.6923370271564298im

[:, :, 1, 2, 2] =
 -0.16556382071485704 + 0.2540132491548349im   0.05546115732751907 + 0.3723175507964387im
 -0.29883021417599165 - 0.07229462525164528im   -1.200173153698329 - 0.45509299328832953im

[:, :, 2, 2, 2] =
  0.289873563752043 + 0.44718981087960125im  0.018357838612906643 + 0.9634127683557584im
 0.5128282969211142 - 0.2865462937979091im   -0.44278618042821827 + 0.2612084385439659im
;;])
````

To compute its norm, we need to construct the transfer operator corresponding to
the partition function representing the overlap $\langle \psi_\text{PEPS} | \psi_\text{PEPS} \rangle$:

````julia
transfer = InfiniteTransferPEPS(peps₀, 1, 1)
````

````
single site InfiniteMPO{Tuple{TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}, TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}}}:
╷  ⋮
┼ O[1]: (TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')), TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')))
╵  ⋮

````

We then find its leading boundary MPS fixed point, where the corresponding eigenvalue
encodes the norm of the state. To that end, let us first we build an initial guess for the
boundary MPS, choosing a bond dimension of 20:

````julia
mps₀ = initializeMPS(transfer, [ComplexSpace(20)])
````

````
single site InfiniteMPS:
│   ⋮
│ C[1]: TensorMap(ℂ^20 ← ℂ^20)
├── AL[1]: TensorMap((ℂ^20 ⊗ ℂ^2 ⊗ (ℂ^2)') ← ℂ^20)
│   ⋮

````

Note that this will just construct a MPS with random Gaussian entries based on the virtual
spaces of the supplied transfer operator. Of course, one might come up with a better initial
guess (leading to better convergence) depending on the application. To find the leading
boundary MPS fixed point, we call [`leading_boundary`](@ref) using the [VUMPS](@extref)
algorithm from MPSKit. Note that, by default, `leading_boundary` uses CTMRG where the
settings are supplied as keyword arguments, so in the present case we need to supply the
VUMPS algorithm struct explicitly:

````julia
mps, env, ϵ = leading_boundary(mps₀, transfer, VUMPS(; tol=1e-6, verbosity=2));
````

````
[ Info: VUMPS init:	obj = +1.674563752306e+00 +3.035692829590e+00im	err = 7.5576e-01
[ Info: VUMPS conv 120:	obj = +6.831610878163e+00 -1.001928853191e-08im	err = 9.5332406967e-07	time = 1.80 sec

````

The norm of the state per unit cell is then given by the expectation value
$\langle \psi_\text{MPS} | \mathbb{T} | \psi_\text{MPS} \rangle$:

````julia
norm_vumps = abs(prod(expectation_value(mps, transfer)))
````

````
6.831610878163006
````

This can be compared to the result obtained using CTMRG, where we see that the results match:

````julia
env_ctmrg, = leading_boundary(
    CTMRGEnv(peps₀, ComplexSpace(20)), peps₀; tol=1e-6, verbosity=2
)
norm_ctmrg = abs(norm(peps₀, env_ctmrg))
@show abs(norm_vumps - norm_ctmrg) / norm_vumps;
````

````
[ Info: CTMRG init:	obj = -1.495741317009e+01 +3.091851579630e-01im	err = 1.0000e+00
[ Info: CTMRG conv 30:	obj = +6.831603585666e+00	err = 6.2262595140e-07	time = 0.28 sec
abs(norm_vumps - norm_ctmrg) / norm_vumps = 1.0674637855860049e-6

````

## Working with unit cells

For PEPS with non-trivial unit cells, the principle is exactly the same. The only difference
is that now the transfer operator of the PEPS norm partition function has multiple lines,
each of which can be represented by an [`InfiniteTransferPEPS`](@ref) object. Such a
multi-line transfer operator is represented by a `MultilineTransferPEPS` object. In this
case, the boundary MPS is an [`MultilineMPS`](@extref) object, which should be initialized
by specifying a virtual space for each site in the partition function unit cell.

First, we construct a PEPS with a $2 \times 2$ unit cell using the `unitcell` keyword
argument and then define the corresponding transfer PEPS:

````julia
peps₀_2x2 = InfinitePEPS(rand, ComplexF64, ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
transfer_2x2 = PEPSKit.MultilineTransferPEPS(peps₀_2x2, 1);
````

Now, the procedure is the same as before: We compute the norm once using VUMPS, once using CTMRG and then compare.

````julia
mps₀_2x2 = initializeMPS(transfer_2x2, fill(ComplexSpace(20), 2, 2))
mps_2x2, = leading_boundary(mps₀_2x2, transfer_2x2, VUMPS(; tol=1e-6, verbosity=2))
norm_2x2_vumps = abs(prod(expectation_value(mps_2x2, transfer_2x2)))

env_ctmrg_2x2, = leading_boundary(
    CTMRGEnv(peps₀_2x2, ComplexSpace(20)), peps₀_2x2; tol=1e-6, verbosity=2
)
norm_2x2_ctmrg = abs(norm(peps₀_2x2, env_ctmrg_2x2))

@show abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps;
````

````
[ Info: VUMPS init:	obj = +8.149302834396e+02 -8.860408249120e+01im	err = 8.6172e-01
[ Info: VUMPS conv 37:	obj = +1.046633709901e+05 -1.858418959285e-05im	err = 4.5282584466e-07	time = 2.10 sec
[ Info: CTMRG init:	obj = -1.240261729401e+02 -1.672150510263e+01im	err = 1.0000e+00
[ Info: CTMRG conv 47:	obj = +1.046633714846e+05	err = 1.6993045675e-07	time = 1.71 sec
abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps = 4.725134987376298e-9

````

Again, the results are compatible. Note that for larger unit cells and non-Hermitian PEPS
the VUMPS algorithm may become unstable, in which case the CTMRG algorithm is recommended.

## Contracting PEPO overlaps

Using exactly the same machinery, we can contract partition functions which encode the
expectation value of a PEPO for a given PEPS state. As an example, we can consider the
overlap of the PEPO correponding to the partition function of 3D classical Ising model with
our random PEPS from before and evaluate the overlap
$\langle \psi_\text{PEPS} | O_\text{PEPO} | \psi_\text{PEPS} \rangle$.

The classical Ising PEPO is defined as follows:

````julia
function ising_pepo(β; unitcell=(1, 1, 1))
    t = ComplexF64[exp(β) exp(-β); exp(-β) exp(β)]
    q = sqrt(t)

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    return InfinitePEPO(O; unitcell)
end;
````

To evaluate the overlap, we instantiate the PEPO and the corresponding [`InfiniteTransferPEPO`](@ref)
in the right direction, on the right row of the partition function (trivial here):

````julia
pepo = ising_pepo(1)
transfer_pepo = InfiniteTransferPEPO(peps₀, pepo, 1, 1)
````

````
single site InfiniteMPO{Tuple{TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}, TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}, TensorMap{ComplexF64, ComplexSpace, 2, 4, Vector{ComplexF64}}}}:
╷  ⋮
┼ O[1]: (TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')), TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')), TensorMap((ℂ^2 ⊗ (ℂ^2)') ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')))
╵  ⋮

````

As before, we converge the boundary MPS using VUMPS and then compute the expectation value:

````julia
mps₀_pepo = initializeMPS(transfer_pepo, [ComplexSpace(20)])
mps_pepo, = leading_boundary(mps₀_pepo, transfer_pepo, VUMPS(; tol=1e-6, verbosity=2))
norm_pepo = abs(prod(expectation_value(mps_pepo, transfer_pepo)));
@show norm_pepo;
````

````
[ Info: VUMPS init:	obj = +2.655321432467e+01 +3.760603778362e-01im	err = 8.9759e-01
┌ Warning: VUMPS cancel 200:	obj = -2.194912861838e+01 -6.105468516794e+01im	err = 5.7061338213e-01	time = 32.78 sec
└ @ MPSKit ~/.julia/packages/MPSKit/EfZBD/src/algorithms/statmech/vumps.jl:51
norm_pepo = 64.88018825545267

````

These objects and routines can be used to optimize PEPS fixed points of 3D partition
functions, see for example [Vanderstraeten et al.](@cite vanderstraeten_residual_2018)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

