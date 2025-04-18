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

We start by initializing a random initial infinite PEPS. Let us use uniformly distributed
complex entries using `rand` (which sometimes lead to better convergence than Gaussian
distributed `randn` elements):

````julia
peps₀ = InfinitePEPS(rand, ComplexF64, ComplexSpace(2), ComplexSpace(2))
````

````
InfinitePEPS{TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}}(TensorMap{ComplexF64, ComplexSpace, 1, 4, Vector{ComplexF64}}[TensorMap(ℂ^2 ← (ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')):
[:, :, 1, 1, 1] =
 0.8343040072662887 + 0.15425705836788395im  0.4612746978522435 + 0.7411151918989216im
 0.6640771294125087 + 0.4428356798799721im   0.9163597170532635 + 0.24145695415210522im

[:, :, 2, 1, 1] =
 0.44289651954161835 + 0.5968081052313008im   0.5473659268881094 + 0.37528062658773764im
 0.00644367423621961 + 0.9414462569909486im  0.36006028879229457 + 0.6157267258321241im

[:, :, 1, 2, 1] =
 0.04956065285909117 + 0.26119820734171617im    0.9153298540884296 + 0.3990244910357601im
 0.17944112964295234 + 0.4233545106724528im   0.020358359069476473 + 0.6501897922267199im

[:, :, 2, 2, 1] =
 0.040493161136161526 + 0.03501665486055905im  0.2591040734810338 + 0.8830094105726012im
    0.781658280511654 + 0.9662812119384394im   0.8169988652653896 + 0.674481616952991im

[:, :, 1, 1, 2] =
   0.2242833355717867 + 0.14929928451790686im  0.6883051212688887 + 0.588769359105893im
 0.046322385671192734 + 0.8543796191082029im   0.6437874016748227 + 0.257253015722232im

[:, :, 2, 1, 2] =
 0.8719996187768273 + 0.25052026742300637im  0.5714417314833022 + 0.9944321644519715im
 0.4273547968422168 + 0.6068478826937488im   0.4946426302106661 + 0.8353867377249198im

[:, :, 1, 2, 2] =
 0.6857354516279699 + 0.0952105576480895im  0.14591923452838773 + 0.0853564870114002im
 0.6779060054394721 + 0.4947495895268207im   0.9280821860668365 + 0.931316796924268im

[:, :, 2, 2, 2] =
 0.3716373366637086 + 0.2556099109043021im  0.7954831107819061 + 0.016909518973250215im
 0.9376161032047406 + 0.6320864548041844im  0.7900851372111909 + 0.5457560526661245im
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
[ Info: VUMPS init:	obj = +5.052950412844e+00 +1.493192627823e-02im	err = 8.4684e-01
[ Info: VUMPS conv 4:	obj = +1.744071150138e+01 +2.417441557995e-08im	err = 1.9047772246e-07	time = 0.10 sec

````

The norm of the state per unit cell is then given by the expectation value
$\langle \psi_\text{MPS} | \mathbb{T} | \psi_\text{MPS} \rangle$:

````julia
norm_vumps = abs(prod(expectation_value(mps, transfer)))
````

````
17.440711501378814
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
[ Info: CTMRG init:	obj = -5.556349490423e-01 +1.605938670370e+00im	err = 1.0000e+00
[ Info: CTMRG conv 37:	obj = +1.744071151099e+01	err = 3.2056303631e-07	time = 0.33 sec
abs(norm_vumps - norm_ctmrg) / norm_vumps = 5.510360046158767e-10

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
[ Info: VUMPS init:	obj = +6.668046237341e+02 -1.267878277078e+01im	err = 8.7901e-01
[ Info: VUMPS conv 69:	obj = +9.723958968917e+04 -3.481605377714e-03im	err = 6.3841720875e-07	time = 3.90 sec
[ Info: CTMRG init:	obj = +1.074898090007e+03 -2.096255594496e+02im	err = 1.0000e+00
[ Info: CTMRG conv 41:	obj = +9.723959008610e+04	err = 6.0518230963e-07	time = 1.50 sec
abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps = 4.082014263000404e-9

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
[ Info: VUMPS init:	obj = +3.309203535702e+01 -4.227375981212e-01im	err = 9.3280e-01
[ Info: VUMPS conv 5:	obj = +2.483696258643e+02 +2.387851822319e-07im	err = 5.0174146749e-08	time = 0.47 sec
norm_pepo = 248.36962586428106

````

These objects and routines can be used to optimize PEPS fixed points of 3D partition
functions, see for example [Vanderstraeten et al.](@cite vanderstraeten_residual_2018)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

