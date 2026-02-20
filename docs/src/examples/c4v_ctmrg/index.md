```@meta
EditURL = "../../../../examples/c4v_ctmrg/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/c4v_ctmrg/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/c4v_ctmrg/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/c4v_ctmrg)


# C₄ᵥ CTMRG and QR-CTMRG

In this example we demonstrate specialized CTMRG variants that exploit the $C_{4v}$ point group
symmetry of the physical model and PEPS at hand. This allows us to significantly reduce the
computational cost of contraction and optimization. To that end, we will consider the Heisenberg
Hamiltonian on the square lattice

```math
H = \sum_{\langle i,j \rangle} \left ( J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j \right )
```

We want to treat the model in the antiferromagnetic regime where the ground state exhibits
bipartite sublattice structure. To be able to represent this ground state in a $C_{4v}$-invariant
manner on a single-site unit cell, we perform a unitary sublattice rotation by setting
$(J_x, J_y, J_z)=(-1, 1, -1)$.

Let's get started by seeding the RNG and doing the imports:

````julia
using Random
using TensorKit, PEPSKit
Random.seed!(123456789);
````

## Defining a specialized Hamiltonian for C₄ᵥ-symmetric PEPS

Since the model under consideration and its ground state are invariant under 90° rotation and
Hermitian reflection, evaluating the expectation values of the horizontal and vertical energy
contributions are exactly equivalent. This allows us to effectively halve the computational cost
by evaluating only half of the terms and multiplying by 2. In practice, we implement this
using a specialized [`LocalOperator`](@ref) that contains only the relevant terms:

````julia
using MPSKitModels: S_xx, S_yy, S_zz

# Heisenberg model assuming C4v symmetric PEPS and environment, which only evaluates necessary term
function heisenberg_XYZ_c4v(lattice::InfiniteSquare; kwargs...)
    return heisenberg_XYZ_c4v(ComplexF64, Trivial, lattice; kwargs...)
end
function heisenberg_XYZ_c4v(
        T::Type{<:Number}, S::Type{<:Sector}, lattice::InfiniteSquare;
        Jx = -1.0, Jy = 1.0, Jz = -1.0, spin = 1 // 2,
    )
    @assert size(lattice) == (1, 1) "only trivial unit cells supported by C4v-symmetric Hamiltonians"
    term =
        rmul!(S_xx(T, S; spin = spin), Jx) +
        rmul!(S_yy(T, S; spin = spin), Jy) +
        rmul!(S_zz(T, S; spin = spin), Jz)
    spaces = fill(domain(term)[1], (1, 1))
    return LocalOperator( # horizontal and vertical contributions are identical
        spaces, (CartesianIndex(1, 1), CartesianIndex(1, 2)) => 2 * term
    )
end;
````

## Initializing C₄ᵥ-invariant PEPSs and environments

In order to use $C_{4v}$-symmetric algorithms, it is of course crucial to use initial guesses
that are exhibit $C_{4v}$ symmetry. First, we create a real-valued random PEPS that we explicitly
symmetrize using [`symmetrize!`](@ref) and the $C_{4v}$ symmetry [`RotateReflect`](@ref):

````julia
symm = RotateReflect()
D = 2
T = Float64
peps_random = InfinitePEPS(randn, T, ComplexSpace(2), ComplexSpace(D))
peps₀ = symmetrize!(peps_random, symm);
````

Initializing an $C_{4v}$-invariant environment is a bit more subtle and there is no one-size-fits-all
solution. As a good starting point one can use the initialization function [`initialize_random_c4v_env`](@ref)
(or also [`initialize_singlet_c4v_env`](@ref)) where we construct a diagonal corner with random
real entries and a random Hermitian edge tensor.

````julia
χ = 16
env_random_c4v = initialize_random_c4v_env(peps₀, ComplexSpace(χ));
````

Then contracting the PEPS using $C_{4v}$ CTMRG is as easy as just calling [`leading_boundary`](@ref)
but passing the invariant initial PEPS and environment as well as the `alg = :c4v` keyword argument:

````julia
env₀, = leading_boundary(env_random_c4v, peps₀; alg = :c4v, tol = 1.0e-10);
````

````
[ Info: CTMRG init:	obj = -1.430301957018e-02	err = 1.0000e+00
[ Info: CTMRG conv 36:	obj = +8.685181513863e+00	err = 6.8827026254e-11	time = 0.13 sec

````

## C₄ᵥ-symmetric optimization

We now take the invariant `peps₀` and `env₀` as a starting point for a gradient-based energy
minimization where we contract using $C_{4v}$ CTMRG such that the energy gradient will also
exhibit rotation and reflection symmetry. For that, we call `fixedpoint` and specify `alg = :c4v`
as the boundary contraction algorithm:

````julia
H = real(heisenberg_XYZ_c4v(InfiniteSquare())) # make Hamiltonian real-valued
peps, env, E, = fixedpoint(
    H, peps₀, env₀; optimizer_alg = (; tol = 1.0e-4), boundary_alg = (; alg = :c4v),
);
````

````
[ Info: LBFGS: initializing with f = -5.047653728981e-01, ‖∇f‖ = 1.9060e-01
[ Info: LBFGS: iter    1, Δt  52.8 ms: f = -5.056459154685e-01, ‖∇f‖ = 1.3798e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt 163.1 ms: f = -6.375540411515e-01, ‖∇f‖ = 1.7202e-01, α = 2.79e+01, m = 1, nfg = 5
[ Info: LBFGS: iter    3, Δt  23.0 ms: f = -6.486432921799e-01, ‖∇f‖ = 1.3180e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt  32.0 ms: f = -6.520905366511e-01, ‖∇f‖ = 1.2693e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt  30.8 ms: f = -6.543779478465e-01, ‖∇f‖ = 8.4374e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt  24.5 ms: f = -6.574474243297e-01, ‖∇f‖ = 9.2229e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt  32.3 ms: f = -6.589601436763e-01, ‖∇f‖ = 4.1340e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  29.7 ms: f = -6.593161746273e-01, ‖∇f‖ = 1.6522e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt  22.7 ms: f = -6.594944356002e-01, ‖∇f‖ = 1.3207e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt  28.5 ms: f = -6.598273620822e-01, ‖∇f‖ = 1.2344e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  28.3 ms: f = -6.600090370393e-01, ‖∇f‖ = 8.5852e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  19.9 ms: f = -6.601648157099e-01, ‖∇f‖ = 3.1453e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  25.9 ms: f = -6.601883494925e-01, ‖∇f‖ = 2.2795e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt  20.3 ms: f = -6.602037369191e-01, ‖∇f‖ = 2.8426e-03, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  27.1 ms: f = -6.602113170029e-01, ‖∇f‖ = 2.0017e-03, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  18.5 ms: f = -6.602199370383e-01, ‖∇f‖ = 1.2403e-03, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt  26.5 ms: f = -6.602252410543e-01, ‖∇f‖ = 7.3832e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt  26.1 ms: f = -6.602292169497e-01, ‖∇f‖ = 6.4978e-04, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt  18.4 ms: f = -6.602308383659e-01, ‖∇f‖ = 3.7433e-04, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt  25.4 ms: f = -6.602310776646e-01, ‖∇f‖ = 2.4482e-04, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: converged after 21 iterations and time 708.8 ms: f = -6.602310927637e-01, ‖∇f‖ = 2.4546e-05

````

We note that this energy is slightly higher than the one obtained from an
[optimization using asymmetric CTMRG](@ref examples_heisenberg) with equivalent settings.
Indeed, this is what one would expect since the $C_{4v}$ symmetry restricts the PEPS ansatz
leading to less free parameters, i.e. an ansatz with a reduced expressivity.
Comparing against Juraj Hasik's data from $J_1\text{-}J_2$
[PEPS simulations](https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat),
we find very good agreement:

````julia
E_ref = -0.6602310934799577 # Juraj's energy at D=2, χ=16 with C4v symmetry
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -1.084830528406467e-9

````

As a consistency check, we can compute the vertical and horizontal correlation lengths,
and should find that they are equal (up to the sparse eigensolver tolerance):

````julia
ξ_h, ξ_v, = correlation_length(peps, env)
@show ξ_h ξ_v;
````

````
ξ_h = [0.6625894995018514]
ξ_v = [0.6625894995018511]

````

## QR-CTMRG

The conventional $C_{4v}$ CTMRG algorithm works by performing an eigendecomposition of
the enlarged corner $A = V D V^\dagger$, taking the diagonal eigenvalue tensor as the new
corner $C' = D$ and then renormalizing the edge using the isometry $V$. There exist modifications
to this standard algorithm, most notably *QR-CTMRG* as presented in a recent publication by
[Zhang, Yang and Corboz](@cite zhang_accelerating_2025). There the idea is to replace the
eigendecomposition by a QR decomposition of a lower-rank approximation of the enlarged corner.
While less accurate in some ways, the QR-CTMRG approach substantially accelerates contraction
and optimization times, and also has vastly improved GPU performance. Notably, it is found
that QR-CTMRG converges to the same fixed point as regular $C_{4v}$ CTMRG.

In PEPSKit terms, using QR-CTMRG just amounts to switching out the projector algorithm that is
used by the [`C4vCTMRG`](@ref) algorithm to `projector_alg = :c4v_qr` (as opposed to `:c4v_eigh`).
QR-CTMRG tends to need significantly more iterations to converge while still being much faster,
hence we need to increase `maxiter`:

````julia
env_qr₀, = leading_boundary(
    env_random_c4v, peps; alg = :c4v, projector_alg = :c4v_qr, maxiter = 500,
);
````

````
[ Info: CTMRG init:	obj = +5.600073842622e-03	err = 1.0000e+00
┌ Warning: CTMRG cancel 500:	obj = +5.924396753059e-01	err = 4.2047306494e-05	time = 0.23 sec
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/ctmrg/ctmrg.jl:168

````

To optimize using QR-CTMRG we proceed analogously by specifiying `projector_alg = :c4v_qr` and
increasing the `maxiter` when setting the boundary algorithm parameters. We make sure to supply
the `env_qr₀` initial environment because it does not use `DiagonalTensorMap`s as its corner
type (only regular `eigh`-based $C_{4v}$ CTMRG produces diagonal corners):

````julia
peps_qr, env_qr, E_qr, = fixedpoint(
    H, peps₀, env_qr₀;
    optimizer_alg = (; tol = 1.0e-4),
    boundary_alg = (; alg = :c4v, projector_alg = :c4v_qr, maxiter = 500),
    gradient_alg = (; alg = :linsolver)
);
@show (E_qr - E_ref) / E_ref;
````

````
[ Info: LBFGS: initializing with f = -5.047653728981e-01, ‖∇f‖ = 1.9060e-01
[ Info: LBFGS: iter    1, Δt  68.5 ms: f = -5.056459386479e-01, ‖∇f‖ = 1.3798e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt 250.7 ms: f = -6.375600504499e-01, ‖∇f‖ = 1.6744e-01, α = 2.79e+01, m = 1, nfg = 5
[ Info: LBFGS: iter    3, Δt  33.8 ms: f = -6.477941757691e-01, ‖∇f‖ = 1.2710e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 255.2 ms: f = -6.489264548265e-01, ‖∇f‖ = 1.2728e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt  56.5 ms: f = -6.520948679742e-01, ‖∇f‖ = 1.9001e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt  53.3 ms: f = -6.556853372287e-01, ‖∇f‖ = 7.4701e-02, α = 3.20e-01, m = 5, nfg = 2
[ Info: LBFGS: iter    7, Δt  19.5 ms: f = -6.577585366615e-01, ‖∇f‖ = 4.6457e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  30.3 ms: f = -6.589067591492e-01, ‖∇f‖ = 5.6776e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt  32.0 ms: f = -6.594497314275e-01, ‖∇f‖ = 2.5279e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt  22.5 ms: f = -6.596013508512e-01, ‖∇f‖ = 1.2057e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  26.8 ms: f = -6.597238062798e-01, ‖∇f‖ = 1.1855e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  19.9 ms: f = -6.598902039179e-01, ‖∇f‖ = 1.2159e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  27.8 ms: f = -6.600647711574e-01, ‖∇f‖ = 9.2790e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt  21.0 ms: f = -6.601721648894e-01, ‖∇f‖ = 3.6997e-03, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  39.7 ms: f = -6.601905466298e-01, ‖∇f‖ = 2.6113e-03, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  28.9 ms: f = -6.602004506851e-01, ‖∇f‖ = 3.2492e-03, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt  21.5 ms: f = -6.602066238957e-01, ‖∇f‖ = 2.9721e-03, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt  30.3 ms: f = -6.602207074042e-01, ‖∇f‖ = 1.5783e-03, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt  39.1 ms: f = -6.602252432623e-01, ‖∇f‖ = 7.4701e-04, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt 250.4 ms: f = -6.602282359103e-01, ‖∇f‖ = 1.2155e-03, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, Δt 126.2 ms: f = -6.602299515427e-01, ‖∇f‖ = 1.0743e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, Δt 253.0 ms: f = -6.602310402232e-01, ‖∇f‖ = 4.7766e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: converged after 23 iterations and time  2.06 s: f = -6.602310919804e-01, ‖∇f‖ = 5.4688e-05
(E_qr - E_ref) / E_ref = -2.2712457151788596e-9

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

