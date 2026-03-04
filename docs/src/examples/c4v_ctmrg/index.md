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
        S_xx(T, S; spin = spin) * Jx +
        S_yy(T, S; spin = spin) * Jy +
        S_zz(T, S; spin = spin) * Jz
    spaces = fill(domain(term)[1], (1, 1))
    return LocalOperator( # horizontal and vertical contributions are identical
        spaces, (CartesianIndex(1, 1), CartesianIndex(1, 2)) => 2 * term
    )
end;
````

## Initializing C₄ᵥ-invariant PEPSs and environments

In order to use $C_{4v}$-symmetric algorithms, it is of course crucial to use initial guesses
with $C_{4v}$ symmetry. First, we create a real-valued random PEPS that we explicitly
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
but passing the initial PEPS and environment as well as the `alg = :c4v` keyword argument:

````julia
env₀, = leading_boundary(env_random_c4v, peps₀; alg = :c4v, tol = 1.0e-10);
````

````
[ Info: CTMRG init:	obj = -1.430301957018e-02	err = 1.0000e+00
[ Info: CTMRG conv 36:	obj = +8.685181513863e+00	err = 6.8459865700e-11	time = 1.30 sec

````

## C₄ᵥ-symmetric optimization

We now take `peps₀` and `env₀` as a starting point for a gradient-based energy
minimization where we contract using $C_{4v}$ CTMRG such that the energy gradient will also
exhibit $C_{4v}$ symmetry. For that, we call `fixedpoint` and specify `alg = :c4v`
as the boundary contraction algorithm:

````julia
H = real(heisenberg_XYZ_c4v(InfiniteSquare())) # make Hamiltonian real-valued
peps, env, E, = fixedpoint(
    H, peps₀, env₀; optimizer_alg = (; tol = 1.0e-4), boundary_alg = (; alg = :c4v),
);
````

````
[ Info: LBFGS: initializing with f = -5.047653728981e-01, ‖∇f‖ = 1.9060e-01
[ Info: LBFGS: iter    1, Δt  1.41 s: f = -5.056459159937e-01, ‖∇f‖ = 1.3798e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt 737.1 ms: f = -6.375541333037e-01, ‖∇f‖ = 1.7202e-01, α = 2.79e+01, m = 1, nfg = 5
[ Info: LBFGS: iter    3, Δt  26.1 ms: f = -6.486427009452e-01, ‖∇f‖ = 1.3183e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt  27.6 ms: f = -6.520903819553e-01, ‖∇f‖ = 1.2693e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt  19.3 ms: f = -6.543775422131e-01, ‖∇f‖ = 8.4368e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt  23.7 ms: f = -6.574414623345e-01, ‖∇f‖ = 9.2421e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt  24.8 ms: f = -6.589599949721e-01, ‖∇f‖ = 4.1336e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  19.2 ms: f = -6.593158985369e-01, ‖∇f‖ = 1.6527e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt  24.1 ms: f = -6.594942583549e-01, ‖∇f‖ = 1.3210e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt  22.2 ms: f = -6.598272997895e-01, ‖∇f‖ = 1.2343e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  18.5 ms: f = -6.600089784768e-01, ‖∇f‖ = 8.5851e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  21.3 ms: f = -6.601648097143e-01, ‖∇f‖ = 3.1456e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  20.4 ms: f = -6.601883355292e-01, ‖∇f‖ = 2.2842e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt  17.2 ms: f = -6.602036974772e-01, ‖∇f‖ = 2.8361e-03, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  20.3 ms: f = -6.602112757039e-01, ‖∇f‖ = 2.0029e-03, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  16.5 ms: f = -6.602199349095e-01, ‖∇f‖ = 1.2421e-03, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt  19.7 ms: f = -6.602252742030e-01, ‖∇f‖ = 7.3767e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt  15.8 ms: f = -6.602292481916e-01, ‖∇f‖ = 6.4543e-04, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt  19.8 ms: f = -6.602308444814e-01, ‖∇f‖ = 3.7333e-04, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt  15.6 ms: f = -6.602310765298e-01, ‖∇f‖ = 2.5277e-04, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: converged after 21 iterations and time  1.42 m: f = -6.602310926956e-01, ‖∇f‖ = 2.8135e-05

````

We note that this energy is slightly higher than the one obtained from an
[optimization using asymmetric CTMRG](@ref examples_heisenberg) with equivalent settings.
Indeed, this is what one would expect since the $C_{4v}$ symmetry restricts the PEPS ansatz
leading to fewer free parameters, i.e. an ansatz with reduced expressivity.
Comparing against Juraj Hasik's data from $J_1\text{-}J_2$
[PEPS simulations](https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat),
we find very good agreement:

````julia
E_ref = -0.6602310934799577 # Juraj's energy at D=2, χ=16 with C4v symmetry
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -1.1879467582794218e-9

````

As a consistency check, we can compute the vertical and horizontal correlation lengths,
and should find that they are equal (up to the sparse eigensolver tolerance):

````julia
ξ_h, ξ_v, = correlation_length(peps, env)
@show ξ_h ξ_v;
````

````
ξ_h = [0.6625965820483917]
ξ_v = [0.6625965820483924]

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
[ Info: CTMRG init:	obj = +5.600046917739e-03	err = 1.0000e+00
┌ Warning: CTMRG cancel 500:	obj = +5.924386039247e-01	err = 3.1337720153e-05	time = 0.23 sec
└ @ PEPSKit ~/repos/PEPSKit.jl/src/algorithms/ctmrg/ctmrg.jl:153

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
[ Info: LBFGS: iter    1, Δt 689.5 ms: f = -5.056459386582e-01, ‖∇f‖ = 1.3798e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt 505.7 ms: f = -6.375600534372e-01, ‖∇f‖ = 1.7220e-01, α = 2.79e+01, m = 1, nfg = 5
[ Info: LBFGS: iter    3, Δt  25.8 ms: f = -6.486459057961e-01, ‖∇f‖ = 1.3187e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt  34.8 ms: f = -6.520930551047e-01, ‖∇f‖ = 1.2680e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt  21.3 ms: f = -6.543790524877e-01, ‖∇f‖ = 8.4444e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt  22.8 ms: f = -6.576333383072e-01, ‖∇f‖ = 8.5748e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt  34.4 ms: f = -6.589645527955e-01, ‖∇f‖ = 4.1392e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  26.4 ms: f = -6.593253867913e-01, ‖∇f‖ = 1.6340e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 175.3 ms: f = -6.595005575055e-01, ‖∇f‖ = 1.3085e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt  20.1 ms: f = -6.598308924091e-01, ‖∇f‖ = 1.2318e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  16.3 ms: f = -6.600131668884e-01, ‖∇f‖ = 8.7100e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  18.9 ms: f = -6.601658264061e-01, ‖∇f‖ = 3.0216e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  14.0 ms: f = -6.601880521970e-01, ‖∇f‖ = 2.4696e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt  19.6 ms: f = -6.602022384904e-01, ‖∇f‖ = 2.3369e-03, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  14.6 ms: f = -6.602097424282e-01, ‖∇f‖ = 1.9888e-03, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  19.8 ms: f = -6.602207442903e-01, ‖∇f‖ = 1.3522e-03, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 161.0 ms: f = -6.602279508700e-01, ‖∇f‖ = 6.7499e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt  99.1 ms: f = -6.602306993252e-01, ‖∇f‖ = 3.3521e-04, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt 185.0 ms: f = -6.602310628576e-01, ‖∇f‖ = 1.8734e-04, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: converged after 20 iterations and time 15.13 s: f = -6.602310908749e-01, ‖∇f‖ = 9.5413e-05
(E_qr - E_ref) / E_ref = -3.945689570922226e-9

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

