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
import TensorKitTensors.SpinOperators as SO

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
        SO.S_x_S_x(T, S; spin = spin) * Jx +
        SO.S_y_S_y(T, S; spin = spin) * Jy +
        SO.S_z_S_z(T, S; spin = spin) * Jz
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
but passing the initial PEPS and environment as well as the `alg = :C4vCTMRG` keyword argument:

````julia
env₀, = leading_boundary(env_random_c4v, peps₀; alg = :C4vCTMRG, tol = 1.0e-10);
````

````
[ Info: CTMRG init:	obj = -1.430301957018e-02	err = 1.0000e+00
[ Info: CTMRG conv 36:	obj = +8.685181513863e+00	err = 6.8747905068e-11	time = 1.43 sec

````

## C₄ᵥ-symmetric optimization

We now take `peps₀` and `env₀` as a starting point for a gradient-based energy
minimization where we contract using $C_{4v}$ CTMRG such that the energy gradient will also
exhibit $C_{4v}$ symmetry. For that, we call `fixedpoint` and specify `alg = :C4vCTMRG`
as the boundary contraction algorithm:

````julia
H = real(heisenberg_XYZ_c4v(InfiniteSquare())) # make Hamiltonian real-valued
peps, env, E, = fixedpoint(
    H, peps₀, env₀; optimizer_alg = (; tol = 1.0e-4), boundary_alg = (; alg = :C4vCTMRG),
);
````

````
[ Info: LBFGS: initializing with f = -5.047653728981e-01, ‖∇f‖ = 1.9060e-01
[ Info: LBFGS: iter    1, Δt  1.72 s: f = -5.056459154685e-01, ‖∇f‖ = 1.3798e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt  1.34 s: f = -6.375540411480e-01, ‖∇f‖ = 1.7202e-01, α = 2.79e+01, m = 1, nfg = 5
[ Info: LBFGS: iter    3, Δt  83.0 ms: f = -6.486432922192e-01, ‖∇f‖ = 1.3180e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 132.9 ms: f = -6.520905366689e-01, ‖∇f‖ = 1.2693e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt  64.1 ms: f = -6.543779478695e-01, ‖∇f‖ = 8.4374e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt 126.3 ms: f = -6.574474245322e-01, ‖∇f‖ = 9.2229e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt  67.0 ms: f = -6.589601436841e-01, ‖∇f‖ = 4.1340e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt  55.0 ms: f = -6.593161746362e-01, ‖∇f‖ = 1.6522e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt  93.3 ms: f = -6.594944356059e-01, ‖∇f‖ = 1.3207e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt  53.9 ms: f = -6.598273620830e-01, ‖∇f‖ = 1.2344e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  53.5 ms: f = -6.600090370406e-01, ‖∇f‖ = 8.5852e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  53.9 ms: f = -6.601648157098e-01, ‖∇f‖ = 3.1453e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  76.2 ms: f = -6.601883494928e-01, ‖∇f‖ = 2.2795e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt  53.6 ms: f = -6.602037369209e-01, ‖∇f‖ = 2.8426e-03, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  51.4 ms: f = -6.602113170057e-01, ‖∇f‖ = 2.0017e-03, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  79.4 ms: f = -6.602199370404e-01, ‖∇f‖ = 1.2403e-03, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt  52.7 ms: f = -6.602252410568e-01, ‖∇f‖ = 7.3832e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt  53.0 ms: f = -6.602292169513e-01, ‖∇f‖ = 6.4978e-04, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt  55.2 ms: f = -6.602308383663e-01, ‖∇f‖ = 3.7433e-04, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt  79.6 ms: f = -6.602310776646e-01, ‖∇f‖ = 2.4482e-04, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: converged after 21 iterations and time  2.96 m: f = -6.602310927637e-01, ‖∇f‖ = 2.4546e-05

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
(E - E_ref) / E_ref = -1.0848271652718446e-9

````

As a consistency check, we can compute the vertical and horizontal correlation lengths,
and should find that they are equal (up to the sparse eigensolver tolerance):

````julia
ξ_h, ξ_v, = correlation_length(peps, env)
@show ξ_h ξ_v;
````

````
ξ_h = [0.6625894993211241]
ξ_v = [0.6625894993211242]

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
used by the [`C4vCTMRG`](@ref) algorithm to `projector_alg = :C4vQRProjector` (as opposed to `:C4vEighProjector`).
QR-CTMRG tends to need significantly more iterations to converge while still being much faster,
hence we need to increase `maxiter`:

````julia
env_qr₀, = leading_boundary(
    env_random_c4v, peps; alg = :C4vCTMRG, projector_alg = :C4vQRProjector, maxiter = 500,
);
````

````
[ Info: CTMRG init:	obj = +5.600073848383e-03	err = 1.0000e+00
┌ Warning: CTMRG cancel 500:	obj = +5.924396753022e-01	err = 3.8244504145e-05	time = 0.69 sec
└ @ PEPSKit ~/git/PEPSKit.jl/src/algorithms/ctmrg/ctmrg.jl:170

````

To optimize using QR-CTMRG we proceed analogously by specifiying `projector_alg = :C4vQRProjector` and
increasing the `maxiter` when setting the boundary algorithm parameters. We make sure to supply
the `env_qr₀` initial environment because it does not use `DiagonalTensorMap`s as its corner
type (only regular `eigh`-based $C_{4v}$ CTMRG produces diagonal corners):

````julia
peps_qr, env_qr, E_qr, = fixedpoint(
    H, peps₀, env_qr₀;
    optimizer_alg = (; tol = 1.0e-4),
    boundary_alg = (; alg = :C4vCTMRG, projector_alg = :C4vQRProjector, maxiter = 500),
    gradient_alg = (; solver_alg = (; alg = :GMRES))
);
@show (E_qr - E_ref) / E_ref;
````

````
[ Info: LBFGS: initializing with f = -5.047653728981e-01, ‖∇f‖ = 1.9060e-01
[ Info: LBFGS: iter    1, Δt  1.36 s: f = -5.056459386885e-01, ‖∇f‖ = 1.3798e-01, α = 1.00e+00, m = 0, nfg = 1
[ Info: LBFGS: iter    2, Δt  1.35 s: f = -6.375600089257e-01, ‖∇f‖ = 1.7192e-01, α = 2.79e+01, m = 1, nfg = 5
[ Info: LBFGS: iter    3, Δt 176.1 ms: f = -6.486276446423e-01, ‖∇f‖ = 1.3249e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, Δt 132.7 ms: f = -6.520871588052e-01, ‖∇f‖ = 1.2678e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    5, Δt 171.2 ms: f = -6.543687186065e-01, ‖∇f‖ = 8.4162e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    6, Δt  94.1 ms: f = -6.572356190771e-01, ‖∇f‖ = 9.8853e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    7, Δt 195.5 ms: f = -6.589568069074e-01, ‖∇f‖ = 4.1118e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    8, Δt 117.3 ms: f = -6.593071471815e-01, ‖∇f‖ = 1.6670e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, Δt 268.1 ms: f = -6.594886940312e-01, ‖∇f‖ = 1.3322e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, Δt 116.2 ms: f = -6.598248521120e-01, ‖∇f‖ = 1.2360e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, Δt  77.6 ms: f = -6.600064497969e-01, ‖∇f‖ = 8.5278e-03, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, Δt  85.6 ms: f = -6.601643801022e-01, ‖∇f‖ = 3.2178e-03, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, Δt  80.5 ms: f = -6.601884099261e-01, ‖∇f‖ = 2.2290e-03, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, Δt 124.3 ms: f = -6.602039607999e-01, ‖∇f‖ = 2.9773e-03, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, Δt  86.4 ms: f = -6.602115692272e-01, ‖∇f‖ = 2.0068e-03, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, Δt  89.2 ms: f = -6.602198110522e-01, ‖∇f‖ = 1.2311e-03, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, Δt 135.4 ms: f = -6.602247768790e-01, ‖∇f‖ = 7.6180e-04, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, Δt 591.1 ms: f = -6.602287407414e-01, ‖∇f‖ = 6.9448e-04, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, Δt 292.6 ms: f = -6.602307244266e-01, ‖∇f‖ = 4.2786e-04, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, Δt 579.4 ms: f = -6.602310787528e-01, ‖∇f‖ = 2.3256e-04, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: converged after 21 iterations and time 46.27 s: f = -6.602310929582e-01, ‖∇f‖ = 1.0732e-05
(E_qr - E_ref) / E_ref = -7.902520534221453e-10

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

