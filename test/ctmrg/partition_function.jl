using Test
using Random
using PEPSKit
using TensorKit
using LinearAlgebra
using QuadGK
using MPSKit

## Setup

"""
    ising_exact(beta, J)

[Exact Onsager solution](https://en.wikipedia.org/wiki/Square_lattice_Ising_model#Exact_solution)
for the 2D classical Ising Model with partition function

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -J \\sum_{\\langle i, j \\rangle} s_i s_j
```
"""
function classical_ising_exact(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J

    k = 1 / sinh(2 * K)^2
    F = quadgk(
        theta -> log(cosh(2 * K)^2 + 1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * theta))), 0, pi
    )[1]
    f = -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)

    m = 1 - (sinh(2 * K))^(-4) > 0 ? (1 - (sinh(2 * K))^(-4))^(1 / 8) : 0

    E = quadgk(theta -> 1 / sqrt(1 - (4 * k) * (1 + k)^(-2) * sin(theta)^2), 0, pi / 2)[1]
    e = -J * cosh(2 * K) / sinh(2 * K) * (1 + 2 / pi * (2 * tanh(2 * K)^2 - 1) * E)

    return f, m, e
end

"""
    classical_ising(; beta=log(1 + sqrt(2)) / 2)

Implements the 2D classical Ising model with partition function

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -J \\sum_{\\langle i, j \\rangle} s_i s_j
```
"""
function classical_ising(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J
    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4] := O[3 4; 2 1] * nt[-3; 3] * nt[-4; 4] * nt[-2; 2] * nt[-1; 1]

    # magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4] := M[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* nt
    @tensor e_hor[-1 -2; -3 -4] :=
        O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * e[-4; 4]
    @tensor e_vert[-1 -2; -3 -4] :=
        O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * e[-3; 3] * nt[-4; 4]
    e = e_hor + e_vert

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end

## Test

# initialize
beta = 0.6
O, M, E = classical_ising(; beta)
Z = InfinitePartitionFunction(O)
Random.seed!(81812781143)

# contract
χenv = ℂ^12
env0 = CTMRGEnv(Z, χenv)

# cover all different flavors
ctm_styles = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs = [HalfInfiniteProjector, FullInfiniteProjector]

@testset "Classical Ising partition function using $ctm_style with $projector_alg" for (
    ctm_style, projector_alg
) in Iterators.product(
    ctm_styles, projector_algs
)
    ctm_alg = ctm_style(; maxiter=150, projector_alg)
    env = leading_boundary(env0, Z, ctm_alg)

    # check observables
    λ = PEPSKit.value(Z, env)
    m = expectation_value((1, 1) => M, Z, env)
    e = expectation_value((1, 1) => E, Z, env)
    f_exact, m_exact, e_exact = classical_ising_exact(; beta)

    # should be real-ish
    @test abs(imag(λ)) < 1e-4
    @test abs(imag(m)) < 1e-4
    @test abs(imag(e)) < 1e-4

    # should match exact solution
    @test -log(λ) / beta ≈ f_exact rtol = 1e-4
    @test abs(m) ≈ abs(m_exact) rtol = 1e-4
    @test e ≈ e_exact rtol = 1e-1 # accuracy limited by bond dimension and maxiter

    # should also work as enlarged matrix or blocked operator
    inds = ((1, 1), (2, 1), (1, 2), (2, 2))
    M_mat = [Z[1] M; Z[1] Z[1]]
    E_mat = [Z[1] Z[1]; E Z[1]]
    m_mat = expectation_value(inds => M_mat, Z, env)
    e_mat = expectation_value(inds => E_mat, Z, env)
    @test m_mat ≈ m rtol = 1e-6
    @test e_mat ≈ e rtol = 1e-6

    @tensor M_block[D_W1 D_W2 D_S1 D_S2; D_N1 D_N2 D_E1 D_E2] :=
        Z[1][D_W1 D1; D_N1 D2] *
        M[D2 D3; D_N2 D_E1] *
        Z[1][D4 D_S2; D3 D_E2] *
        Z[1][D_W2 D_S1; D1 D4]
    @tensor E_block[D_W1 D_W2 D_S1 D_S2; D_N1 D_N2 D_E1 D_E2] :=
        Z[1][D_W1 D1; D_N1 D2] *
        Z[1][D2 D3; D_N2 D_E1] *
        Z[1][D4 D_S2; D3 D_E2] *
        E[D_W2 D_S1; D1 D4]
    m_block = expectation_value(inds => M_block, Z, env)
    e_block = expectation_value(inds => E_block, Z, env)
    @test m_block ≈ m rtol = 1e-6
    @test e_block ≈ e rtol = 1e-6
end
