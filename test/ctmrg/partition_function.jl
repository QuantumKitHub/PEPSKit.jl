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
    m = expectation_value(Z, ((1, 1) => M,), env)
    e = expectation_value(Z, ((1, 1) => E,), env)
    f_exact, m_exact, e_exact = classical_ising_exact(; beta)

    # should be real-ish
    @test abs(imag(λ)) < 1e-4
    @test abs(imag(m)) < 1e-4
    @test abs(imag(e)) < 1e-4

    # should match exact solution
    @test -log(λ) / beta ≈ f_exact rtol = 1e-4
    @test abs(m) ≈ abs(m_exact) rtol = 1e-4
    @test e ≈ e_exact rtol = 1e-1 # accuracy limited by bond dimension and maxiter
end

@testset "Classical Ising correlation functions" begin
    ctm_alg = SimultaneousCTMRG(; maxiter=300)
    β = [0.1, log(1 + sqrt(2)) / 2 + 0.05, 2.0]

    # contract at high, critical and low temperature
    O_high, M_high, = classical_ising(; beta=β[1])
    Z_high = InfinitePartitionFunction(O_high)
    env0_high = CTMRGEnv(Z_high, χenv)
    env_high = leading_boundary(env0_high, Z_high, ctm_alg)

    O_crit, M_crit, = classical_ising(; beta=β[2])
    Z_crit = InfinitePartitionFunction(O_crit)
    env0_crit = CTMRGEnv(Z_crit, χenv)
    env_crit = leading_boundary(env0_crit, Z_crit, ctm_alg)

    O_low, M_low, = classical_ising(; beta=β[3])
    Z_low = InfinitePartitionFunction(O_low)
    env0_low = CTMRGEnv(Z_low, χenv)
    env_low = leading_boundary(env0_low, Z_low, ctm_alg)

    # compute correlators
    corr_zz_high = expectation_value(Z_high, ((1, 1) => M_high, (2, 1) => M_high), env_high)
    corr_zz_crit = expectation_value(Z_crit, ((1, 1) => M_crit, (2, 1) => M_crit), env_crit)
    corr_zz_low = expectation_value(Z_low, ((1, 1) => M_low, (2, 1) => M_low), env_low)
    @test abs(corr_zz_high) < abs(corr_zz_crit) < abs(corr_zz_low)
    @test abs(corr_zz_low) ≈ 1.0 rtol = 1e-6

    corr_zzz_high = expectation_value(
        Z_high, ((1, 1) => M_high, (2, 1) => M_high, (1, 2) => M_high), env_high
    )
    corr_zzz_crit = expectation_value(
        Z_crit, ((1, 1) => M_crit, (2, 1) => M_crit, (1, 2) => M_crit), env_crit
    )
    corr_zzz_low = expectation_value(
        Z_low, ((1, 1) => M_low, (2, 1) => M_low, (1, 2) => M_low), env_low
    )
    @test abs(corr_zzz_high) ≈ 0.0 atol = 1e-6
    @test abs(corr_zzz_low) ≈ 1.0 rtol = 1e-6
end
