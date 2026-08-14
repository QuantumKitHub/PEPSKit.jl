using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: random_dual!

ds = Dict(
    U1Irrep => U1Space(i => d for (i, d) in zip(-1:1, (1, 1, 2))),
    FermionParity => Vect[FermionParity](0 => 2, 1 => 1)
)
Ds = Dict(
    U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (1, 3, 2))),
    FermionParity => Vect[FermionParity](0 => 3, 1 => 2)
)
Random.seed!(41973582)

@testset "Expectation values of BPEnv ($S)" for S in keys(ds)
    d, D, uc = ds[S], Ds[S], (2, 3)
    ψds = fill(d, uc)
    ψDNs = random_dual!(fill(D, uc))
    ψDEs = random_dual!(fill(D, uc))
    ψ0 = InfinitePEPS(ψds, ψDNs, ψDEs)

    ψ, wts, _ = gauge_fix(ψ0, SUGauge(; maxiter = 100, tol = 1.0e-10))
    for (a0, a) in zip(ψ0.A, ψ.A)
        @test space(a0) == space(a)
    end
    bp_env = BPEnv(wts)
    ctm_env = CTMRGEnv(wts)
    @test ctm_env ≈ CTMRGEnv(bp_env)

    # SU fixed point wts should already be a BP fixed point of ψ
    bp_alg = BeliefPropagation(; miniter = 1, maxiter = 1, tol = 1.0e-7)
    _, err = leading_boundary(bp_env, ψ, bp_alg)
    @test err < 1.0e-9

    op = randn(d → d)
    for site in CartesianIndices(size(ψ))
        lo = LocalOperator(ψds, (site,) => op)
        val1 = expectation_value(ψ, lo, bp_env)
        val2 = expectation_value(ψ, lo, ctm_env)
        @test val1 ≈ val2
    end

    op = randn(d ⊗ d → d ⊗ d)
    vs = [CartesianIndex(1, 0), CartesianIndex(0, 1)]
    for site1 in CartesianIndices(size(ψ)), v in vs
        site2 = site1 + v
        lo = LocalOperator(ψds, (site1, site2) => op)
        val1 = expectation_value(ψ, lo, bp_env)
        val2 = expectation_value(ψ, lo, ctm_env)
        @test val1 ≈ val2
    end
end
