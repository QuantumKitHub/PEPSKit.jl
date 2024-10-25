using Test
using Random
using PEPSKit
using PEPSKit: initial_A, left_canonical, right_canonical, leftenv, rightenv
using TensorKit
using Zygote
using LinearAlgebra

begin "test utility"
    ds = [ℂ^2]
    Ds = [ℂ^2]
    χs = [ℂ^4]

    function num_grad(f, K::Number; δ::Real=1e-5)
        if eltype(K) == ComplexF64
            (f(K + δ / 2) - f(K - δ / 2)) / δ + 
                (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
        else
            (f(K + δ / 2) - f(K - δ / 2)) / δ
        end
    end
end

@testset "InfinitePEPS for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipepes = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    function f(β)
        norm(β * ipepes)
    end
    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "InfiniteTransferPEPS for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    function f(β)
        T = PEPSKit.InfiniteTransferPEPS(β * ipeps)
        norm(T.bot .* T.top)
    end

    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "leftenv and rightenv for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs), ifobs in [true, false]
    Random.seed!(50)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, FL = leftenv(AL, adjoint.(AL), itp; ifobs)
    λR, FR = rightenv(AR, adjoint.(AR), itp; ifobs)

    S1 = TensorMap(rand, ComplexF64, χ*D'*D*χ' ← χ*D'*D*χ')
    S2 = TensorMap(rand, ComplexF64, χ*D*D'*χ' ← χ*D*D'*χ')

    function foo1(β)
        ipeps = β * ipeps
        itp = InfiniteTransferPEPS(ipeps)

        _, FL = leftenv(AL, adjoint.(AL), itp; ifobs)

        tol = [@tensor d = conj(FL[1 2 3 4]) * S1[1 2 3 4; 5 6 7 8] * FL[5 6 7 8] for FL in FL]
        return norm(tol)
    end
    @test Zygote.gradient(foo1, 1.0)[1] ≈ num_grad(foo1, 1.0) atol = 1e-8

    function foo2(β)
        ipeps = β * ipeps
        itp = InfiniteTransferPEPS(ipeps)

        _, FR = rightenv(AR, adjoint.(AR), itp; ifobs)

        tol = [@tensor d = conj(FR[1 2 3 4]) * S2[1 2 3 4; 5 6 7 8] * FR[5 6 7 8] for FR in FR]
        return norm(tol)
    end
    @test Zygote.gradient(foo2, 1.0)[1] ≈ num_grad(foo2, 1.0) atol = 1e-8
end

@testset "(1, 1) ctm PEPS" begin
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
    ctm = leading_boundary(CTMRGEnv(psi, ComplexSpace(20)), psi, CTMRG(; verbosity=1))

    boundary_algs = [
        CTMRG(;
            tol=1e-10,
            verbosity=0,
            ctmrgscheme=:simultaneous,
            svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
        ),
        CTMRG(; tol=1e-10, verbosity=0, ctmrgscheme=:sequential),
    ]

    gradtol = 1e-4

    gradmodes = [
        [
            nothing,
            GeomSum(; tol=gradtol, iterscheme=:fixed),
            GeomSum(; tol=gradtol, iterscheme=:diffgauge),
            ManualIter(; tol=gradtol, iterscheme=:fixed),
            ManualIter(; tol=gradtol, iterscheme=:diffgauge),
            LinSolver(; solver=KrylovKit.GMRES(; tol=gradtol), iterscheme=:fixed),
            LinSolver(; solver=KrylovKit.GMRES(; tol=gradtol), iterscheme=:diffgauge),
        ],
        [
            nothing,
            GeomSum(; tol=gradtol, iterscheme=:diffgauge),
            ManualIter(; tol=gradtol, iterscheme=:diffgauge),
            LinSolver(; solver=KrylovKit.GMRES(; tol=gradtol), iterscheme=:diffgauge),
        ],
    ]

    boundary_alg = boundary_algs[1]
    function foo2(psi)
        # psi = x * psi
       
        ctm = PEPSKit.hook_pullback(
                    leading_boundary, ctm, psi, boundary_alg; alg_rrule = gradmodes[1][2]
                )
        abs(norm(psi, ctm))
    end

    function foo3(psi)
        # psi = x * psi
       
        ctm = PEPSKit.hook_pullback(
                    leading_boundary, ctm, psi, boundary_alg; alg_rrule = gradmodes[1][3]
                )
        abs(norm(psi, ctm))
    end
    # @show foo2(1.0)
    # @show norm(Zygote.gradient(foo2, psi)[1] - Zygote.gradient(foo3, psi)[1])
    # @show num_grad(foo2, 1.0)
    # @test N ≈ N´ atol = 1e-3
end

@testset "(1, 1) vumps PEPS " begin
    Random.seed!(42)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
    T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
    mps = PEPSKit.initializeMPS(T, [ComplexSpace(20)])

    mps, envs, ϵ = leading_boundary(mps, T, vumps_alg)
    # AC_0 = mps.AC[1]
    # mps, envs, ϵ = vumps_iter(mps, T, vumps_alg, envs, ϵ)
    # AC_1 = mps.AC[1]
    # @show norm(AC_0 - AC_1)
    @show abs(prod(expectation_value(mps, T)))

    # @show propertynames(envs) envs.dependency envs.lock
    function foo1(psi)
        T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
        # mps, envs, ϵ = vumps_iter(mps, T, vumps_alg, envs, ϵ)
        mps = convert(MPSMultiline, mps)
        T = convert(TransferPEPSMultiline, T)
        ca = environments(mps, T)
        return abs(prod(expectation_value(mps, T, ca)))
    end
    @show foo1(psi)

    @show Zygote.gradient(foo1, psi)[1]
    # @show foo2(1.0)
    # @show norm(Zygote.gradient(foo1, psi)[1] - Zygote.gradient(foo3, psi)[1])
    # @show num_grad(foo2, 1.0)
    # @test N ≈ N´ atol = 1e-3
end