using Test
using Random
using PEPSKit
using PEPSKit: initial_A, left_canonical, right_canonical, leftenv, rightenv, ACenv, Cenv, LRtoC, ALCtoAC, ACCtoALAR
using TensorKit
using Zygote
using LinearAlgebra

begin "test utility"
    ds = [ℂ^2]
    Ds = [ℂ^2]
    χs = [ℂ^5]

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

@testset "leftenv and rightenv for unitcell $Ni x $Nj" for Ni in 1:1, Nj in 1:1, (d, D, χ) in zip(ds, Ds, χs), ifobs in [true, false]
    Random.seed!(50)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    A = initial_A(ipeps, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, FL = leftenv(AL, adjoint.(AL), ipeps; ifobs)
    λR, FR = rightenv(AR, adjoint.(AR), ipeps; ifobs)

    S1 = TensorMap(rand, ComplexF64, χ*D'*D*χ' ← χ*D'*D*χ')
    S2 = TensorMap(rand, ComplexF64, χ*D*D'*χ' ← χ*D*D'*χ')

    function foo1(ipeps)
        # ipeps = β * ipeps

        _, FL = leftenv(AL, adjoint.(AL), ipeps; ifobs)

        tol = [(@tensor conj(FL[1 2 3 4]) * S1[1 2 3 4; 5 6 7 8] * FL[5 6 7 8]) / dot(FL, FL) for FL in FL]
        return norm(tol)
    end
    # @test Zygote.gradient(foo1, 1.0)[1] ≈ num_grad(foo1, 1.0) atol = 1e-8
    @show typeof(Zygote.gradient(foo1, ipeps)[1])
    # function foo2(β)
    #     ipeps = β * ipeps

    #     _, FR = rightenv(AR, adjoint.(AR), ipeps; ifobs)

    #     tol = [(@tensor conj(FR[1 2 3 4]) * S2[1 2 3 4; 5 6 7 8] * FR[5 6 7 8]) / dot(FR, FR) for FR in FR]
    #     return norm(tol)
    # end
    # @test Zygote.gradient(foo2, 1.0)[1] ≈ num_grad(foo2, 1.0) atol = 1e-8
end

@testset "ACenv and Cenv for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(50)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    A = initial_A(ipeps, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    _, FL = leftenv(AL, adjoint.(AL), ipeps)
    _, FR = rightenv(AR, adjoint.(AR), ipeps)
     C =   LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    S1 = TensorMap(randn, ComplexF64, χ*D*D'*χ' ← χ*D*D'*χ')
    S2 = TensorMap(randn, ComplexF64, χ*χ' ← χ*χ')

    function foo1(β)
        ipeps = β * ipeps
        _, AC = ACenv(AC, FL, FR, ipeps)

        tol = [(@tensor conj(AC[1 2 3 4]) * S1[1 2 3 4; 5 6 7 8] * AC[5 6 7 8]) / dot(AC, AC) for AC in AC]
        return norm(tol)
    end
    @test Zygote.gradient(foo1, 1.0)[1] ≈ num_grad(foo1, 1.0) atol = 1e-8

    function foo2(β)
        _, C = Cenv(C, β*FL, β*FR)

        tol = [(@tensor d = conj(C[1 2]) * S2[1 2; 3 4] * C[3 4]) / dot(C, C) for C in C]
        return norm(tol)
    end
    @test Zygote.gradient(foo2, 1.0)[1] ≈ num_grad(foo2, 1.0) atol = 1e-8
end

@testset "ACCtoALAR for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    A = initial_A(ipeps, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, FL = leftenv(AL, adjoint.(AL), ipeps)
    λR, FR = rightenv(AR, adjoint.(AR), ipeps)

     C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, FR, ipeps)
     λC,  C =  Cenv( C, FL, FR) 

    S = TensorMap(rand, ComplexF64, χ*D*D'*χ' ← χ*D*D'*χ')
    function foo1(β)
        ipeps = β * ipeps
        _, AC = ACenv(AC, FL, FR, ipeps)
        AL, AR, _, _ = ACCtoALAR(AC, C)
        tol1 = [(@tensor conj(AL[1 2 3 4]) * S[1 2 3 4; 5 6 7 8] * AL[5 6 7 8]) / dot(AL, AL) for AL in AL]
        tol2 = [(@tensor conj(AR[1 2 3 4]) * S[1 2 3 4; 5 6 7 8] * AR[5 6 7 8]) / dot(AR, AR) for AR in AR]
        return norm(tol1 + tol2)
    end
    @test Zygote.gradient(foo1, 1.0)[1] ≈ num_grad(foo1, 1.0) atol = 1e-8
end

@testset "ad vumps iPEPS one side for unitcell $Ni x $Nj" for Ni in 1:1, Nj in 1:1, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(50)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    ipeps = symmetrize!(ipeps, RotateReflect())

    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=false)
    rt = leading_boundary(VUMPSRuntime(ipeps, χ, alg), ipeps, alg)

    function foo1(ipeps)
        rt = leading_boundary(rt, ipeps, alg)
        env = VUMPSEnv(rt)
        Z = abs(norm(ipeps, env))
        return Z
    end

    function foo2(ipeps)
        ctm = leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
        Z = abs(norm(ipeps, ctm))^(1/Ni/Nj)
        return Z
    end

    # @show foo1(ipeps) - foo2(ipeps)

    @test norm(Zygote.gradient(foo1, ipeps)[1].A - Zygote.gradient(foo2, ipeps)[1].A) < 1e-8 
end

@testset "_fit_spaces" for (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    A = TensorMap(randn, ComplexF64, d ← D*D*D'*D')
    B = TensorMap(randn, ComplexF64, d ← D*D*D'*D)

    function f(β)
        C = PEPSKit._fit_spaces(β * B, A)
        return norm(C)
    end

    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "ad vumps iPEPS two side for unitcell $Ni x $Nj" for Ni in 1:1, Nj in 1:1, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(50)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    ipeps = symmetrize!(ipeps, RotateReflect())
    
    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=true)
    rt = leading_boundary(VUMPSRuntime(ipeps, χ, alg), ipeps, alg)

    function foo1(ipeps)
        rt = leading_boundary(rt, ipeps, alg)
        env = VUMPSEnv(rt, ipeps)
        Z = abs(norm(ipeps, env))
        return Z
    end

    function foo2(ipeps)
        ctm = leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
        Z = abs(norm(ipeps, ctm))^(1/Ni/Nj)
        return Z
    end
    @show foo1(ipeps) - foo2(ipeps)
    @show Zygote.gradient(foo1, ipeps)[1]  Zygote.gradient(foo2, ipeps)[1].A
    # @show Zygote.gradient(foo1, ipeps)[1].A - Zygote.gradient(foo2, ipeps)[1].A
    # @test norm(Zygote.gradient(foo1, ipeps)[1].A - Zygote.gradient(foo2, ipeps)[1].A) < 1e-8 
end