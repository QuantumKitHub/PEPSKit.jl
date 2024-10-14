using Test
using Random
using PEPSKit
using MPSKit
using MPSKit: ∂∂AC, ∂∂C, MPSMultiline, Multiline, fixedpoint,updatetol, vumps_iter
using KrylovKit
using Zygote
using TensorKit
using ChainRulesCore
const vumps_alg = VUMPS(; alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false))

function num_grad(f, K::Number; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end

function num_grad(f, a; δ::Real=1e-5)
    b = copy(a)
    df = map(CartesianIndices(size())) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(ac))
        num_grad(foo, b[i], δ=δ)
    end
    return df
end

@testset "InfinitePEPS" begin
    Random.seed!(42)
    psi = InfinitePEPS(ℂ^2, ℂ^2)
    function f(β)
        psir = β * psi
        norm(psir)
    end
    # @show typeof(psi.A) psi.A
    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "InfiniteTransferPEPS" begin
    Random.seed!(42)
    psi = InfinitePEPS(ℂ^2, ℂ^2)
    function f(β)
        psir = β * psi
        T = PEPSKit.InfiniteTransferPEPS(psir, 1, 1)
        real(dot(T.top, T.bot))
    end

    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "dominant eigensolve" begin
    Random.seed!(42)
    alg_eigsolve = updatetol(vumps_alg.alg_eigsolve, 1, 1)
    psi = InfinitePEPS(ℂ^2, ℂ^2)
    psi = symmetrize!(psi, RotateReflect())

    T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
    mps = PEPSKit.initializeMPS(T, [ℂ^3])
    envs=environments(mps, T)

    mps, T = convert(MPSMultiline, mps), Multiline([T])

    ac = RecursiveVec(mps.AC[:, 1])
    c = RecursiveVec(mps.CR[:, 1])

    S1 = TensorMap(randn, ComplexF64, ℂ^3*ℂ^2*(ℂ^2)'*(ℂ^3)'← ℂ^3*ℂ^2*(ℂ^2)'*(ℂ^3)')
    S2 = TensorMap(randn, ComplexF64, ℂ^3*(ℂ^3)'← ℂ^3*(ℂ^3)')

    function foo1(β)
        psi = β * psi
        T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1) 
        T = MPSKit.Multiline([T])

        H_AC = MPSKit.∂∂AC(1, mps, T, envs)
        
        _, ac′ = MPSKit.fixedpoint(H_AC, ac, :LM, alg_eigsolve)

        AC = ac′.vecs[1]
        @tensor d = conj(AC[1 2 3 4]) * S1[1 2 3 4; 5 6 7 8] * AC[5 6 7 8]
        norm(d)
    end

    @test Zygote.gradient(foo1, 1.0)[1] ≈ num_grad(foo1, 1.0) atol = 1e-4

    # function foo2(β)
    #     psi = β * psi
    #     T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1) 
    #     T = MPSKit.Multiline([T])

    #     H_C = ∂∂C(1, mps, T, envs)
        
    #     _, c′ = MPSKit.fixedpoint(H_C, c, :LM, alg_eigsolve)
    #     C = c′.vecs[1]
    #     # @show space(C)
    #     # norm(C)
    #     @tensor d = conj(C[1 2]) * S2[1 2; 3 4] * C[3 4]
    #     norm(d)
    # end

    # @show foo2(1.0) num_grad(foo2, 1.0)
    # @test Zygote.gradient(foo2, 1.0)[1] ≈ num_grad(foo2, 1.0) atol = 1e-4
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
    AC_0 = mps.AC[1]
    mps, envs, ϵ = vumps_iter(mps, T, vumps_alg, envs, ϵ)
    AC_1 = mps.AC[1]
    # @show norm(AC_0 - AC_1)

    function foo1(psi)
        T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
        mps, envs, ϵ = vumps_iter(mps, T, vumps_alg, envs, ϵ)
        return abs(prod(expectation_value(mps, T)))
    end
    @show foo1(psi)

    @show Zygote.gradient(foo1, psi)[1]
    # @show foo2(1.0)
    # @show norm(Zygote.gradient(foo1, psi)[1] - Zygote.gradient(foo3, psi)[1])
    # @show num_grad(foo2, 1.0)
    # @test N ≈ N´ atol = 1e-3
end