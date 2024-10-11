using Test
using Random
using PEPSKit
using MPSKit
using MPSKit: ∂∂AC, MPSMultiline, Multiline, fixedpoint,updatetol
using KrylovKit
using Zygote
using TensorKit
using ChainRulesCore
const vumps_alg = VUMPS(; alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=true))

function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
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

    S = TensorMap(randn, ComplexF64, ℂ^3*ℂ^2*(ℂ^2)'*(ℂ^3)'← ℂ^3*ℂ^2*(ℂ^2)'*(ℂ^3)')

    function f(β)
        psi = β * psi
        T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1) 
        T = MPSKit.Multiline([T])

        H_AC = MPSKit.∂∂AC(1, mps, T, envs)
        
        _, ac′ = MPSKit.fixedpoint(H_AC, ac, :LM, alg_eigsolve)

        AC = ac′.vecs[1]
        @tensor d = conj(AC[1 2 3 4]) * S[1 2 3 4; 5 6 7 8] * AC[5 6 7 8]
        norm(d)
    end

    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0) atol = 1e-3
end

