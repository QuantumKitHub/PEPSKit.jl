using Test
using Random
using PEPSKit
using MPSKit
using MPSKit: ∂∂AC, MPSMultiline, Multiline,fixedpoint,updatetol
using KrylovKit
using Zygote
using TensorKit
using ChainRulesTestUtils
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
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
    function f(β)
        psir = β * psi
        norm(psir)
    end
    # @show typeof(psi.A) psi.A
    @test Zygote.gradient(f, 1.0)[1] ≈ num_grad(f, 1.0)
end

@testset "InfiniteTransferPEPS" begin
    Random.seed!(42)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
    function f(β)
        psir = β * psi
        # @show 123 typeof(psir)
        T = PEPSKit.InfiniteTransferPEPS(psir, 1, 1)
        # @show 213123 typeof(T)
        real(dot(T.top, T.bot))
    end
    @show f(1)
    @show Zygote.gradient(f, 1.0)[1]
    # @show num_grad(f, 1.0)
end

@testset "dominant eigensolve" begin
    Random.seed!(42)
    alg_eigsolve = updatetol(vumps_alg.alg_eigsolve, 1, 1)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
    psi = symmetrize!(psi, RotateReflect())

    T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
    mps = PEPSKit.initializeMPS(T, [ComplexSpace(3)])
    envs=environments(mps, T)

    mps, T = convert(MPSMultiline, mps), Multiline([T])

    # envs=environments(mps, T)
    # H_AC = ∂∂AC(1, mps, T, envs)
    ac = RecursiveVec(mps.AC[:, 1])

    # _, ac′ = MPSKit.fixedpoint(H_AC, ac, :LM, alg_eigsolve)
    # @show typeof(H_AC)
    function f(β)
        # InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
        psi = β * psi
        # @show @which PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
        # T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
        # mps, T = convert(MPSMultiline, mps), Multiline([T])

        # @show typeof(mps)  typeof(T) 
        # H_AC = MPSKit.∂∂AC(1, mps, T, envs)
        
        # _, ac′ = MPSKit.fixedpoint(H_AC, ac, :LM, alg_eigsolve)

        # AC = ac′.vecs[1]
        # real(dot(AC, AC))
        @show typeof(T.bot)
        
        real(dot(T.bot,T.bot))
    end
    # # # # # Zygote.gradient(f, 1.0)
    @show f(1.0)
    @show Zygote.gradient(f, 1.0)
    # end
    

end

