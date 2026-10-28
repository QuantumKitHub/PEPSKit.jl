using Test
using TensorKit
using PEPSKit
using Adapt
using JLArrays

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    @testset "Adapt methods for PEPSKit types" begin
        @testset "LocalOperator" begin
            vd = ℂ^2
            t = randn(ComplexF64, vd ⊗ vd ← vd ⊗ vd)
            physical_spaces = fill(vd, (2, 2))

            terms = ((CartesianIndex(1, 1), CartesianIndex(1, 2)) => t,)
            op = LocalOperator(physical_spaces, terms...)
            @test TensorKit.storagetype(op) == TensorKit.storagetype(t)
            jl_op = adapt(JLArray, op)
            jl_t = adapt(JLArray, t)
            @test TensorKit.storagetype(jl_op) == TensorKit.storagetype(jl_t)
            @test TensorKit.storagetype(jl_op) <: JLArray
        end
        @testset "InfinitePEPS/InfinitePEPO" begin
            Vphy = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 2)
            Vs = (
                # Espace
                Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 3, (1, -1 // 2) => 2),
                # Nspace
                Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 4),
            )
            Nr, Nc = 2, 3
            peps = InfinitePEPS(rand, Float64, Vphy, Vs[2], Vs[1]'; unitcell = (Nr, Nc))
            pepo = InfinitePEPO(rand, Float64, Vphy, Vs[2], Vs[1]'; unitcell = (Nr, Nc, 1))
            jl_peps = adapt(JLArray, peps)
            jl_pepo = adapt(JLArray, pepo)
            @test TensorKit.storagetype(jl_peps) <: JLArray
            @test TensorKit.storagetype(jl_pepo) <: JLArray
        end
        @testset "InfinitePartitionFunction" begin
            S = Trivial
            V = ComplexSpace(3)
            sz = (1, 1)
            pf = InfinitePartitionFunction(randn, ComplexF64, V; unitcell = sz)
            jl_pf = adapt(JLArray, pf)
            @test TensorKit.storagetype(jl_pf) <: JLArray
        end
        @testset "CTMRGEnv" begin
            function get_spaces(sym::Type{<:Sector})
                Nr, Nc = 2, 2
                Vphy = Vect[sym](0 => 1, 1 => 1)
                V = Vect[sym](0 => 1, 1 => 2)
                Venv = Vect[sym](0 => 2, 1 => 2)
                Nspaces = [V' V; V V']
                Espaces = [V V'; V' V]
                return Vphy, Venv, Nspaces, Espaces
            end
            Vphy, Venv, Nspaces, Espaces = get_spaces(Z2Irrep)
            for Vp in [Vphy]
                Pspaces = fill(Vp, size(Nspaces))
                peps = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
                env = CTMRGEnv(randn, ComplexF64, peps, Venv)
                jl_env = adapt(JLArray, env)
                @test TensorKit.storagetype(jl_env) <: JLArray
            end
        end
        @testset "C4vCTMRGEnv" begin
            Vphy = Z2Space(0 => 1, 1 => 1)
            V = Z2Space(0 => 1, 1 => 2)
            Venv = Z2Space(0 => 2, 1 => 2)
            peps = InfinitePEPS(randn, ComplexF64, Vphy, V)
            symmetrize!(peps, RotateReflect())
            env = initialize_random_c4v_env(peps, Venv)
            jl_env = adapt(JLArray, env)
            @test jl_env isa C4vCTMRGEnv
            @test size(jl_env.corners) == (1, 1, 1)
            @test size(jl_env.edges) == (4, 1, 1)
            @test TensorKit.storagetype(jl_env) <: JLArray
            @test adapt(Array, jl_env) ≈ env
        end
    end
end
