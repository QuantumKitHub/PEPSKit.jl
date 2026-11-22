using Test, Random, LinearAlgebra, TensorKit, MatrixAlgebraKit, PEPSKit, Zygote, Adapt
using MatrixAlgebraKit: TruncatedAlgorithm

function utility_subspace_iteration(AT)
    return @testset "SubspaceIteration decomposition ($AT)" begin
        for subspace in (0, -1, true, truncrank(4))
            @test_throws ArgumentError SubspaceIteration(; subspace)
        end
        @test_throws ArgumentError SubspaceIteration(; subspace = 6, n_subspace_iter = -1)
        @test_throws ArgumentError SubspaceIteration(; subspace = 6, charge_probe = 0)
        @testset "Ranges: $T, q=$q, $W" for T in (Float64, ComplexF64), q in (0, 1), W in (ℂ^8, U1Space(0 => 4, 1 => 4), SU2Space(0 => 4, 1 // 2 => 4))
            Random.seed!(264)
            K = W isa ComplexSpace ? ℂ^5 : typeof(W)(c => 3 for c in sectors(W))
            χ = W isa ComplexSpace ? ℂ^2 : typeof(W)(c => 1 for c in sectors(W))
            M = adapt(AT, zeros(T, W ← W))
            for (_, b) in blocks(M)
                b[diagind(b)] .= 0.2 .^ (0:(size(b, 1) - 1))
            end
            M = first(left_orth(adapt(AT, randn(T, space(M))))) * M * last(right_orth(adapt(AT, randn(T, space(M)))))
            fwd = SubspaceIteration(; subspace = K, n_subspace_iter = q)
            alg = TruncatedAlgorithm(fwd, truncspace(χ))
            local u, s, v
            for _ in 1:12
                u, s, v, = svd_trunc(M, alg)
            end
            ur, sr, vr, = svd_trunc(M, TruncatedAlgorithm(DefaultAlgorithm(), truncspace(χ)))
            @test u * s * v ≈ ur * sr * vr rtol = 1.0e-9
            @test fwd.cache[].X' * fwd.cache[].X ≈ id(T, K)
        end
        @testset "Rectangular complete ranges, $m ← $n" for (m, n) in ((5, 7), (7, 5))
            M = adapt(AT, randn(ComplexF64, ℂ^m ← ℂ^n))
            alg = TruncatedAlgorithm(SubspaceIteration(; subspace = ℂ^20), truncspace(ℂ^2))
            u, s, v, = svd_trunc(M, alg)
            ur, sr, vr, = svd_trunc(M, TruncatedAlgorithm(DefaultAlgorithm(), truncspace(ℂ^2)))
            @test u * s * v ≈ ur * sr * vr rtol = 1.0e-10
            @test dim(domain(alg.alg.cache[].X)) == min(m, n)
        end
        @testset "Charge discovery and reallocation, probe=$probe" for probe in (nothing, 1, 2)
            W = U1Space(0 => 3, 1 => 3, 2 => 3)
            A, B = (zeros(ComplexF64, W ← W) for _ in 1:2)
            for (hot, M) in ((2, A), (1, B)), (c, b) in blocks(M)
                b[diagind(b)] .= c == U1Irrep(hot) ? [100, 90, 80] : c == U1Irrep(0) ? [6, 5, 4] : [3, 2, 1]
            end
            A, B = adapt(AT, A), adapt(AT, B)
            fwd = SubspaceIteration(; subspace = 6, charge_refresh = 1, charge_probe = probe)
            alg = TruncatedAlgorithm(fwd, truncrank(4))
            svd_trunc(A, alg)
            @test fuse(domain(fwd.cache[].X)) == U1Space(0 => 3, 2 => 3)
            for n in 1:3
                svd_trunc(B, alg)
                @test dim(fuse(domain(fwd.cache[].X)), U1Irrep(1)) == (isnothing(probe) ? 3 : min(n * probe, 3))
            end
            @test fuse(domain(fwd.cache[].X)) == U1Space(0 => 3, 1 => 3)
            fixed = TruncatedAlgorithm(SubspaceIteration(; subspace = 6), truncspace(U1Space(0 => 2, 1 => 2)))
            svd_trunc(A, fixed)
            @test fuse(domain(fixed.alg.cache[].X)) == U1Space(0 => 2, 1 => 2, 2 => 2)
        end
        @testset "Factorized contractions: $S" for S in (Trivial, FermionParity, FermionParity ⊠ U1Irrep)
            V = S == Trivial ? ℂ^2 : S == FermionParity ? Vect[S](0 => 1, 1 => 1) : Vect[S]((0, 0) => 1, (1, 1) => 1)
            W = V ⊗ V'
            Q = [adapt(AT, randn(ComplexF64, W ← W)) for _ in 1:4]
            L = PEPSKit._factorized_environment(Q[1], Q[2])
            R = PEPSKit._factorized_environment(Q[3], Q[4])
            M = PEPSKit._factorized_environment(L, R)
            dense = PEPSKit.full_infinite_environment(Q...)
            X = adapt(AT, randn(ComplexF64, W ← fuse(W)))
            @test TensorMap(L) ≈ PEPSKit.half_infinite_environment(Q[1], Q[2])
            @test TensorMap(M) ≈ dense
            @test M * X ≈ dense * X
            @test M' * X ≈ dense' * X
            @test X' * M ≈ X' * dense
        end
    end
end
