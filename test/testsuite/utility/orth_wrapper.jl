using Test, Random, LinearAlgebra
using TensorKit, MatrixAlgebraKit, PEPSKit, Zygote, Adapt
using ChainRulesCore: rrule, NoTangent, ZeroTangent

"""Check shared QR/LQ algorithms and pullbacks, including mixed dual fermionic legs."""
function utility_orth_wrapper(AT)
    Random.seed!(73194)
    algorithms = (OrthAdjoint(), OrthAdjoint(; fwd_alg = (; alg = :Householder)))
    spaces = (ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionNumber](-1 => 1, 0 => 1, 1 => 1))
    return @testset "Orthogonalization wrapper ($AT)" begin
        @test QRAdjoint === OrthAdjoint
        @test QRAdjoint() isa OrthAdjoint
        fwd_alg = Householder(; positive = true)
        rrule_alg = PEPSKit.FullPullback(; verbosity = 1)
        configured = OrthAdjoint(; fwd_alg, rrule_alg)
        @test configured.fwd_alg === fwd_alg
        @test configured.rrule_alg === rrule_alg
        @test configured == OrthAdjoint(fwd_alg, rrule_alg)

        @testset "$(typeof(alg.fwd_alg)), $(sectortype(V)), $nout ← $nin, $T" for alg in algorithms, V in spaces, (nout, nin) in ((2, 3), (3, 2)), T in (Float64, ComplexF64)
            output, input = (prod(ntuple(i -> isodd(i) ? V : V', n)) for n in (nout, nin))
            A = adapt(AT, randn(T, output ← input))
            H, K = (adapt(AT, randn(T, W ← W)) for W in (output, input))
            direction = normalize(adapt(AT, randn(T, space(A))))

            @testset "$orth" for (orth, orth!, opposite) in ((left_orth, left_orth!, right_orth), (right_orth, right_orth!, left_orth))
                factorize = X -> orth(X, alg)
                # Explicit tuple adjoints avoid costly inference in the broadcast pullback.
                reference = X -> let (a, b) = opposite(X', alg)
                    (b', a')
                end
                A_before = copy(A)
                a, b = factorize(A)
                @test A == A_before
                @test all(isapprox.((a, b), reference(A)))
                @test a * b ≈ A
                @test orth!(copy(A), alg) == (a, b)

                # The real Gram-matrix loss is gauge invariant and exercises both factors.
                loss = decomposition -> X -> let (a, b) = decomposition(X)
                    real(dot(H, a * a') + dot(K, b' * b))
                end
                f, fref = loss(factorize), loss(reference)
                gradient = only(Zygote.gradient(f, A))
                @test gradient ≈ only(Zygote.gradient(fref, A))
                @test storagetype(gradient) == storagetype(A)
                epsilon = 1.0e-5
                finite = (f(A + epsilon * direction) - f(A - epsilon * direction)) / (2 * epsilon)
                @test real(dot(gradient, direction)) ≈ finite atol = 1.0e-7 rtol = 1.0e-7

                _, back = rrule(orth!, copy(A), alg)
                @test back((ZeroTangent(), ZeroTangent())) == (NoTangent(), ZeroTangent(), NoTangent())
            end
        end
    end
end
