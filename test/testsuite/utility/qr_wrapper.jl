using Test, Random, LinearAlgebra
using TensorKit, MatrixAlgebraKit, PEPSKit, Zygote, Adapt
using ChainRulesCore: rrule, NoTangent, ZeroTangent

"""Check LQ factors and pullbacks against adjointed QR, including mixed dual fermionic legs."""
function utility_qr_wrapper(AT)
    Random.seed!(73194)
    qr_alg, lq_alg = QRAdjoint(), LQAdjoint()
    factorize = X -> right_orth(X, lq_alg)
    reference = X -> reverse(adjoint.(left_orth(X', qr_alg)))
    spaces = (ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionNumber](-1 => 1, 0 => 1, 1 => 1))
    return @testset "QR/LQ wrapper ($(sectortype(V)), $nout ← $nin, $AT)" for V in spaces, (nout, nin) in ((2, 3), (3, 2))
        output, input = (prod(ntuple(i -> isodd(i) ? V : V', n)) for n in (nout, nin))
        A = adapt(AT, randn(ComplexF64, output ← input))
        L, Q = factorize(A)
        @test all(isapprox.((L, Q), reference(A)))
        @test L * Q ≈ A
        @test right_orth!(copy(A), lq_alg) == (L, Q)

        H, K = (adapt(AT, randn(ComplexF64, W ← W)) for W in (output, input))
        # The real Gram-matrix loss is gauge invariant and exercises both factors.
        loss = decomposition -> X -> let (l, q) = decomposition(X)
            real(dot(H, l * l') + dot(K, q' * q))
        end
        f, fref = loss(factorize), loss(reference)
        gradient = only(Zygote.gradient(f, A))
        @test gradient ≈ only(Zygote.gradient(fref, A))
        direction = normalize(adapt(AT, randn(ComplexF64, space(A))))
        epsilon = 1.0e-5
        finite = (f(A + epsilon * direction) - f(A - epsilon * direction)) / (2 * epsilon)
        @test real(dot(gradient, direction)) ≈ finite atol = 1.0e-7 rtol = 1.0e-7

        _, back = rrule(right_orth!, copy(A), lq_alg)
        @test back((ZeroTangent(), ZeroTangent())) == (NoTangent(), ZeroTangent(), NoTangent())
    end
end
