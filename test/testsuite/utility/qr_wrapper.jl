using Test, Random, LinearAlgebra
using TensorKit, MatrixAlgebraKit, PEPSKit, Zygote, Adapt
using ChainRulesCore: rrule, NoTangent, ZeroTangent

"""Check LQ factors and pullbacks against adjointed QR, including mixed dual fermionic legs."""
function utility_qr_wrapper(AT)
    Random.seed!(73194)
    qr_alg, lq_alg = QRAdjoint(), LQAdjoint()
    spaces = (ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionNumber](-1 => 1, 0 => 1, 1 => 1))
    return @testset "QR/LQ wrapper ($AT)" begin
        for V in spaces, (nout, nin) in ((2, 3), (3, 2))
            output = prod(ntuple(i -> isodd(i) ? V : V', nout))
            input = prod(ntuple(i -> isodd(i) ? V' : V, nin))
            A = adapt(AT, randn(ComplexF64, output ← input))
            reference = X -> begin
                U, R = left_orth(X', qr_alg)
                return R', U'
            end
            factorize = X -> right_orth(X, lq_alg)
            L, Q = factorize(A)
            Lref, Qref = reference(A)
            @test L ≈ Lref
            @test Q ≈ Qref
            @test L * Q ≈ A
            @test right_orth!(copy(A), lq_alg) == (L, Q)

            H = adapt(AT, randn(ComplexF64, output ← output))
            H = H + H'
            K = adapt(AT, randn(ComplexF64, input ← input))
            K = K + K'
            # Both factors contribute to this gauge-invariant loss.
            loss = decomposition -> X -> begin
                l, q = decomposition(X)
                return real(dot(H, l * l')) + real(dot(K, q' * q))
            end
            gradient = only(Zygote.gradient(loss(factorize), A))
            @test gradient ≈ only(Zygote.gradient(loss(reference), A))
            direction = adapt(AT, randn(ComplexF64, space(A)))
            direction /= norm(direction)
            epsilon = 1.0e-5
            finite = (loss(factorize)(A + epsilon * direction) - loss(factorize)(A - epsilon * direction)) / (2 * epsilon)
            @test real(dot(gradient, direction)) ≈ finite atol = 1.0e-7 rtol = 1.0e-7

            _, back = rrule(right_orth!, copy(A), lq_alg)
            @test back((ZeroTangent(), ZeroTangent())) == (NoTangent(), ZeroTangent(), NoTangent())
        end
    end
end
