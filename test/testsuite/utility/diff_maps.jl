using Test
using ChainRulesTestUtils
using Adapt, TensorKit, Zygote
using PEPSKit: dtmap, sdiag_pow

function utility_diff_maps(AT)
    # Can the rrule of dtmap be made inferable? (if check_inferred=true, tests error at the moment)
    return @testset "Differentiable maps ($AT)" begin
        test_rrule(dtmap, x -> x^3, randn(5, 5); check_inferred = false)

        @testset "Diagonal powers ($T)" for T in (Float32, ComplexF32, Float64, ComplexF64)
            s = adapt(AT, DiagonalTensorMap(T[0.25, 4], ComplexSpace(2)))
            for power in (0.5, -0.5)
                powered = sdiag_pow(s, power)
                @test storagetype(powered) === storagetype(s)
                @test powered ≈ sdiag_pow(TensorMap(s), power)
                gradient = only(Zygote.gradient(X -> real(sum(sdiag_pow(X, power).data)), s))
                @test storagetype(gradient) === storagetype(s)
                @test gradient.data ≈ power .* s.data .^ (power - 1)
            end
        end
    end
end
