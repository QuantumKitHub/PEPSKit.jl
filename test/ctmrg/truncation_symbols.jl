using Test
using TensorKit
using PEPSKit

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    @testset "Truncation strategy from symbol" begin
        V = ℂ^4
        truncs = fill(truncrank(4), 2, 1, 1)
        # (symbol, η, expected strategy without η, expected strategy with η)
        cases = (
            (:notrunc, 4, notrunc(), nothing),
            (:truncerror, 1.0e-6, nothing, truncerror(; atol = 1.0e-6)),
            (:truncrank, 4, nothing, truncrank(4)),
            (:trunctol, 1.0e-6, nothing, trunctol(; atol = 1.0e-6)),
            (:truncspace, V, nothing, truncspace(V)),
            (:FixedSpaceTruncation, 4, FixedSpaceTruncation(), nothing),
            (:SiteDependentTruncation, truncs, nothing, SiteDependentTruncation(truncs)),
        )
        @testset "$alg" for (alg, η, expected, expected_η) in cases
            for (trunc, t) in (((; alg), expected), ((; alg, η), expected_η))
                if isnothing(t)
                    @test_throws ArgumentError SimultaneousCTMRG(; trunc)
                elseif t isa SiteDependentTruncation
                    @test SimultaneousCTMRG(; trunc).projector_alg.trunc.truncs == t.truncs
                else
                    @test SimultaneousCTMRG(; trunc).projector_alg.trunc == t
                end
            end
        end
        @test_throws ArgumentError SimultaneousCTMRG(; trunc = (; alg = :truncrank, η = 4.0))
    end
end
