using ChainRulesTestUtils
using PEPSKit: dtmap

# Can the rrule of dtmap be made inferable? (if check_inferred=true, tests error at the moment)
@testset "Differentiable tmap" begin
    test_rrule(dtmap, x -> x^3, randn(5, 5); check_inferred = false)
end
