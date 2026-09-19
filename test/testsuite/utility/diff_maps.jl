using Test
using ChainRulesTestUtils
using PEPSKit: dtmap

function utility_diff_maps(AT)
    # Can the rrule of dtmap be made inferable? (if check_inferred=true, tests error at the moment)
    return @testset "Differentiable tmap ($AT)" begin
        test_rrule(dtmap, x -> x^3, randn(5, 5); check_inferred = false)
    end
end
