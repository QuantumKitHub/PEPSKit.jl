using Enzyme, EnzymeTestUtils
using PEPSKit: dtmap

# Can the rrule of dtmap be made inferable? (if check_inferred=true, tests error at the moment)
@testset "Differentiable tmap" begin
    test_reverse(dtmap, Duplicated, Const(x -> x^3), (randn(5, 5), Duplicated))
end
