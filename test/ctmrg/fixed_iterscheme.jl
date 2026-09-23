using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_fixed_iterscheme_asymmetric(Vector)
    TestSuite.ctmrg_fixed_iterscheme_c4v(Vector)
    TestSuite.ctmrg_fixed_iterscheme_divide_and_conquer(Vector)
end
