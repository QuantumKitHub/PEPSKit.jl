using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_unitcell_random_cartesian_spaces(Vector)
    TestSuite.ctmrg_unitcell_specific_u1_spaces(Vector)
end
