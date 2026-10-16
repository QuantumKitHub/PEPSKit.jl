using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.compress_fermionic_twists(Vector)
    TestSuite.compress_cost_function(Vector)
    TestSuite.compress_virtual_space_matching(Vector)
end
