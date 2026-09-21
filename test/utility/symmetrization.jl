using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_symmetrization_reflect_depth(Vector)
    TestSuite.utility_symmetrization_reflect_width(Vector)
    TestSuite.utility_symmetrization_rotate(Vector)
    TestSuite.utility_symmetrization_rotate_reflect(Vector)
end
