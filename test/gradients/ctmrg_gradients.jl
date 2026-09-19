using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.gradients_asymmetric(Vector)
    TestSuite.gradients_asymmetric_276(Vector)
end
