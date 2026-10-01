using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.enzyme_gradients_asymmetric(Vector)
    TestSuite.enzyme_gradients_asymmetric_276(Vector)
end
