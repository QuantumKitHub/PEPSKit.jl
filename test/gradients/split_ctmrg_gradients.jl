@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.gradients_split_gauge(Vector)
    TestSuite.gradients_split_sweep(Vector)
    TestSuite.gradients_split_fixedpoint(Vector)
end
