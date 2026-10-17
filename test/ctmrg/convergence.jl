@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_split_gradient_defaults(Vector)
    TestSuite.ctmrg_fixedpoint_forward_info(Vector)
end
