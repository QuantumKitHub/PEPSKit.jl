using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.toolbox_tensorproduct_ising(Vector)
    TestSuite.toolbox_tensorproduct_bookkeeping(Vector)
end
