using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.toolbox_mpo_terms_dense(Vector)
    TestSuite.toolbox_mpo_bookkeeping(Vector)
    TestSuite.toolbox_mpo_validation(Vector)
end
