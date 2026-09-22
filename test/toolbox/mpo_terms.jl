using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.toolbox_mpo_convention(Vector)
    TestSuite.toolbox_mpo_dispatch(Vector)
    TestSuite.toolbox_mpo_terms_dense(Vector)
    TestSuite.toolbox_mpo_heisenberg(Vector)
    TestSuite.toolbox_mpo_bookkeeping(Vector)
end
