using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.enzyme_ctmrg_pepo_runthroughs(Vector)
    TestSuite.enzyme_ctmrg_pepo_fixed_point(Vector)
end
