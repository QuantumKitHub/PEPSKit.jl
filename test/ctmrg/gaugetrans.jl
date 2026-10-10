using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_gaugetrans(Vector)
    TestSuite.ctmrg_gaugetrans_pepo(Vector)
    TestSuite.ctmrg_gaugetrans_stacked_env(Vector)
end
