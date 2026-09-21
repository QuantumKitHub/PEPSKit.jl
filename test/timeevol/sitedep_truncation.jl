using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.timeevol_sitedep_rotation(Vector)
    TestSuite.timeevol_sitedep_su(Vector)
end
