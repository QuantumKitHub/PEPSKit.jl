@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite
import PEPSKit

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    PEPSKit.Defaults.set_scheduler!(:serial)
    TestSuite.examples_j1j2(Vector)
end
