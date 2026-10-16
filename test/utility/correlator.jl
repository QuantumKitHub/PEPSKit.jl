using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_correlator_infinite_peps(Vector)
    TestSuite.utility_correlator_purified_ipepo(Vector)
    TestSuite.utility_correlator_single_layer_ipepo(Vector)
end
