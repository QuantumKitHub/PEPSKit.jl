using Test
using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.toolbox_mpo_routing_fusers(Vector)
    TestSuite.toolbox_mpo_routing_identities(Vector)
    TestSuite.toolbox_mpo_routing_paths(Vector)
end
