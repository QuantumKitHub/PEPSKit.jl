using PEPSKit

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.timeevol_cluster_bond_truncation(Vector)
    TestSuite.timeevol_cluster_identity_gate(Vector)
    TestSuite.timeevol_cluster_hubbard(Vector)
end
