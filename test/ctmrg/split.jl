@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_split_projectors(Vector)
    TestSuite.ctmrg_split_compression(Vector)
    TestSuite.ctmrg_split_fishman_cutoff(Vector)
    TestSuite.ctmrg_split_environment(Vector)
    TestSuite.ctmrg_split_unitcell(Vector)
    TestSuite.ctmrg_split_expectation_values(Vector)
end
