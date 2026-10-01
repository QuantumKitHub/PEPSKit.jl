Random.seed!(29384293742893)

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.boundary_mps_one_one_peps(Vector)
    TestSuite.boundary_mps_two_two_peps(Vector)
    TestSuite.boundary_mps_fermionic_peps(Vector)
    TestSuite.boundary_mps_pepo_runthrough(Vector)
end
