@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_flavors_unitcells(Vector)
    TestSuite.ctmrg_flavors_fixedspace_truncation(Vector)
    TestSuite.ctmrg_flavors_c4v(Vector)
end
