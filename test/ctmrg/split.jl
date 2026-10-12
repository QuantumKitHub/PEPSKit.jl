@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.ctmrg_split_projectors(Vector)
TestSuite.ctmrg_split_environment(Vector)
TestSuite.ctmrg_split_unitcell(Vector)
TestSuite.ctmrg_split_expectation_values(Vector)
