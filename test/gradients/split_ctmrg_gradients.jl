@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.gradients_split_gauge(Vector)
TestSuite.gradients_split_sweep(Vector)
TestSuite.gradients_split_fixedpoint(Vector)
