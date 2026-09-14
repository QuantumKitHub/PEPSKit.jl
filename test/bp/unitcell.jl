using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bp_unitcell_random_cartesian_spaces(Vector)
    TestSuite.bp_unitcell_specific_u1_spaces(Vector)
end

if CUDA.functional()
    TestSuite.bp_unitcell_random_cartesian_spaces(CuArray)
    TestSuite.bp_unitcell_specific_u1_spaces(CuArray)
end

if AMDGPU.functional()
    TestSuite.bp_unitcell_random_cartesian_spaces(ROCArray)
    TestSuite.bp_unitcell_specific_u1_spaces(ROCArray)
end
