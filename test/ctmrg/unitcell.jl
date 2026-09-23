using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_unitcell_random_cartesian_spaces(Vector)
    TestSuite.ctmrg_unitcell_specific_u1_spaces(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_unitcell_random_cartesian_spaces(CuArray; minimal = true)
    TestSuite.ctmrg_unitcell_specific_u1_spaces(CuArray; minimal = true)
end

if AMDGPU.functional()
    TestSuite.ctmrg_unitcell_random_cartesian_spaces(ROCArray; minimal = true)
    TestSuite.ctmrg_unitcell_specific_u1_spaces(ROCArray; minimal = true)
end
