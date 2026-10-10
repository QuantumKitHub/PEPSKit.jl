using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.boundary_mps_one_one_peps(Vector)
    TestSuite.boundary_mps_two_two_peps(Vector)
    TestSuite.boundary_mps_fermionic_peps(Vector)
    TestSuite.boundary_mps_pepo_runthrough(Vector)
end

if CUDA.functional()
    TestSuite.boundary_mps_one_one_peps(CuArray)
    TestSuite.boundary_mps_two_two_peps(CuArray)
    TestSuite.boundary_mps_fermionic_peps(CuArray)
    TestSuite.boundary_mps_pepo_runthrough(CuArray)
end

if AMDGPU.functional()
    TestSuite.boundary_mps_one_one_peps(ROCArray)
    TestSuite.boundary_mps_two_two_peps(ROCArray)
    TestSuite.boundary_mps_fermionic_peps(ROCArray)
    TestSuite.boundary_mps_pepo_runthrough(ROCArray)
end
