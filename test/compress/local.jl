using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.compress_fermionic_twists(Vector)
    TestSuite.compress_cost_function(Vector)
    TestSuite.compress_virtual_space_matching(Vector)
end

if CUDA.functional()
    TestSuite.compress_fermionic_twists(CuArray)
    TestSuite.compress_cost_function(CuArray)
    TestSuite.compress_virtual_space_matching(CuArray)
end

if AMDGPU.functional()
    TestSuite.compress_fermionic_twists(ROCArray)
    TestSuite.compress_cost_function(ROCArray)
    TestSuite.compress_virtual_space_matching(ROCArray)
end
