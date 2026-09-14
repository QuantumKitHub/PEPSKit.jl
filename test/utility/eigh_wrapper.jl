using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_eigh_wrapper(Vector)
end

if CUDA.functional()
    TestSuite.utility_eigh_wrapper(CuArray)
end

if AMDGPU.functional()
    TestSuite.utility_eigh_wrapper(ROCArray)
end
