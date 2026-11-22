using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_subspace_iteration(Vector)
end

if CUDA.functional()
    TestSuite.utility_subspace_iteration(CuArray)
end

if AMDGPU.functional()
    TestSuite.utility_subspace_iteration(ROCArray)
end
