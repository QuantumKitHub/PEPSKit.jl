using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.gradients_asymmetric(Vector)
    TestSuite.gradients_asymmetric_276(Vector)
end

if CUDA.functional()
    TestSuite.gradients_asymmetric(CuArray)
    TestSuite.gradients_asymmetric_276(CuArray)
end

if AMDGPU.functional()
    TestSuite.gradients_asymmetric(ROCArray)
    TestSuite.gradients_asymmetric_276(ROCArray)
end
