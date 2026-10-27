using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.gradients_c4v(Vector)
end

if CUDA.functional()
    TestSuite.gradients_c4v(CuArray; minimal = true)
end

if AMDGPU.functional()
    TestSuite.gradients_c4v(ROCArray; minimal = true)
end
