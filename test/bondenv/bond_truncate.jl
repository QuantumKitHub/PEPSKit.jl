using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bondenv_truncate(Vector)
end

if CUDA.functional()
    TestSuite.bondenv_truncate(CuArray)
end

if AMDGPU.functional()
    TestSuite.bondenv_truncate(ROCArray)
end
