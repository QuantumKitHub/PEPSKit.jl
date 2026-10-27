using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bondenv_ctm(Vector)
end

if CUDA.functional()
    TestSuite.bondenv_ctm(CuArray)
end

if AMDGPU.functional()
    TestSuite.bondenv_ctm(ROCArray)
end
