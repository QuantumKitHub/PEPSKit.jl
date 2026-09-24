using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bondenv_gaugefix(Vector)
end

if CUDA.functional()
    TestSuite.bondenv_gaugefix(CuArray)
end

if AMDGPU.functional()
    TestSuite.bondenv_gaugefix(ROCArray)
end
