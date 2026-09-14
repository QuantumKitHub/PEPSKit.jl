using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_gaugefix_asymmetric(Vector)
    TestSuite.ctmrg_gaugefix_c4v(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_gaugefix_asymmetric(CuArray)
    TestSuite.ctmrg_gaugefix_c4v(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_gaugefix_asymmetric(ROCArray)
    TestSuite.ctmrg_gaugefix_c4v(ROCArray)
end
