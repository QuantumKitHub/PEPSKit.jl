using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bp_gaugefix_bp_vs_su(Vector)
end

if CUDA.functional()
    TestSuite.bp_gaugefix_bp_vs_su(CuArray)
end

if AMDGPU.functional()
    TestSuite.bp_gaugefix_bp_vs_su(ROCArray)
end
