using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bp_expvals(Vector)
end

if CUDA.functional()
    TestSuite.bp_expvals(CuArray)
end

if AMDGPU.functional()
    TestSuite.bp_expvals(ROCArray)
end
