using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_jacobian_real_linear(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_jacobian_real_linear(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_jacobian_real_linear(ROCArray)
end
