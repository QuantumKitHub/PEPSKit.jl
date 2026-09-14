using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.timeevol_j1j2_finiteT(Vector)
end

if CUDA.functional()
    TestSuite.timeevol_j1j2_finiteT(CuArray)
end

if AMDGPU.functional()
    TestSuite.timeevol_j1j2_finiteT(ROCArray)
end
