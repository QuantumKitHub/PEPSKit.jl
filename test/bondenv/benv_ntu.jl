using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bondenv_ntu_neighbors(Vector)
    TestSuite.bondenv_ntu(Vector)
end

if CUDA.functional()
    TestSuite.bondenv_ntu_neighbors(CuArray)
    TestSuite.bondenv_ntu(CuArray)
end

if AMDGPU.functional()
    TestSuite.bondenv_ntu_neighbors(ROCArray)
    TestSuite.bondenv_ntu(ROCArray)
end
