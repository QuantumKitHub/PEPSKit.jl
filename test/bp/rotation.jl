using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.bp_rotations(Vector)
end

if CUDA.functional()
    TestSuite.bp_rotations(CuArray)
end

if AMDGPU.functional()
    TestSuite.bp_rotations(ROCArray)
end
