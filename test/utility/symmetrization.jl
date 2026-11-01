using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_symmetrization_reflect_depth(Vector)
    TestSuite.utility_symmetrization_reflect_width(Vector)
    TestSuite.utility_symmetrization_rotate(Vector)
    TestSuite.utility_symmetrization_rotate_reflect(Vector)
end

if CUDA.functional()
    TestSuite.utility_symmetrization_reflect_depth(CuArray)
    TestSuite.utility_symmetrization_reflect_width(CuArray)
    TestSuite.utility_symmetrization_rotate(CuArray)
    TestSuite.utility_symmetrization_rotate_reflect(CuArray)
end

if AMDGPU.functional()
    TestSuite.utility_symmetrization_reflect_depth(ROCArray)
    TestSuite.utility_symmetrization_reflect_width(ROCArray)
    TestSuite.utility_symmetrization_rotate(ROCArray)
    TestSuite.utility_symmetrization_rotate_reflect(ROCArray)
end
