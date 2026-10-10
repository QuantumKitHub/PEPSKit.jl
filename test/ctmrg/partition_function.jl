using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_partition_function_spaces(Vector)
    TestSuite.ctmrg_partition_function(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_partition_function_spaces(CuArray)
    TestSuite.ctmrg_partition_function(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_partition_function_spaces(ROCArray)
    TestSuite.ctmrg_partition_function(ROCArray)
end
