using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_eigh_wrapper(Vector)
end

if CUDA.functional()
    # CUSOLVER doesn't provide QRIteration for eigh
    TestSuite.utility_eigh_wrapper(CuArray; default_alg = :DivideAndConquer)
end

if AMDGPU.functional()
    TestSuite.utility_eigh_wrapper(ROCArray)
end
