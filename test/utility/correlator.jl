using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.utility_correlator_infinite_peps(Vector)
    TestSuite.utility_correlator_purified_ipepo(Vector)
    TestSuite.utility_correlator_single_layer_ipepo(Vector)
end

if CUDA.functional()
    TestSuite.utility_correlator_infinite_peps(CuArray)
    TestSuite.utility_correlator_purified_ipepo(CuArray)
    TestSuite.utility_correlator_single_layer_ipepo(CuArray)
end

if AMDGPU.functional()
    TestSuite.utility_correlator_infinite_peps(ROCArray)
    TestSuite.utility_correlator_purified_ipepo(ROCArray)
    TestSuite.utility_correlator_single_layer_ipepo(ROCArray)
end
