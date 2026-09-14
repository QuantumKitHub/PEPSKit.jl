using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_initialization_critical_ising(Vector)
    TestSuite.ctmrg_initialization_peps(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_initialization_critical_ising(CuArray)
    TestSuite.ctmrg_initialization_peps(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_initialization_critical_ising(ROCArray)
    TestSuite.ctmrg_initialization_peps(ROCArray)
end
