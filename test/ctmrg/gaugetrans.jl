using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_gaugetrans(Vector)
    TestSuite.ctmrg_gaugetrans_pepo(Vector)
    TestSuite.ctmrg_gaugetrans_stacked_env(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_gaugetrans(CuArray)
    TestSuite.ctmrg_gaugetrans_pepo(CuArray)
    TestSuite.ctmrg_gaugetrans_stacked_env(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_gaugetrans(ROCArray)
    TestSuite.ctmrg_gaugetrans_pepo(ROCArray)
    TestSuite.ctmrg_gaugetrans_stacked_env(ROCArray)
end
