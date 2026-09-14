using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_pepo_runthroughs(Vector)
    TestSuite.ctmrg_pepo_fixed_point(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_pepo_runthroughs(CuArray)
    TestSuite.ctmrg_pepo_fixed_point(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_pepo_runthroughs(ROCArray)
    TestSuite.ctmrg_pepo_fixed_point(ROCArray)
end
