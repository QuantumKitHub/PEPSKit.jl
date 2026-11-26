using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_c4v_krylov_restart(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_c4v_krylov_restart(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_c4v_krylov_restart(ROCArray)
end
