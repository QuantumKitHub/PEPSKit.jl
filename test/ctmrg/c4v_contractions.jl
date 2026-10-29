using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.c4v_ctmrg_contractions_cartesian_spaces(Vector)
    TestSuite.c4v_ctmrg_contractions_u1_spaces(Vector)
    TestSuite.c4v_ctmrg_contractions_fermionic_spaces(Vector)
end

if CUDA.functional()
    TestSuite.c4v_ctmrg_contractions_cartesian_spaces(CuArray)
    TestSuite.c4v_ctmrg_contractions_u1_spaces(CuArray)
    TestSuite.c4v_ctmrg_contractions_fermionic_spaces(CuArray)
end

if AMDGPU.functional()
    TestSuite.c4v_ctmrg_contractions_cartesian_spaces(ROCArray)
    TestSuite.c4v_ctmrg_contractions_u1_spaces(ROCArray)
    TestSuite.c4v_ctmrg_contractions_fermionic_spaces(ROCArray)
end
