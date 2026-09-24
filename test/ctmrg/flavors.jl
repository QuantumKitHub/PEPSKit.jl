using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_flavors_unitcells(Vector)
    TestSuite.ctmrg_flavors_fixedspace_truncation(Vector)
    TestSuite.ctmrg_flavors_c4v(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_flavors_unitcells(CuArray; minimal = true)
    TestSuite.ctmrg_flavors_fixedspace_truncation(CuArray; minimal = true)
    # CUSOLVER doesn't provide heev
    TestSuite.ctmrg_flavors_c4v(CuArray; eigh_alg = :DivideAndConquer, minimal = true)
end

if AMDGPU.functional()
    TestSuite.ctmrg_flavors_unitcells(ROCArray; minimal = true)
    TestSuite.ctmrg_flavors_fixedspace_truncation(ROCArray; minimal = true)
    TestSuite.ctmrg_flavors_c4v(ROCArray; minimal = true)
end
