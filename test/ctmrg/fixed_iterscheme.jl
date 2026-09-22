using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.ctmrg_fixed_iterscheme_asymmetric(Vector)
    TestSuite.ctmrg_fixed_iterscheme_c4v(Vector)
    TestSuite.ctmrg_fixed_iterscheme_divide_and_conquer(Vector)
end

if CUDA.functional()
    TestSuite.ctmrg_fixed_iterscheme_asymmetric(CuArray; svd_alg = :QRIteration)
    # CUSOLVER doesn't provide heev
    TestSuite.ctmrg_fixed_iterscheme_c4v(CuArray; eigh_alg = :DivideAndConquer)
    # CUSOLVER doesn't provide gesdd
    #TestSuite.ctmrg_fixed_iterscheme_divide_and_conquer(CuArray)
end

if AMDGPU.functional()
    TestSuite.ctmrg_fixed_iterscheme_asymmetric(ROCArray)
    TestSuite.ctmrg_fixed_iterscheme_c4v(ROCArray)
    TestSuite.ctmrg_fixed_iterscheme_divide_and_conquer(ROCArray)
end
