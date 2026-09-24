using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.timeevol_cluster_bond_truncation(Vector)
    TestSuite.timeevol_cluster_identity_gate(Vector)
    TestSuite.timeevol_cluster_hubbard(Vector)
end

if CUDA.functional()
    TestSuite.timeevol_cluster_bond_truncation(CuArray)
    TestSuite.timeevol_cluster_identity_gate(CuArray)
    TestSuite.timeevol_cluster_hubbard(CuArray)
end

if AMDGPU.functional()
    TestSuite.timeevol_cluster_bond_truncation(ROCArray)
    TestSuite.timeevol_cluster_identity_gate(ROCArray)
    TestSuite.timeevol_cluster_hubbard(ROCArray)
end
