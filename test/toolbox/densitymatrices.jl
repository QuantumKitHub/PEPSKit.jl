using Test
using PEPSKit, CUDA, AMDGPU

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    TestSuite.toolbox_single_layer_densitymatrix(Vector)
    TestSuite.toolbox_double_layer_densitymatrix(Vector)
    TestSuite.toolbox_densitymatrix_too_many_layers(Vector)
    TestSuite.toolbox_densitymatrix_generic_fallback(Vector)
end

if CUDA.functional()
    TestSuite.toolbox_single_layer_densitymatrix(CuArray)
    TestSuite.toolbox_double_layer_densitymatrix(CuArray)
end

if AMDGPU.functional()
    TestSuite.toolbox_single_layer_densitymatrix(ROCArray)
    TestSuite.toolbox_double_layer_densitymatrix(ROCArray)
end
