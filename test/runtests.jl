using ParallelTestRunner
using PEPSKit

testsuite = ParallelTestRunner.find_tests(@__DIR__)

# CUDA tests: only run if CUDA is functional
using CUDA: CUDA
CUDA.functional() || filter!(!startswith("cuda") ∘ first, testsuite)
# AMDGPU tests: only run if AMDGPU is functional
using AMDGPU
AMDGPU.functional() || filter!(!startswith("amd") ∘ first, testsuite)

# On Buildkite (GPU CI runner): only run CUDA and AMDGPU tests
if get(ENV, "BUILDKITE", "false") == "true"
    f(str) = startswith(first(str), "cuda") || startswith(first(str), "amd")
    filter!(f, testsuite)
end

# --fast to indicate a smaller set of tests
args = parse_args(ARGS; custom = ["fast"])
fast = !isnothing(args.custom["fast"])

const init_code = quote
    const fast_tests = $fast
end

ParallelTestRunner.runtests(PEPSKit, args; testsuite, init_code)
