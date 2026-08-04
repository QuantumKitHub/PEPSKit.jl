using ParallelTestRunner
using PEPSKit

testsuite = ParallelTestRunner.find_tests(@__DIR__)

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
