using ParallelTestRunner
using PEPSKit

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# remove testsuite
filter!(!(startswith("testsuite") ∘ first), testsuite)

# --fast to indicate a smaller set of tests
args = parse_args(ARGS; custom = ["fast"])
fast = !isnothing(args.custom["fast"])

const init_code = quote
    const fast_tests = $fast
end

ParallelTestRunner.runtests(PEPSKit, args; testsuite, init_code)
