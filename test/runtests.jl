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

# workers peak at ~4.5-6 GiB (examples, AD-heavy tests), well above PTR's 2 GiB default;
# running 4 at once on a 16 GB GitHub runner exhausts memory and the job gets cancelled
ParallelTestRunner.runtests(PEPSKit, args; testsuite, init_code, memory_per_worker = 6 * 2^30)
