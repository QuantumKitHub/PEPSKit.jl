using ParallelTestRunner
using PEPSKit

# --fast to indicate a smaller set of tests
args = parse_args(ARGS; custom = ["fast"])
fast = !isnothing(args.custom["fast"])

const init_code = quote
    const fast_tests = $fast
end

ParallelTestRunner.runtests(PEPSKit, args; init_code)
