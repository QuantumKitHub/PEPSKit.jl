using TensorKit
using ChainRulesTestUtils

using TensorKit: sqrtdim, isqrtdim
using VectorInterface: scale!
using FiniteDifferences

## Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return TensorMap(randn, scalartype(x), space(x))
end
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::CTMRGEnv)
    Ctans = x.corners
    Etans = x.edges
    for i in eachindex(x.corners)
        Ctans[i] = rand_tangent(rng, x.corners[i])
    end
    for i in eachindex(x.edges)
        Etans[i] = rand_tangent(rng, x.edges[i])
    end
    return CTMRGEnv(Ctans, Etans)
end
function ChainRulesTestUtils.test_approx(
    actual::AbstractTensorMap, expected::AbstractTensorMap, msg=""; kwargs...
)
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
end
function ChainRulesTestUtils.test_approx(
    actual::InfinitePEPS, expected::InfinitePEPS, msg=""; kwargs...
)
    for i in eachindex(size(actual, 1))
        for j in eachindex(size(actual, 2))
            ChainRulesTestUtils.@test_msg msg isapprox(
                actual[i, j], expected[i, j]; kwargs...
            )
        end
    end
end
function ChainRulesTestUtils.test_approx(
    actual::CTMRGEnv, expected::CTMRGEnv, msg=""; kwargs...
)
    for i in eachindex(actual.corners)
        ChainRulesTestUtils.@test_msg msg isapprox(
            actual.corners[i], expected.corners[i]; kwargs...
        )
    end
    for i in eachindex(actual.edges)
        ChainRulesTestUtils.@test_msg msg isapprox(
            actual.edges[i], expected.edges[i]; kwargs...
        )
    end
end
