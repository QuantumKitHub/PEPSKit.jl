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

# TODO: remove these functions once TensorKit is updated
function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    # convert to vector of vectors to make use of existing functionality
    vec_of_vecs = [b * sqrtdim(c) for (c, b) in blocks(t)]
    vec, back = FiniteDifferences.to_vec(vec_of_vecs)

    function from_vec(x)
        t′ = similar(t)
        xvec_of_vecs = back(x)
        for (i, (c, b)) in enumerate(blocks(t′))
            scale!(b, xvec_of_vecs[i], isqrtdim(c))
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))
