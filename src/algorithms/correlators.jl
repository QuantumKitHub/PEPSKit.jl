const CoordCollection{N} = Union{AbstractVector{CartesianIndex{N}}, CartesianIndices{N}}

# Correlators in InfinitePEPS or purified InfinitePEPO

function MPSKit.correlator(
        bra::S,
        O,
        i::CartesianIndex{2}, js::CoordCollection{2},
        ket::S,
        env::CTMRGEnv,
    ) where {S <: InfiniteState}
    return _correlator(_braket_correlator(bra, ket, env), O, i, vec(js))
end

function MPSKit.correlator(
        bra::S,
        O,
        i::CartesianIndex{2}, j::CartesianIndex{2},
        ket::S,
        env::CTMRGEnv,
    ) where {S <: InfiniteState}
    return only(correlator(bra, O, i, j:j, ket, env))
end

## reserved for InfinitePEPS
function MPSKit.correlator(state::InfinitePEPS, O, i::CartesianIndex{2}, j, env::CTMRGEnv)
    return MPSKit.correlator(state, O, i, j, state, env)
end

# Correlators in InfinitePEPO (tr(ρO))

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, js::CoordCollection{2},
        env::CTMRGEnv,
    )
    return _correlator(_PEPOTraceCorrelator(ρ, env), O, i, vec(js))
end

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, j::CartesianIndex{2},
        env::CTMRGEnv,
    )
    return only(correlator(ρ, O, i, j:j, env))
end
