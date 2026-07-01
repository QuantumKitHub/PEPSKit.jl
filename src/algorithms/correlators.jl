const CoordCollection{N} = Union{AbstractVector{CartesianIndex{N}}, CartesianIndices{N}}

# Correlators in InfinitePEPS

function MPSKit.correlator(
        bra::InfinitePEPS,
        O,
        i::CartesianIndex{2}, js::CoordCollection{2},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
    return _correlator(_PEPSCorrelator(bra, ket, env), O, i, vec(js))
end

function MPSKit.correlator(
        bra::InfinitePEPS,
        O,
        i::CartesianIndex{2}, j::CartesianIndex{2},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
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

# TODO: Correlators in InfinitePEPO (⟨ρ|O|ρ⟩)
