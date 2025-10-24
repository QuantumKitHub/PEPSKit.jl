const CoordCollection{N} = Union{AbstractVector{CartesianIndex{N}}, CartesianIndices{N}}

# Correlators in InfinitePEPS

function MPSKit.correlator(
        bra::InfinitePEPS,
        O,
        i::CartesianIndex{2}, js::CoordCollection{2},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
    js = vec(js) # map CartesianIndices to actual Vector instead of Matrix

    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return correlator_horizontal(bra, O, i, js, ket, env)
    elseif all(==(i[2]) ∘ last ∘ Tuple, js)
        return correlator_vertical(bra, O, i, js, ket, env)
    else
        error("Only horizontal or vertical correlators are implemented")
    end
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

function MPSKit.correlator(state::InfinitePEPS, O, i::CartesianIndex{2}, j, env::CTMRGEnv)
    return MPSKit.correlator(state, O, i, j, state, env)
end

# Correlators in InfinitePEPO (tr(ρO))

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, js::CoordCollection{2},
        env::CTMRGEnv,
    )
    js = vec(js) # map CartesianIndices to Vector instead of Matrix
    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return correlator_horizontal(ρ, O, i, js, env)
    elseif all(==(i[2]) ∘ last ∘ Tuple, js)
        return correlator_vertical(ρ, O, i, js, env)
    else
        error("Only horizontal or vertical correlators are implemented")
    end
end

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, j::CartesianIndex{2},
        env::CTMRGEnv,
    )
    return only(correlator(ρ, O, i, j:j, env))
end

# TODO: Correlators in InfinitePEPO (⟨ρ|O|ρ⟩)
