const CoordCollection{N} = Union{AbstractVector{CartesianIndex{N}}, CartesianIndices{N}}

# Correlators in InfinitePEPS

function correlator_horizontal(
        bra::InfinitePEPS,
        operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
    size(ket) == size(bra) ||
        throw(DimensionMismatch("The ket and bra must have the same unit cell."))
    all(==(i[1]) ∘ first ∘ Tuple, js) ||
        throw(ArgumentError("Not a horizontal correlation function"))
    issorted(vcat(i, js); by = last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))
    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))
    # preallocate with correct scalartype
    G = similar(
        js,
        TensorOperations.promote_contract(
            scalartype(bra), scalartype(ket), scalartype(env), scalartype.(O)...
        ),
    )
    # left start for operator and norm contractions
    Vn, Vo = start_correlator(i, bra, O[1], ket, env)
    i += CartesianIndex(0, 1)
    for (k, j) in enumerate(js)
        # transfer until left of site j
        while j > i
            Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
            Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
            sandwich = (
                ket[mod1(i[1], end), mod1(i[2], end)], bra[mod1(i[1], end), mod1(i[2], end)],
            )
            T = TransferMatrix(Atop, sandwich, _dag(Abot))
            Vo = Vo * T
            twistdual!(T.below, 2:numout(T.below))
            Vn = Vn * T
            i += CartesianIndex(0, 1)
        end
        # compute overlap with operator
        numerator = end_correlator_numerator(j, Vo, bra, O[2], ket, env)
        # transfer right of site j
        Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
        Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
        sandwich = (
            ket[mod1(i[1], end), mod1(i[2], end)], bra[mod1(i[1], end), mod1(i[2], end)],
        )
        T = TransferMatrix(Atop, sandwich, _dag(Abot))
        if k < length(js)
            Vo = Vo * T
        end
        twistdual!(T.below, 2:numout(T.below))
        Vn = Vn * T
        i += CartesianIndex(0, 1)
        # compute overlap without operator
        denominator = end_correlator_denominator(j, Vn, env)
        G[k] = numerator / denominator
    end
    return G
end

function correlator_vertical(
        bra::InfinitePEPS,
        operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
    rotated_bra = rotl90(bra)
    rotated_ket = bra === ket ? rotated_bra : rotl90(ket)
    rotated_i = siterotl90(i, size(bra))
    rotated_js = map(j -> siterotl90(j, size(bra)), js)
    return correlator_horizontal(
        rotated_bra, operator, rotated_i, rotated_js, rotated_ket, rotl90(env)
    )
end

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

function correlator_horizontal(
        ρ::InfinitePEPO, operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    all(==(i[1]) ∘ first ∘ Tuple, js) ||
        throw(ArgumentError("Not a horizontal correlation function"))
    issorted(vcat(i, js); by = last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))
    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))
    # preallocate with correct scalartype
    G = similar(
        js,
        TensorOperations.promote_contract(
            scalartype(ρ), scalartype(env), scalartype.(O)...
        ),
    )
    # left start for operator and norm contractions
    Vn, Vo = start_correlator(i, ρ, O[1], env)
    i += CartesianIndex(0, 1)
    for (k, j) in enumerate(js)
        # transfer until left of site j
        while j > i
            Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
            Amid = trace_physicalspaces(ρ[mod1(i[1], end), mod1(i[2], end)])
            Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
            T = TransferMatrix(Atop, Amid, _dag(Abot))
            Vo = Vo * T
            Vn = Vn * T
            i += CartesianIndex(0, 1)
        end
        # compute overlap with operator
        numerator = end_correlator_numerator(j, Vo, ρ, O[2], env)
        # transfer right of site j
        Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
        Amid = trace_physicalspaces(ρ[mod1(i[1], end), mod1(i[2], end)])
        Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
        T = TransferMatrix(Atop, Amid, _dag(Abot))
        if k < length(js)
            Vo = Vo * T
        end
        Vn = Vn * T
        i += CartesianIndex(0, 1)
        # compute overlap without operator
        denominator = end_correlator_denominator(j, Vn, env)
        G[k] = numerator / denominator
    end
    return G
end

function correlator_vertical(
        ρ::InfinitePEPO, operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        env::CTMRGEnv,
    )
    rotated_ρ = rotl90(ρ)
    unitcell = size(ρ)[1:2]
    rotated_i = siterotl90(i, unitcell)
    rotated_js = map(j -> siterotl90(j, unitcell), js)
    return correlator_horizontal(
        rotated_ρ, operator, rotated_i, rotated_js, rotl90(env)
    )
end

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
