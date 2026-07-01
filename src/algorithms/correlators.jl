# Correlators in InfinitePEPS

function _correlator_horizontal_pos(
        bra::InfinitePEPS,
        operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
    size(ket) == size(bra) ||
        throw(DimensionMismatch("The ket and bra must have the same unit cell."))
    _issorted_correlator_sites(i, js)
    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))
    # preallocate with correct scalartype
    G = similar(
        js, TensorOperations.promote_contract(
            scalartype(bra), scalartype(ket), scalartype(env), scalartype.(O)...
        ),
    )
    # left start for operator and norm contractions
    c = i # current column being handled
    Vn, Vo = start_correlator_left(c, bra, O[1], ket, env)
    j_last = last(js)
    for (k, j) in enumerate(js)
        local numerator
        while j > c
            c += CartesianIndex(0, 1)
            if c == j
                numerator = end_correlator_right_numerator(j, Vo, bra, O[2], ket, env)
            end
            T = _edge_transfermatrix(c[1], c[2], bra, ket, env)
            c != j_last && (Vo = Vo * T)
            Vn = Vn * T
        end
        # compute overlap without operator
        denominator = end_correlator_right_denominator(j, Vn, env)
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
    return _correlator_horizontal_pos(
        rotated_bra, operator, rotated_i, rotated_js, rotated_ket, rotl90(env)
    )
end

function MPSKit.correlator(
        bra::InfinitePEPS,
        O,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        ket::InfinitePEPS,
        env::CTMRGEnv,
    )
    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return _correlator_horizontal_pos(bra, O, i, js, ket, env)
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

## reserved for InfinitePEPS
function MPSKit.correlator(state::InfinitePEPS, O, i::CartesianIndex{2}, j, env::CTMRGEnv)
    return MPSKit.correlator(state, O, i, j, state, env)
end

# Correlators in InfinitePEPO (tr(ρO))

function _correlator_horizontal_pos(
        ρ::InfinitePEPO, operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    _issorted_correlator_sites(i, js)
    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))
    # preallocate with correct scalartype
    G = similar(
        js, TensorOperations.promote_contract(
            scalartype(ρ), scalartype(env), scalartype.(O)...
        ),
    )
    # left start for operator and norm contractions
    c = i # current column being handled
    Vn, Vo = start_correlator_left(c, ρ, O[1], env)
    j_last = last(js)
    for (k, j) in enumerate(js)
        local numerator
        while j > c
            c += CartesianIndex(0, 1)
            if c == j
                numerator = end_correlator_right_numerator(j, Vo, ρ, O[2], env)
            end
            T = _edge_transfermatrix(c[1], c[2], ρ, env)
            c != j_last && (Vo = Vo * T)
            Vn = Vn * T
        end
        # compute overlap without operator
        denominator = end_correlator_right_denominator(j, Vn, env)
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
    return _correlator_horizontal_pos(
        rotated_ρ, operator, rotated_i, rotated_js, rotl90(env)
    )
end

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        env::CTMRGEnv,
    )
    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return _correlator_horizontal_pos(ρ, O, i, js, env)
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

# utility functions

function _issorted_correlator_sites(i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}})
    return issorted(vcat(i, js); by = last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))
end
