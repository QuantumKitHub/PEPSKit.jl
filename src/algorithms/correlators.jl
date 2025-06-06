function correlator_horizontal(
    bra::InfinitePEPS,
    O::Tuple{AbstractTensorMap{T,S},AbstractTensorMap{T,S}},
    i::CartesianIndex{2},
    j::CartesianIndices{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
) where {T,S}
    @assert size(ket) == size(bra) "The ket and bra must have the same unit cell."
    (r, c₁) = Tuple(i)
    cs = sort([ind[2] for ind in j]; dims=2)
    @assert all([r == ind[1] for ind in j]) "Not a horizontal correlation function."
    @assert all(c₁ .< cs) "The first column index must be less than the second."

    (Nr, Nc) = size(ket)
    corr = T[]

    left_start = start_left(
        env.edges[4, mod1(r, Nr), _prev(c₁, Nc)],
        env.corners[1, _prev(r, Nr), _prev(c₁, Nc)],
        env.corners[4, _next(r, Nr), _prev(c₁, Nc)],
    )
    left_side =
        left_start * MPSKit.TransferMatrix(
            env.edges[1, _prev(r, Nr), mod1(c₁, Nc)],
            (ket[mod1(r, Nr), mod1(c₁, Nc)], O[1], bra[mod1(r, Nr), mod1(c₁, Nc)]),
            env.edges[3, _next(r, Nr), mod1(c₁, Nc)],
        )
    left_side_norm =
        left_start * MPSKit.TransferMatrix(
            env.edges[1, _prev(r, Nr), mod1(c₁, Nc)],
            (ket[mod1(r, Nr), mod1(c₁, Nc)], bra[mod1(r, Nr), mod1(c₁, Nc)]),
            env.edges[3, _next(r, Nr), mod1(c₁, Nc)],
        )
    for c in (c₁ + 1):cs[end]
        if c ∈ cs
            left_side_final =
                left_side * MPSKit.TransferMatrix(
                    env.edges[1, _prev(r, Nr), mod1(c, Nc)],
                    (ket[mod1(r, Nr), mod1(c, Nc)], O[2], bra[mod1(r, Nr), mod1(c, Nc)]),
                    env.edges[3, _next(r, Nr), mod1(c, Nc)],
                )
            final = end_right(
                left_side_final,
                env.edges[2, mod1(r, Nr), _next(c, Nc)],
                env.corners[2, _prev(r, Nr), _next(c, Nc)],
                env.corners[3, _next(r, Nr), _next(c, Nc)],
            )
        end
        (left_side, left_side_norm) = [
            l * MPSKit.TransferMatrix(
                env.edges[1, _prev(r, Nr), mod1(c, Nc)],
                (ket[mod1(r, Nr), mod1(c, Nc)], bra[mod1(r, Nr), mod1(c, Nc)]),
                env.edges[3, _next(r, Nr), mod1(c, Nc)],
            ) for l in (left_side, left_side_norm)
        ]
        if c ∈ cs
            final_norm = end_right(
                left_side_norm,
                env.edges[2, mod1(r, Nr), _next(c, Nc)],
                env.corners[2, _prev(r, Nr), _next(c, Nc)],
                env.corners[3, _next(r, Nr), _next(c, Nc)],
            )
            push!(corr, final / final_norm)
        end
    end
    return corr
end

function correlator_horizontal(
    bra::InfinitePEPS,
    O::AbstractTensorMap{T,S,2,2},
    i::CartesianIndex{2},
    j::CartesianIndices{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
) where {T,S}
    U, Σ, V = tsvd(O, ((1, 3), (2, 4)))
    O₁ = permute(U * sqrt(Σ), ((1,), (2, 3)))
    O₂ = permute(sqrt(Σ) * V, ((1, 2), (3,)))
    return correlator_horizontal(bra, (O₁, O₂), i, j, ket, env)
end

function correlator_vertical(
    bra::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    j::CartesianIndices{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    i_rot = CartesianIndex(i[2], i[1])
    j_rot = CartesianIndex(j[1][2], j[1][1]):CartesianIndex(j[end][2], j[end][1])

    return correlator_horizontal(rotr90(bra), O, i_rot, j_rot, rotr90(ket), rotr90(env))
end

function MPSKit.correlator(
    bra::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    j::CartesianIndices{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    if i[1] == j[1][1]
        return correlator_horizontal(bra, O, i, j, ket, env)
    elseif i[2] == j[1][2]
        return correlator_vertical(bra, O, i, j, ket, env)
    else
        error("The indices must be either horizontal or vertical.")
    end
end

function MPSKit.correlator(
    bra::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    j::CartesianIndex{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    if i[1] == j[1]
        return first(correlator_horizontal(bra, O, i, j:j, ket, env))
    elseif i[2] == j[2]
        return first(correlator_vertical(bra, O, i, j:j, ket, env))
    else
        error("Only horizontal and vertical correlators are implemented.")
    end
end

function MPSKit.correlator(
    state::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    j::Union{CartesianIndex{2},CartesianIndices{2}},
    env::CTMRGEnv,
)
    return MPSKit.correlator(state, O, i, j, state, env)
end
