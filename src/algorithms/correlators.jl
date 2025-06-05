function correlator_horizontal(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S},
    O₂::AbstractTensorMap{T,S},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}},
) where {T,S}
    @assert size(ket) == size(bra) "The ket and bra must have the same unit cell."
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])
    @assert r == r₂ "Not a horizontal correlation function."
    @assert c₁ < c₂ "The first column index must be less than the second."

    (Nr, Nc) = size(ket)
    corr = T[]

    left_start = start_left(
        env.corners[1, _prev(r, Nr), _prev(c₁, Nc)],
        env.edges[4, mod1(r, Nr), _prev(c₁, Nc)],
        env.corners[4, _next(r, Nr), _prev(c₁, Nc)],
    )
    left_side = transfer_left(
        left_start,
        env.edges[1, _prev(r, Nr), mod1(c₁, Nc)],
        (ket[mod1(r, Nr), mod1(c₁, Nc)], O₁, bra[mod1(r, Nr), mod1(c₁, Nc)]),
        env.edges[3, _next(r, Nr), mod1(c₁, Nc)],
    )
    left_side_norm = transfer_left(
        left_start,
        env.edges[1, _prev(r, Nr), mod1(c₁, Nc)],
        (ket[mod1(r, Nr), mod1(c₁, Nc)], bra[mod1(r, Nr), mod1(c₁, Nc)]),
        env.edges[3, _next(r, Nr), mod1(c₁, Nc)],
    )

    for c in (c₁ + 1):c₂
        left_side_final =
            left_side * HorizontalTransferMatrix(
                env.edges[1, _prev(r, Nr), mod1(c, Nc)],
                (ket[mod1(r, Nr), mod1(c, Nc)], O₂, bra[mod1(r, Nr), mod1(c, Nc)]),
                env.edges[3, _next(r, Nr), mod1(c, Nc)],
            )
        final = end_right(
            left_side_final,
            env.corners[2, _prev(r, Nr), _next(c, Nc)],
            env.edges[2, mod1(r, Nr), _next(c, Nc)],
            env.corners[3, _next(r, Nr), _next(c, Nc)],
        )

        left_side_norm_final =
            left_side_norm * HorizontalTransferMatrix(
                env.edges[1, _prev(r, Nr), mod1(c, Nc)],
                (ket[mod1(r, Nr), mod1(c, Nc)], bra[mod1(r, Nr), mod1(c, Nc)]),
                env.edges[3, _next(r, Nr), mod1(c, Nc)],
            )
        final_norm = end_right(
            left_side_norm_final,
            env.corners[2, _prev(r, Nr), _next(c, Nc)],
            env.edges[2, mod1(r, Nr), _next(c, Nc)],
            env.corners[3, _next(r, Nr), _next(c, Nc)],
        )

        push!(corr, final / final_norm)
        if c ≠ c₂
            (left_side, left_side_norm) = [
                l * HorizontalTransferMatrix(
                    env.edges[1, _prev(r, Nr), mod1(c, Nc)],
                    (ket[mod1(r, Nr), mod1(c, Nc)], bra[mod1(r, Nr), mod1(c, Nc)]),
                    env.edges[3, _next(r, Nr), mod1(c, Nc)],
                ) for l in (left_side, left_side_norm)
            ]
        end
    end
    return corr
end

function correlator_horizontal(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O::AbstractTensorMap{T,S,2,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}},
) where {T,S}
    U, Σ, V = tsvd(O, ((1, 3), (2, 4)))
    O₁ = permute(U * sqrt(Σ), ((1,), (2, 3)))
    O₂ = permute(sqrt(Σ) * V, ((1, 2), (3,)))
    return correlator_horizontal(ket, bra, env, O₁, O₂, inds)
end

function correlator_vertical(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O::AbstractTensorMap{T,S,2,2},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}},
) where {T,S}
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])

    return correlator_horizontal(
        rotr90(ket),
        rotr90(bra),
        rotr90(env),
        O,
        (CartesianIndex(c₁, r), CartesianIndex(c₂, r₂)),
    )
end

function correlator_vertical(
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
    O₁::AbstractTensorMap{T,S},
    O₂::AbstractTensorMap{T,S},
    inds::Tuple{CartesianIndex{2},CartesianIndex{2}};
) where {T,S}
    (r, c₁) = Tuple(inds[1])
    (r₂, c₂) = Tuple(inds[2])

    return correlator_horizontal(
        rotr90(ket),
        rotr90(bra),
        rotr90(env),
        O₁,
        O₂,
        (CartesianIndex(c₁, r), CartesianIndex(c₂, r₂)),
    )
end
