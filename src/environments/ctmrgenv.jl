struct CTMRGEnv{C,T}
    corners::Array{C,3}
    edges::Array{T,3}
end

# initialize ctmrg environments with some random tensors
function CTMRGEnv(peps::InfinitePEPS{P}; χenv=1) where {P}
    envspace = field(space(peps, 1, 1))^χenv  # Environment space 
    C_type = tensormaptype(spacetype(P), 1, 1, storagetype(P))
    T_type = tensormaptype(spacetype(P), 3, 1, storagetype(P))

    # First index is direction
    corners = Array{C_type}(undef, 4, size(peps)...)
    edges = Array{T_type}(undef, 4, size(peps)...)

    for dir in 1:4, i in 1:size(peps, 1), j in 1:size(peps, 2)
        @diffset corners[dir, i, j] = TensorMap(randn, scalartype(P), envspace, envspace)
        @diffset edges[dir, i, j] = TensorMap(
            randn,
            scalartype(P),
            envspace * space(peps[i, j], dir + 1)' * space(peps[i, j], dir + 1),
            envspace,
        )
    end

    @diffset corners[:, :, :] ./= norm.(corners[:, :, :])
    @diffset edges[:, :, :] ./= norm.(edges[:, :, :])

    return CTMRGEnv(corners, edges)
end

function Base.rotl90(env::CTMRGEnv{C,T}) where {C,T}
    corners′ = similar(env.corners)
    edges′ = similar(env.edges)

    for dir in 1:4
        @diffset corners′[_prev(dir, 4), :, :] .= rotl90(env.corners[dir, :, :])
        @diffset edges′[_prev(dir, 4), :, :] .= rotl90(env.edges[dir, :, :])
    end

    return CTMRGEnv(corners′, edges′)
end

Base.eltype(env::CTMRGEnv) = eltype(env.corners[1])
