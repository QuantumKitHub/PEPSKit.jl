struct CTMRGEnv{C,T}
    corners::Array{C,3}
    edges::Array{T,3}
end

# Initialize ctmrg environments with some random tensors
function CTMRGEnv(peps::InfinitePEPS{P}; Venv=oneunit(spacetype(P))) where {P}
    C_type = tensormaptype(spacetype(P), 1, 1, storagetype(P))
    T_type = tensormaptype(spacetype(P), 3, 1, storagetype(P))

    # First index is direction
    corners = Array{C_type}(undef, 4, size(peps)...)
    edges = Array{T_type}(undef, 4, size(peps)...)

    for dir in 1:4, i in 1:size(peps, 1), j in 1:size(peps, 2)
        @diffset corners[dir, i, j] = TensorMap(randn, scalartype(P), Venv, Venv)
        @diffset edges[dir, i, j] = TensorMap(
            randn,
            scalartype(P),
            Venv * space(peps[i, j], dir + 1)' * space(peps[i, j], dir + 1),
            Venv,
        )
    end

    @diffset corners[:, :, :] ./= norm.(corners[:, :, :])
    @diffset edges[:, :, :] ./= norm.(edges[:, :, :])

    return CTMRGEnv(corners, edges)
end

# Custom adjoint for CTMRGEnv constructor, needed for fixed-point differentiation
function ChainRulesCore.rrule(::Type{CTMRGEnv}, corners, edges)
    ctmrgenv_pullback(ē) = NoTangent(), ē.corners, ē.edges
    return CTMRGEnv(corners, edges), ctmrgenv_pullback
end

# Rotate corners & edges counter-clockwise
function Base.rotl90(env::CTMRGEnv{C,T}) where {C,T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C,3}(undef, 4, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T,3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))

    for dir in 1:4
        corners′[_prev(dir, 4), :, :] = rotl90(env.corners[dir, :, :])
        edges′[_prev(dir, 4), :, :] = rotl90(env.edges[dir, :, :])
    end

    return CTMRGEnv(copy(corners′), copy(edges′))
end

Base.eltype(env::CTMRGEnv) = eltype(env.corners[1])

# Functions used for FP differentiation and by KrylovKit.linsolve
function Base.:+(e₁::CTMRGEnv, e₂::CTMRGEnv)
    return CTMRGEnv(e₁.corners + e₂.corners, e₁.edges + e₂.edges)
end

function Base.:-(e₁::CTMRGEnv, e₂::CTMRGEnv)
    return CTMRGEnv(e₁.corners - e₂.corners, e₁.edges - e₂.edges)
end

Base.:*(α::Number, e::CTMRGEnv) = CTMRGEnv(α * e.corners, α * e.edges)
Base.similar(e::CTMRGEnv) = CTMRGEnv(similar(e.corners), similar(e.edges))

function LinearAlgebra.mul!(edst::CTMRGEnv, esrc::CTMRGEnv, α::Number)
    edst.corners .= α * esrc.corners
    edst.edges .= α * esrc.edges
    return edst
end

function LinearAlgebra.rmul!(e::CTMRGEnv, α::Number)
    rmul!(e.corners, α)
    rmul!(e.edges, α)
    return e
end

function LinearAlgebra.axpy!(α::Number, e₁::CTMRGEnv, e₂::CTMRGEnv)
    e₂.corners .+= α * e₁.corners
    e₂.edges .+= α * e₁.edges
    return e₂
end

function LinearAlgebra.axpby!(α::Number, e₁::CTMRGEnv, β::Number, e₂::CTMRGEnv)
    e₂.corners .= α * e₁.corners + β * e₂.corners
    e₂.edges .= α * e₁.edges + β * e₂.edges
    return e₂
end

function LinearAlgebra.dot(e₁::CTMRGEnv, e₂::CTMRGEnv)
    return dot(e₁.corners, e₂.corners) + dot(e₁.edges, e₂.edges)
end

LinearAlgebra.norm(e::CTMRGEnv) = norm(e.corners) + norm(e.edges)

# VectorInterface (TODO: implement !! methods)
VectorInterface.scalartype(e::CTMRGEnv) = eltype(e.corners[1])

# VectorInterface.zerovector(e::CTMRGEnv) = zerovector(e, scalartype(e))  # Why does uncommenting this error?
function VectorInterface.zerovector(e::CTMRGEnv, ::Type{S}) where {S<:Number}
    return CTMRGEnv(
        map(c -> TensorMap(zeros, S, space(c)), e.corners),
        map(t -> TensorMap(zeros, S, space(t)), e.edges),
    )
end
function VectorInterface.zerovector!(e::CTMRGEnv)
    e.corners .= map(c -> TensorMap(zeros, S, space(c)), e.corners)
    e.edges .= map(t -> TensorMap(zeros, S, space(t)), e.edges)
    return e
end
# function VectorInterface.zerovector!!(e::CTMRGEnv, S::Number)
#     return CTMRGEnv(
#         map(c -> TensorMap(zeros, S, space(c)), e.corners),
#         map(t -> TensorMap(zeros, S, space(t)), e.edges),
#     )
# end
# VectorInterface.zerovector!!(e::CTMRGEnv) = zerovector!!(e, scalartype(e))

VectorInterface.scale(e::CTMRGEnv, α::Number) = CTMRGEnv(α * e.corners, α * e.edges)
function VectorInterface.scale!(e::CTMRGEnv, α::Number)
    e.corners .*= α
    e.edges .*= α
    return e
end
function VectorInterface.scale!(e1::CTMRGEnv, e2::CTMRGEnv, α)
    e1.corners .= α * e2.corners
    e1.edges .= α * e2.edges
    return e1
end
# VectorInterface.scale!!(e::CTMRGEnv, α::Number) = CTMRGEnv(α * e.corners, α * e.edges)
# function VectorInterface.scale!!(e1::CTMRGEnv, e2::CTMRGEnv, α::Number)
#     e1.corners .= α * e2.corners
#     e1.edges .= α * e2.edges
#     return e1
# end

function VectorInterface.add(e1::CTMRGEnv, e2::CTMRGEnv, α=1, β=1)
    corners = α * e1.corners + β * e2.corners
    edges = α * e1.edges + β * e2.edges
    return CTMRGEnv(corners, edges)
end
function VectorInterface.add!(e1::CTMRGEnv, e2::CTMRGEnv, α=1, β=1)
    e1.corners .= α * e1.corners + β * e2.corners
    e1.edges .= α * e1.edges + β * e2.edges
    return e1
end
# function VectorInterface.add!!(e1, e2, α=1, β=1)
#     corners = α * e1.corners + β * e2.corners
#     edges = α * e1.edges + β * e2.edges
#     return CTMRGEnv(corners, edges)
# end

function VectorInterface.inner(e1::CTMRGEnv, e2::CTMRGEnv)
    return dot(e1.corners, e2.corners) + dot(e1.edges, e2.edges)
end

VectorInterface.norm(e::CTMRGEnv) = norm(e.corners) + norm(e.edges)
