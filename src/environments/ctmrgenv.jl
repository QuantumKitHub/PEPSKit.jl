"""
    struct CTMRGEnv{C,T}

Corner transfer-matrix environment containing unit-cell arrays of corner and edge tensors.
"""
struct CTMRGEnv{C,T}
    corners::Array{C,3}
    edges::Array{T,3}
end

"""
    CTMRGEnv(peps::InfinitePEPS{P}; Venv=oneunit(spacetype(P)))

Create a random CTMRG environment from a PEPS tensor. The environment bond dimension
defaults to one and can be specified using the `Venv` space.
"""
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

# Custom adjoint for CTMRGEnv getproperty, to avoid creating named tuples in backward pass
function ChainRulesCore.rrule(::typeof(getproperty), e::CTMRGEnv, name::Symbol)
    result = getproperty(e, name)
    if name === :corners
        function corner_pullback(Δcorners)
            return NoTangent(), CTMRGEnv(Δcorners, zerovector.(e.edges)), NoTangent()
        end
        return result, corner_pullback
    elseif name === :edges
        function edge_pullback(Δedges)
            return NoTangent(), CTMRGEnv(zerovector.(e.corners), Δedges), NoTangent()
        end
        return result, edge_pullback
    else
        # this should never happen because already errored in forwards pass
        throw(ArgumentError("No rrule for getproperty of $name"))
    end
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

# In-place update of environment
function update!(env::CTMRGEnv{C,T}, env´::CTMRGEnv{C,T}) where {C,T}
    env.corners .= env´.corners
    env.edges .= env´.edges
    return env
end

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

# VectorInterface
# ---------------

# Note: the following methods consider the environment tensors as separate components of one
# big vector. In other words, the associated vector space is not the natural one associated
# to the original (physical) system, and addition, scaling, etc. are performed element-wise.

import VectorInterface as VI

function VI.scalartype(::Type{CTMRGEnv{C,T}}) where {C,T}
    S₁ = scalartype(C)
    S₂ = scalartype(T)
    return promote_type(S₁, S₂)
end

function VI.zerovector(env::CTMRGEnv, ::Type{S}) where {S<:Number}
    _zerovector = Base.Fix2(zerovector, S)
    return CTMRGEnv(map(_zerovector, env.corners), map(_zerovector, env.edges))
end
function VI.zerovector!(env::CTMRGEnv)
    foreach(zerovector!, env.corners)
    foreach(zerovector!, env.edges)
    return env
end
VI.zerovector!!(env::CTMRGEnv) = zerovector!(env)

function VI.scale(env::CTMRGEnv, α::Number)
    _scale = Base.Fix2(scale, α)
    return CTMRGEnv(map(_scale, env.corners), map(_scale, env.edges))
end
function VI.scale!(env::CTMRGEnv, α::Number)
    _scale! = Base.Fix2(scale!, α)
    foreach(_scale!, env.corners)
    foreach(_scale!, env.edges)
    return env
end
function VI.scale!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number)
    _scale!(x, y) = scale!(x, y, α)
    foreach(_scale!, env₁.corners, env₂.corners)
    foreach(_scale!, env₁.edges, env₂.edges)
    return env₁
end
VI.scale!!(env::CTMRGEnv, α::Number) = scale!(env, α)
VI.scale!!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number) = scale!(env₁, env₂, α)

function VI.add(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number, β::Number)
    _add(x, y) = add(x, y, α, β)
    return CTMRGEnv(
        map(_add, env₁.corners, env₂.corners), map(_add, env₁.corners, env₂.corners)
    )
end
function VI.add!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    foreach(_add!, env₁.corners, env₂.corners)
    foreach(_add!, env₁.edges, env₂.edges)
    return env₁
end
VI.add!!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number, β::Number) = add!(env₁, env₂, α, β)

# Exploiting the fact that VectorInterface works for tuples:
function VI.inner(env₁::CTMRGEnv, env₂::CTMRGEnv)
    return inner((env₁.corners, env₁.edges), (env₂.corners, env₂.edges))
end
VI.norm(env::CTMRGEnv) = norm((env.corners, env.edges))
