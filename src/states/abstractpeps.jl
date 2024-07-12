"""
    const PEPSTensor{S}

Default type for PEPS tensors with a single physical index, and 4 virtual indices,
conventionally ordered as: ``T : P ← N ⊗ E ⊗ S ⊗ W``. Here, ``P``, ``N``, ``E``, ``S`` and
``W`` denote the physics, north, east, south and west spaces, respectively.
"""
const PEPSTensor{S} = AbstractTensorMap{S,1,4} where {S<:ElementarySpace}

"""
    PEPSTensor(f, ::Type{T}, Pspace::S, Nspace::S,
               [Espace::S], [Sspace::S], [Wspace::S]) where {T,S<:ElementarySpace}
    PEPSTensor(f, ::Type{T}, Pspace::Int, Nspace::Int,
               [Espace::Int], [Sspace::Int], [Wspace::Int]) where {T}
                
Construct a PEPS tensor based on the physical, north, east, west and south spaces.
Alternatively, only the space dimensions can be provided and ℂ is assumed as the field.
The tensor elements are generated based on `f` and the element type is specified in `T`.
"""
function PEPSTensor(
    f,
    ::Type{T},
    Pspace::S,
    Nspace::S,
    Espace::S=Nspace,
    Sspace::S=Nspace',
    Wspace::S=Espace',
) where {T,S<:ElementarySpace}
    return TensorMap(f, T, Pspace ← Nspace ⊗ Espace ⊗ Sspace ⊗ Wspace)
end
function PEPSTensor(
    f,
    ::Type{T},
    Pspace::Int,
    Nspace::Int,
    Espace::Int=Nspace,
    Sspace::Int=Nspace,
    Wspace::Int=Espace,
) where {T}
    return TensorMap(f, T, ℂ^Pspace ← ℂ^Nspace ⊗ ℂ^Espace ⊗ (ℂ^Sspace)' ⊗ (ℂ^Wspace)')
end

"""
    const PEPOTensor{S}

Default type for PEPO tensors with a single incoming and outgoing physical index, and 4
virtual indices, conventionally ordered as: O : P ⊗ P' ← N ⊗ E ⊗ S ⊗ W.
"""
const PEPOTensor{S} = AbstractTensorMap{S,2,4} where {S<:ElementarySpace}

"""
    abstract type AbstractPEPS end

Abstract supertype for a 2D projected entangled-pair state.
"""
abstract type AbstractPEPS end

"""
    abstract type AbstractPEPO end

Abstract supertype for a 2D projected entangled-pair operator.
"""
abstract type AbstractPEPO end

# Rotations
Base.rotl90(t::PEPSTensor) = permute(t, ((1,), (3, 4, 5, 2)))
Base.rotr90(t::PEPSTensor) = permute(t, ((1,), (5, 2, 3, 4)))
Base.rot180(t::PEPSTensor) = permute(t, ((1,), (4, 5, 2, 3)))

Base.rotl90(t::PEPOTensor) = permute(t, ((1, 2), (4, 5, 6, 3)))
Base.rotr90(t::PEPOTensor) = permute(t, ((1, 2), (6, 3, 4, 5)))
Base.rot180(t::PEPOTensor) = permute(t, ((1, 2), (5, 6, 3, 4)))
