"""
    const PEPSTensor{S}

Default type for PEPS tensors with a single physical index, and 4 virtual indices,
conventionally ordered as: T : P ← N ⊗ E ⊗ S ⊗ W.
"""
const PEPSTensor{S} = AbstractTensorMap{S,1,4} where {S<:ElementarySpace}

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

Abstract supertype for a 2D projected entangled pairs state.
"""
abstract type AbstractPEPS end

"""
    abstract type AbstractPEPO end

Abstract supertype for a 2D projected entangled pairs operator.
"""
abstract type AbstractPEPO end

Base.rotl90(t::PEPSTensor) = permute(t, ((1,), (3, 4, 5, 2)))
Base.rotl90(t::PEPOTensor) = permute(t, ((1, 2), (4, 5, 6, 3)))