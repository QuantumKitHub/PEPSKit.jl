#
# Partition function
#

"""
    const PartitionFunctionTensor{S}

Default type for partition function tensors with 4 virtual indices, conventionally ordered
as: ``T : W ⊗ S ← N ⊗ E``. Here, ``N``, ``E``, ``S`` and ``W`` denote the north, east, south
and west spaces, respectively.

```
          N
         ╱
        ╱
  W---- ----E
      ╱
     ╱
    S 
```
"""
const PartitionFunctionTensor{S} = AbstractTensorMap{S,2,2} where {S<:ElementarySpace}
const PFTensor = PartitionFunctionTensor

Base.rotl90(t::PFTensor) = permute(t, ((3, 1), (4, 2)))
Base.rotr90(t::PFTensor) = permute(t, ((2, 4), (1, 3)))
Base.rot180(t::PFTensor) = permute(t, ((4, 3), (2, 1)))

#
# PEPS
#

"""
    const PEPSTensor{S}

Default type for PEPS tensors with a single physical index, and 4 virtual indices,
conventionally ordered as: ``T : P ← N ⊗ E ⊗ S ⊗ W``. Here, ``P`` denotes the physical space
and ``N``, ``E``, ``S`` and ``W`` denote the north, east, south and west virtual spaces,
respectively.

```
           N
          ╱
         ╱
   W---- ----E
       ╱|
      ╱ |
     S  P
```
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

Base.rotl90(t::PEPSTensor) = permute(t, ((1,), (3, 4, 5, 2)))
Base.rotr90(t::PEPSTensor) = permute(t, ((1,), (5, 2, 3, 4)))
Base.rot180(t::PEPSTensor) = permute(t, ((1,), (4, 5, 2, 3)))

#
# PEPO
#

"""
    const PEPOTensor{S}

Default type for PEPO tensors with a single incoming and outgoing physical index, and 4
virtual indices, conventionally ordered as: ``T : P ⊗ P´ ← N ⊗ E ⊗ S ⊗ W``. Here, ``P´`` and
``P`` denote the incoming and outgoing physical space respectively, encoding the physical
mapping from ``P´'`` to ``P`` where ``P´'`` corresponds to a physical PEPS index. ``N``,
``E``, ``S`` and ``W`` denote the physics, north, east, south and west spaces, respectively.


```
        P´ N
        | ╱
        |╱
   W---- ----E
       ╱|
      ╱ |
     S  P
```
"""
const PEPOTensor{S} = AbstractTensorMap{S,2,4} where {S<:ElementarySpace}

Base.rotl90(t::PEPOTensor) = permute(t, ((1, 2), (4, 5, 6, 3)))
Base.rotr90(t::PEPOTensor) = permute(t, ((1, 2), (6, 3, 4, 5)))
Base.rot180(t::PEPOTensor) = permute(t, ((1, 2), (5, 6, 3, 4)))
