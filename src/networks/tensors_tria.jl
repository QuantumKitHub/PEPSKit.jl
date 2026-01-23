#
# Partition function
#

"""
    const PartitionFunctionTriaTensor{S}

Default type for partition function tensors with 6 virtual indices, conventionally ordered
as: ``T : W ⊗ SW ⊗ SE ← NW ⊗ NE ⊗ E``. Here ``NW``, ``NE``, ``E``, ``SE``, ``SW`` and ``W`` denote the
north-west (120°), north-east (60°), east (0°), south-east (300°), south-west (240°) and west (180°) spaces, respectively,
where the angles denote the directions of the legs with respect to the positive horizontal axis.

```
      NW        NE
        ╲       ╱
         ╲     ╱
          ╲   ╱
    W ----- T ----- E
           ╱ ╲
          ╱   ╲
         ╱     ╲
        SW  P  SE
```
"""
const PartitionFunctionTriaTensor{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 3, 3}
const PFTriaTensor = PartitionFunctionTriaTensor

"""
    PartitionFunctionTriaTensor(f, ::Type{T}, NWspace::S,
               [NEspace::S], [Espace::S], [SEspace::S], 
               [SWspace::S], [Wspace::S]) where {T,S<:Union{Int,ElementarySpace}}
                
Construct a PartitionFunctionTriaTensor tensor based on the 
north-west, north-east, east, south-east, south-west and west spaces
The tensor elements are generated based on `f` and the element type is specified in `T`.
"""
function PartitionFunctionTriaTensor(
        f, ::Type{T},
        NWspace::S, NEspace::S = NWspace, Espace::S = NWspace, SEspace::S = NWspace,
        SWspace::S = NWspace, Wspace::S = NWspace,
    ) where {T, S <: ElementarySpace}
    return f(T, Wspace ⊗ SWspace ⊗ SEspace ← NWspace ⊗ NEspace ⊗ Espace)
end

function virtualspace(t::PFTriaTensor, dir)
    invp = (4, 5, 6, 3, 2, 1) # internally, virtual directions are ordered as NW, NE, E, SE, SW...
    return space(t, invp[dir])
end

#
# PEPS
#

"""
    const PEPSTriaTensor{S}

Default type for PEPS tensors with a single physical index, and 6 virtual indices,
conventionally ordered as: ``T : P ← NW ⊗ NE ⊗ E ⊗ SE ⊗ SW ⊗ W``. Here, ``P`` denotes the physical space
and ``NW``, ``NE``, ``E``, ``SE``, ``SW`` and ``W`` denote the north-west (120°), north-east (60°), 
east (0°), south-east (300°), south-west (240°) and west (180°) spaces, respectively,
where the angles denote the directions of the legs with respect to the positive horizontal axis.

```
```
      NW        NE
        ╲       ╱
         ╲     ╱
          ╲   ╱
    W ----- T ----- E
           ╱|╲
          ╱ | ╲
         ╱  |  ╲
        SW  P  SE
```
```
"""
const PEPSTriaTensor{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 1, 6}

"""
    PEPSTensor(f, ::Type{T}, Pspace::S, NWspace::S,
               [NEspace::S], [Espace::S], [SEspace::S], 
               [SWspace::S], [Wspace::S]) where {T,S<:Union{Int,ElementarySpace}}
                
Construct a PEPS tensor based on the physical, north-west, north-east, east, south-east, south-west and west spaces.
The tensor elements are generated based on `f` and the element type is specified in `T`.
"""
function PEPSTriaTensor(
        f, ::Type{T},
        Pspace::S,
        NWspace::S, NEspace::S = NWspace, Espace::S = NWspace, SEspace::S = NWspace,
        SWspace::S = NWspace, Wspace::S = NWspace,
    ) where {T, S <: ElementarySpace}
    return f(T, Pspace ← NWspace ⊗ NEspace ⊗ Espace ⊗ SEspace ⊗ SWspace ⊗ Wspace)
end

rotl60(t::PEPSTriaTensor) = permute(t, ((1,), (3, 4, 5, 6, 7, 2)))
rotr60(t::PEPSTriaTensor) = permute(t, ((1,), (7, 2, 3, 4, 5, 6)))
Base.rot180(t::PEPSTriaTensor) = permute(t, ((1,), (5, 6, 7, 2, 3, 4)))

physicalspace(t::PEPSTriaTensor) = space(t, 1)
virtualspace(t::PEPSTriaTensor, dir) = space(t, dir + 1)
