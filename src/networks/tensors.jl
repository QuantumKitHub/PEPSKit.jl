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
const PartitionFunctionTensor{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 2, 2}
const PFTensor = PartitionFunctionTensor

"""
    PartitionFunctionTensor(f, ::Type{TorA}, Pspace::S, Nspace::S,
               [Espace::S], [Sspace::S], [Wspace::S]) where {TorA, S<:Union{Int,ElementarySpace}}
                
Construct a `PartitionFunctionTensor` tensor based on the north, east, west and south spaces.
The tensor elements are generated based on `f` and the element type is specified in `T`.
"""
function PartitionFunctionTensor(
        f, ::Type{TorA},
        Nspace::S, Espace::S = Nspace, Sspace::S = Nspace, Wspace::S = Espace,
    ) where {TorA, S <: ElementarySpace}
    return f(TorA, Wspace ⊗ Sspace ← Nspace ⊗ Espace)
end

Base.rotl90(t::PFTensor) = permute(t, ((3, 1), (4, 2)))
Base.rotr90(t::PFTensor) = permute(t, ((2, 4), (1, 3)))
Base.rot180(t::PFTensor) = permute(t, ((4, 3), (2, 1)))

function virtualspace(t::PFTensor, dir)
    invp = (3, 4, 2, 1) # internally, virtual directions are ordered as N, E, S, W...
    return space(t, invp[dir])
end

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
const PEPSTensor{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 1, 4}

"""
    PEPSTensor(f, ::Type{TorA}, Pspace::S, Nspace::S,
               [Espace::S], [Sspace::S], [Wspace::S]) where {TorA, S<:Union{Int,ElementarySpace}}
                
Construct a PEPS tensor based on the physical, north, east, south and west spaces.
The tensor elements are generated based on `f` and the element type is specified in `T`.
"""
function PEPSTensor(
        f,
        ::Type{TorA},
        Pspace::S,
        Nspace::S, Espace::S = Nspace, Sspace::S = Nspace', Wspace::S = Espace',
    ) where {TorA, S <: ElementarySpace}
    return f(TorA, Pspace ← Nspace ⊗ Espace ⊗ Sspace ⊗ Wspace)
end

Base.rotl90(t::PEPSTensor) = permute(t, ((1,), (3, 4, 5, 2)))
Base.rotr90(t::PEPSTensor) = permute(t, ((1,), (5, 2, 3, 4)))
Base.rot180(t::PEPSTensor) = permute(t, ((1,), (4, 5, 2, 3)))

physicalspace(t::PEPSTensor) = space(t, 1)
virtualspace(t::PEPSTensor, dir) = space(t, dir + 1)

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
const PEPOTensor{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 2, 4}

Base.rotl90(t::PEPOTensor) = permute(t, ((1, 2), (4, 5, 6, 3)))
Base.rotr90(t::PEPOTensor) = permute(t, ((1, 2), (6, 3, 4, 5)))
Base.rot180(t::PEPOTensor) = permute(t, ((1, 2), (5, 6, 3, 4)))

domain_physicalspace(t::PEPOTensor) = space(t, 2)'
codomain_physicalspace(t::PEPOTensor) = space(t, 1)
function physicalspace(t::PEPOTensor)
    codomain_physicalspace(t) == domain_physicalspace(t) ||
        throw(SpaceMismatch("Domain and codomain physical spaces do not match."))
    return codomain_physicalspace(t)
end
virtualspace(t::PEPOTensor, dir) = space(t, dir + 2)

"""
    $(SIGNATURES)

Fuse the physical indices of a PEPO tensor, obtaining a PEPS tensor.
"""
function fuse_physicalspaces(O::PEPOTensor)
    F = isomorphism(Int, fuse(codomain(O)), codomain(O))
    return F * O, F
end

"""
    $(SIGNATURES)

Trace out the physical indices of a PEPO tensor, obtaining a partition function tensor.
"""
function trace_physicalspaces(O::PEPOTensor)
    @plansor t[W S; N E] := O[p p; N E S W]
    return t
end
