"""
    absorb_weight(t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight, rowcol::CartesianIndex{2}, virt_axes::NTuple{N, Int}; inv::Bool = false)
    absorb_weight(t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight, row::Int, col::Int, virt_axes::NTuple{N, Int}; inv::Bool = false)

Absorb or remove (in a twist-free way) the square root of environment weight
on an axis of the PEPS/PEPO tensor `t` known to be at position (`row`, `col`)
in the unit cell of an InfinitePEPS/InfinitePEPO. The involved weights are
```
                    |
                 [2,r,c]
                    |
    - [1,r,c-1] - T[r,c] - [1,r,c] -
                    |
                [2,r+1,c]
                    |
```

## Arguments

- `t::Union{PEPSTensor, PEPOTensor}` : PEPSTensor or PEPOTensor to which the weight will be absorbed.
- `weights::SUWeight` : All simple update weights.
- `row::Int` : The row index specifying the position in the tensor network.
- `col::Int` : The column index specifying the position in the tensor network.
- `virt_axes::Int` : The axis into which the weight is absorbed, taking values from 1 to 4, standing for north, east, south, west respectively.

## Keyword arguments

- `inv::Bool=false` : If `true`, the inverse square root of the weight is absorbed.

## Examples

```julia
# Absorb the weight into the north axis of tensor at position (2, 3)
absorb_weight(t, weights, 2, 3, (1,))

# Absorb the inverse of (i.e. remove) the weight into the east axis
absorb_weight(t, weights, 2, 3, (2,); inv=true)
```
"""
function absorb_weight(
        t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight,
        rowcol::CartesianIndex{2}, virt_axes::NTuple{N, Int}; inv::Bool = false
    ) where {N}
    return absorb_weight(t, weights, rowcol[1], rowcol[2], virt_axes; inv)
end

function absorb_weight(
        t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight,
        row::Int, col::Int, virt_axes::NTuple{N, Int}; inv::Bool = false
    ) where {N}
    wts = map(ax -> weight_to_absorb(weights, row, col, ax; inv), virt_axes)
    T = promote_type(scalartype(t), map(scalartype, wts)...)
    return _absorb_weights!(similar(t, T), t, wts, virt_axes)
end

"""
    absorb_weight!(t, weights::SUWeight, rowcol::CartesianIndex{2}, virt_axes; inv = false)

In-place version of [`absorb_weight`](@ref).
"""
function absorb_weight!(
        t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight,
        rowcol::CartesianIndex{2}, virt_axes::NTuple{N, Int}; inv::Bool = false
    ) where {N}
    wts = map(ax -> weight_to_absorb(weights, rowcol[1], rowcol[2], ax; inv), virt_axes)
    return _absorb_weights!(t, t, wts, virt_axes)
end

# Write `src` into `dst` with `weights[i]` absorbed into virtual axis `virtual_axes[i]`,
# where `domain(weights[i]) == domain(src)[virtual_axes[i]]`; `dst === src` is allowed.
# A diagonal weight only rescales the basis vectors of its leg, by a factor that depends on
# the sector of the leg and the index within it. This commutes with any permutation, so it is
# applied directly on each fusion-tree block of `src` in its own leg order; it is equivalent
# to `permute(src, ...) * wt` followed by the inverse permutation.
# Since the domain of `wt` is the space of the leg itself (dual legs get `transpose(wt)` from
# `weight_to_absorb`), `wt` is composed onto the leg without bending it, so even for dual legs
# no twist is involved.
function _absorb_weights!(
        dst::AbstractTensorMap, src::AbstractTensorMap,
        weights::NTuple{N, DiagonalTensorMap}, virtual_axes::NTuple{N, Int}
    ) where {N}
    Np, M = numout(src), Val(numind(src)) # static rank, so the reshaped weights are inferred
    space(dst) == space(src) ||
        throw(SpaceMismatch(lazy"destination $(space(dst)) does not match source $(space(src))"))
    for (wt, ax) in zip(weights, virtual_axes)
        domain(wt)[1] == domain(src)[ax] ||
            throw(SpaceMismatch(lazy"weight $(space(wt)) does not match virtual axis $ax of $(space(src))"))
    end
    # Look up every block of the weights only once, reshaped to broadcast along their leg, and
    # iterate over the subblocks of `dst` and `src` (which have the same order) with `subblocks`,
    # as the per-block `block(wt, c)` and `t[f₁, f₂]` would both redo the structure lookups.
    λdicts = map(weights, virtual_axes) do wt, ax
        return Dict(
            c => sreshape(StridedView(b.diag), ntuple(i -> i == Np + ax ? length(b.diag) : 1, M))
                for (c, b) in blocks(wt)
        )
    end
    for (((f₁, f₂), d), (_, s)) in zip(subblocks(dst), subblocks(src))
        λs = map((λdict, ax) -> λdict[f₂.uncoupled[ax]], λdicts, virtual_axes)
        @. d = *(s, λs...)
    end
    return dst
end

"""
Pick out the weight to be absorbed to the `ax`th domain
of the tensor at position `[row, col]`, and take its
square root (or inverse square root if `inv = true`).
"""
function weight_to_absorb(
        weights::SUWeight, row::Int, col::Int, ax::Int; inv::Bool = false
    )
    pow = inv ? -1 / 2 : 1 / 2
    wt = sdiag_pow(
        if ax == NORTH
            weights[2, row, col]
        elseif ax == EAST
            weights[1, row, col]
        elseif ax == SOUTH
            weights[2, row + 1, col]
        else # WEST
            weights[1, row, col - 1]
        end,
        pow,
    )
    (ax == SOUTH || ax == WEST) && return transpose(wt)
    return wt
end
