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
    return _absorb_weight(t, weights, row, col, virt_axes; inv)
end

"""
Absorb weights on the virtual directions in `virt_axes`, or remove them when `inv = true`.

- `virt_axes` selects virtual directions using `1 = NORTH`, `2 = EAST`, etc., regardless of the current leg order.
- `input_axes` specifies how the legs of the input `t` are already permuted with respect to the standard order.
  For example, `input_axes = (5, 1, 2, 4, 3)` for a PEPS tensor means the legs are currently ordered as west, physical, north, south, east.
- `output_perm` specifies how the returned tensor should permute the legs with respect to the standard order.
  For example, `output_perm = ((5, 1, 2, 4), (3,))` puts west, physical, north, south in the codomain and east in the domain.
- There are always four virtual legs on the square lattice, so the physical leg count is `numind(t) - 4`.
"""
function _absorb_weight(
        t::AbstractTensorMap, weights::SUWeight, row::Int, col::Int,
        virt_axes::NTuple{N, Int}; inv::Bool = false,
        input_axes::Tuple = ntuple(identity, numind(t)),
        output_perm::Tuple{Tuple, Tuple} = (
            ntuple(identity, numind(t) - 4), ntuple(i -> numind(t) - 4 + i, 4),
        )
    ) where {N}
    Np = numind(t) - 4
    vax = first(virt_axes)
    axes, biperm = _permute_to_last(input_axes, vax + Np)
    wt = weight_to_absorb(weights, row, col, vax; inv)
    # Allocate the first result to preserve the input and allow scalar promotion.
    t2 = permute(t, biperm) * wt
    for vax in Base.tail(virt_axes)
        axes, biperm = _permute_to_last(axes, vax + Np)
        wt = weight_to_absorb(weights, row, col, vax; inv)
        # Diagonal composition is twist-free and only mutates the intermediate.
        t2 = rmul!(permute(t2, biperm), wt)
    end
    perm_back = invperm(axes)
    return permute(t2, map(p -> map(i -> perm_back[i], p), output_perm))
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
