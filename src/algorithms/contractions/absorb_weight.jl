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
    Np = numout(t)
    vax = first(virt_axes)
    wt = weight_to_absorb(weights, row, col, vax; inv)
    axes, t2 = absorb_first_weight(t, wt, vax)
    for vax in Base.tail(virt_axes)
        axes, biperm = _permute_to_last(axes, vax + Np)
        wt = weight_to_absorb(weights, row, col, vax; inv)
        # use `*` to make absorption/removal twist-free
        t2 = permute(t2, biperm) * wt
    end
    perm_back = invperm(axes)
    return permute(t2, (perm_back[begin:numout(t)], perm_back[(numout(t) + 1):end]))
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

function absorb_first_weight(
        t::Union{PEPSTensor, PEPOTensor}, wt::DiagonalTensorMap, vax::Int
    )
    Np = numout(t)
    axes, biperm = _permute_to_last(ntuple(identity, numind(t)), vax + Np)
    # use `*` to make absorption/removal twist-free
    t2 = permute(t, biperm) * wt
    return axes, t2
end
