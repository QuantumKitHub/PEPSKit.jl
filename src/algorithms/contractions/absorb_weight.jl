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
    vax = first(virt_axes)
    weight_vax = weight_to_absorb(weights, row, col, vax; inv)
    legs, t2 = absorb_first_weight(t, weight_vax, vax)
    for vax in Base.tail(virt_axes)
        legs, biperm = biperm_absorb_weight(legs, vax)
        weight_vax = weight_to_absorb(weights, row, col, vax; inv)
        t2 = permute(t2, biperm) * weight_vax
    end
    perm_back = invperm(legs)
    return permute(t2, (perm_back[begin:numout(t)], perm_back[(numout(t) + 1):end]))
end

function weight_to_absorb(
        weights::SUWeight, row::Int, col::Int, ax::Int; inv::Bool = false
    )
    _, Nr, Nc = size(weights)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    pow = inv ? -1 / 2 : 1 / 2
    wt = sdiag_pow(
        if ax == NORTH
            weights[2, row, col]
        elseif ax == EAST
            weights[1, row, col]
        elseif ax == SOUTH
            weights[2, _next(row, Nr), col]
        else # WEST
            weights[1, row, _prev(col, Nc)]
        end,
        pow,
    )
    # make absorption/removal twist-free
    twistdual!(wt, 1)
    (ax == SOUTH || ax == WEST) && return transpose(wt)  # not sure this can be factorized due to twistdual
    return wt
end

function biperm_absorb_weight(legs::NTuple{N, Int}, vax::Int) where {N}
    @assert N == 5 || N == 6
    nin = N - 4
    a = vax + nin
    codomain_axes = TupleTools.deleteat(ntuple(identity, N), a)
    q = invperm(legs)
    biperm = (map(i -> q[i], codomain_axes), (q[a],))
    new_legs = (ntuple(i -> legs[biperm[1][i]], N - 1)..., a)
    return new_legs, biperm
end

function absorb_first_weight(t::Union{PEPSTensor, PEPOTensor}, wt, vax)
    legs = ntuple(identity, numind(t))
    new_legs, biperm = biperm_absorb_weight(legs, vax)
    t2 = permute(t, biperm) * wt
    return new_legs, t2
end

