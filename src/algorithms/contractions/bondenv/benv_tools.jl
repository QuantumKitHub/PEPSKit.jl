#=
The construction of bond environment for Neighborhood Tensor Update (NTU)
is adapted from YASTN (https://github.com/yastn/yastn).
Copyright 2024 The YASTN Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0
=#

const BondEnv{T, S} = AbstractTensorMap{T, S, 2, 2} where {T <: Number, S <: ElementarySpace}
const BondEnv3site{T, S} = AbstractTensorMap{T, S, 4, 4} where {T <: Number, S <: ElementarySpace}
const Hair{T, S} = AbstractTensor{T, S, 2} where {T <: Number, S <: ElementarySpace}
# Orthogonal tensors obtained PEPSTensor/PEPOTensor
# from `bond_tensor_...` functions
const PEPSOrth{T, S} = AbstractTensor{T, S, 4} where {T <: Number, S <: ElementarySpace}
const PEPOOrth{T, S} = AbstractTensor{T, S, 5} where {T <: Number, S <: ElementarySpace}

"Convert tensor `t` connected by the bond to be truncated to a `PEPSTensor`."
_prepare_site_tensor(t::PEPSTensor) = t
_prepare_site_tensor(t::PEPOTensor) = first(fuse_physicalspaces(t))
_prepare_site_tensor(t::PEPSOrth) = permute(insertleftunit(t, 1), ((1,), (2, 3, 4, 5)))
_prepare_site_tensor(t::PEPOOrth) = permute(t, ((1,), (2, 3, 4, 5)))

"""
Extract tensors in an InfinitePEPS or 1-layer InfinitePEPO
at positions `neighbors` relative to `(row, col)`
"""
function collect_neighbors(
        state::InfiniteState, row::Int, col::Int, neighbors::Vector{Tuple{Int, Int}}
    )
    return Dict(
        nb => _prepare_site_tensor(state[row + nb[1], col + nb[2]])
            for nb in neighbors
    )
end

"""
    benv_tensor(ket::PEPSTensor, bra::PEPSTensor, open_axs::NTuple{N, Int}) where {N}
    benv_tensor(ket::PEPSTensor, bra::PEPSTensor, open_axs::NTuple{N, Int}, hairs::NTuple{Nh, H}) where {N, Nh, H <: Union{Nothing, Hair}}

Contract the physical axes and the virtual axes of `ket` with `bra` to obtain the tensor on the boundary of the bond environment.
Virtual axes specified by `open_axs` (in ascending order) are not contracted.
Hair tensors can be inserted on contracted legs between `ket` and `bra`.

# Examples

- West "hair" tensor (`open_axs = (EAST,)`)
    ```
                 ╱|
        ┌-----ket----- 2
        |    ╱ |  |
        |   |  |  |
        |   |  | ╱
        └---|-bra----- 1
            |╱
    ```
- Northwest corner tensor (`open_axs = (EAST, SOUTH)`, `hairs = (h, nothing)`)
    ```
                 ╱|
        ┌-----ket----- 2
        |    ╱ |  h
        |   4  |  |
        |      |  |
        |      | ╱
        └-----bra----- 1
             ╱
            3
    ```
- West edge tensor (`open_axs = (NORTH, EAST, SOUTH)`)
    ```
                   2
                 ╱
        ┌-----ket----- 4
        |    ╱ |
        |   6  |   1
        |      | ╱
        └-----bra----- 3
             ╱
            5
    ```
"""
function benv_tensor(
        ket::PEPSTensor, bra::PEPSTensor, open_axs::NTuple{N, Int}
    ) where {N}
    # no hair tensors to be attached to virtual legs
    return benv_tensor(ket, bra, open_axs, ntuple(Returns(nothing), 4 - N))
end
function benv_tensor(
        ket::PEPSTensor, bra::PEPSTensor,
        open_vaxs::NTuple{N, Int}, hairs::NTuple{Nh, Union{Nothing, H}}
    ) where {N, Nh, H <: Hair}
    @assert 1 <= N <= 3 && Nh == 4 - N
    open_axs = open_vaxs .+ 1
    # axes to be contracted
    hair_axs = Tuple(ax for ax in 2:5 if ax ∉ open_axs)
    # attach hairs to ket
    ket = twistdual(ket, 1)
    axes = ntuple(identity, Val(5))
    for (h, ax) in zip(hairs, hair_axs)
        twistdual!(ket, ax)
        h === nothing && continue
        axes, biperm = _permute_to_first(axes, ax)
        ket = permute(h, ((1,), (2,))) * permute(ket, biperm)
    end
    perm_back = invperm(axes)
    # combine bra and ket
    cont_axs = (1, hair_axs...)
    pbra = (open_axs, cont_axs)
    pket = (
        map(p -> perm_back[p], cont_axs),
        map(p -> perm_back[p], open_axs),
    )
    pbraket = (ntuple(j -> isodd(j) ? (j + 1) ÷ 2 : (j ÷ 2) + N, 2N), ())
    return tensorcontract(bra, pbra, true, ket, pket, false, pbraket)
end

#= Free axes of different boundary tensors
(C/E/H mean corner/edge/hair)

                       H_n
                        |

            C_nw -   - E_n -   - C_ne
            |           |           |

            |           |           |
    H_w -   E_w -    - ket -    - E_e   - H_e
            |           |           |

            |           |           |
            C_sw -   - E_s -   - C_se

                        |
                       H_s
=#
# construction of hairs
for (dir, open_axs) in [:n => (SOUTH,), :e => (WEST,), :s => (NORTH,), :w => (EAST,)]
    fname = Symbol("hair_", dir)
    @eval begin
        $(fname)(ket) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket, h1, h2, h3) = benv_tensor(ket, ket, $open_axs, (h1, h2, h3))
    end
end

# construction of corners
for (dir, open_axs) in [
        :nw => (EAST, SOUTH), :ne => (SOUTH, WEST),
        :se => (NORTH, WEST), :sw => (NORTH, EAST),
    ]
    fname = Symbol("cor_", dir)
    @eval begin
        $(fname)(ket) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket, h1, h2) = benv_tensor(ket, ket, $open_axs, (h1, h2))
    end
end

# construction of edges
for (dir, open_axs) in [
        :n => (EAST, SOUTH, WEST), :e => (NORTH, SOUTH, WEST),
        :s => (NORTH, EAST, WEST), :w => (NORTH, EAST, SOUTH),
    ]
    fname = Symbol("edge_", dir)
    @eval begin
        $(fname)(ket) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket, h) = benv_tensor(ket, ket, $open_axs, (h,))
    end
end

"""
Enlarge the northwest corner
```
    ctl══ D1 ══ et ══ -5/-6
    ║           ║
    D2          D3
    ║           ║
    el ══ D4 ══ X ═══ -7/-8
    ║           ║
    -1/-2       -3/-4
```
"""
function enlarge_corner_nw(
        ctl::AbstractTensor{E, S, 4},
        et::AbstractTensor{E, S, 6}, el::AbstractTensor{E, S, 6},
        ket::PEPSTensor, bra::PEPSTensor = ket
    ) where {E, S}
    return @tensoropt ctl2[:] := ctl[D11 D10 D21 D20] *
        et[-5 -6 D31 D30 D11 D10] * el[D21 D20 D41 D40 -1 -2] *
        conj(bra[d D31 -7 -3 D41]) * twistdual(ket, 1)[d D30 -8 -4 D40]
end

"""
Enlarge the southeast corner
```
              -1/-2       -3/-4
                ║           ║
    -5/-6 ═════ Y ══ D1 ═══ er
                ║           ║
                D2          D3
                ║           ║
    -7/-8 ═════ eb ═ D4 ══ cbr
```
"""
function enlarge_corner_se(
        cbr::AbstractTensor{E, S, 4},
        eb::AbstractTensor{E, S, 6}, er::AbstractTensor{E, S, 6},
        ket::PEPSTensor, bra::PEPSTensor = ket
    ) where {E, S}
    return @tensoropt cbr2[:] := cbr[D31 D30 D41 D40] *
        eb[D21 D20 D41 D40 -7 -8] * er[-3 -4 D31 D30 D11 D10] *
        conj(bra[d -1 D11 D21 -5]) * twistdual(ket, 1)[d -2 D10 D20 -6]
end
