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
# with one physical leg factored out by `_qr_bond`
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
    Nr, Nc = size(state)
    return Dict(
        nb => _prepare_site_tensor(state[mod1(row + nb[1], Nr), mod1(col + nb[2], Nc)])
            for nb in neighbors
    )
end

"""
    benv_tensor(ket::PEPSTensor, bra::PEPSTensor, open_axs::Vector{Int})
    benv_tensor(ket::PEPSTensor, bra::PEPSTensor, open_axs::Vector{Int}, hairs::Vector{H}) where {T, H <: Union{Nothing, Hair}}

Contract the physical axes and the virtual axes of `ket` with `bra` to obtain the tensor on the boundary of the bond environment. 
Virtual axes specified by `open_axs` (in ascending order) are not contracted. 

# Examples

- West "hair" tensor (`open_axs = [EAST]`) 
    ```
                 ╱|
        ┌-----ket----- 2
        |    ╱ |  |
        |   |  |  |
        |   |  | ╱
        └---|-bra----- 1
            |╱
    ```
- Northwest corner tensor (`open_axs = [EAST, SOUTH]`, `hairs = [h, nothing]`)
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
- West edge tensor (`open_axs = [1, 2, 3]`)
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
        ket::PEPSTensor, bra::PEPSTensor, open_axs::Vector{Int}
    )
    # no hair tensors to be attached to virtual legs
    return benv_tensor(ket, bra, open_axs, fill(nothing, 4 - length(open_axs)))
end
function benv_tensor(
        ket::PEPSTensor, bra::PEPSTensor, open_axs::Vector{Int}, hairs::Vector{H}
    ) where {H <: Union{Nothing, Hair}}
    @assert length(hairs) == 4 - length(open_axs)
    ket2, nax = copy(ket), numind(ket)
    axs, open_axs2 = (2:5), open_axs .+ 1
    # contract with hair tensors
    hair_axs = Tuple(ax for ax in axs if ax ∉ open_axs2)
    for (h, ax) in zip(hairs, hair_axs)
        if h === nothing
            twistdual!(ket2, ax)
            continue
        end
        # ensure the hair doesn't change the virtual spaces
        @assert space(h, 1) == space(h, 2)'
        ket_indices = collect(-1:-1:-nax)
        ket_indices[ax] = 1
        ket2 = ncon([h, ket2], [[-ax, 1], ket_indices])
    end
    # combine bra and ket
    indexlist = [-collect(1:2:(2 * nax)), -collect(2:2:(2 * nax))]
    for ax in 1:nax
        if ax ∉ open_axs2
            indexlist[1][ax] = indexlist[2][ax] = ax
        end
    end
    return ncon([bra, ket2], indexlist, [true, false])
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
const open_axs_hair = Dict(:n => [SOUTH], :e => [WEST], :s => [NORTH], :w => [EAST])
const open_axs_cor = Dict(
    :nw => [EAST, SOUTH], :ne => [SOUTH, WEST], :se => [NORTH, WEST], :sw => [NORTH, EAST]
)
const open_axs_edge = Dict(
    :n => [EAST, SOUTH, WEST],
    :e => [NORTH, SOUTH, WEST],
    :s => [NORTH, EAST, WEST],
    :w => [NORTH, EAST, SOUTH],
)

# construction of hairs
for (dir, open_axs) in open_axs_hair
    fname = Symbol("hair_", dir)
    @eval begin
        $(fname)(ket) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket, h1, h2, h3) = benv_tensor(ket, ket, $open_axs, [h1, h2, h3])
    end
end

# construction of corners
for (dir, open_axs) in open_axs_cor
    fname = Symbol("cor_", dir)
    @eval begin
        $(fname)(ket) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket, h1, h2) = benv_tensor(ket, ket, $open_axs, [h1, h2])
    end
end

# construction of edges
for (dir, open_axs) in open_axs_edge
    fname = Symbol("edge_", dir)
    @eval begin
        $(fname)(ket) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket, h) = benv_tensor(ket, ket, $open_axs, [h])
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
