#= 
The construction of bond environment for Neighborhood Tensor Update (NTU) 
is adapted from YASTN (https://github.com/yastn/yastn).
Copyright 2024 The YASTN Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0
=#

const BondEnv{T, S} = AbstractTensorMap{T, S, 2, 2} where {T <: Number, S <: ElementarySpace}
const Hair{T, S} = AbstractTensor{T, S, 2} where {T <: Number, S <: ElementarySpace}
# Orthogonal tensors obtained PEPSTensor/PEPOTensor
# with one physical leg being factored out by `_qr_bond`
const PEPSOrth{T, S} = AbstractTensor{T, S, 4} where {T <: Number, S <: ElementarySpace}
const PEPOOrth{T, S} = AbstractTensor{T, S, 5} where {T <: Number, S <: ElementarySpace}
const StateTensor = Union{PEPSTensor, PEPOTensor, PEPSOrth, PEPOOrth}

"""
Extract tensors in an InfinitePEPS or 1-layer InfinitePEPO
at positions `neighbors` relative to `(row, col)`
"""
function collect_neighbors(
        state::InfiniteState, row::Int, col::Int, neighbors::Vector{Tuple{Int, Int}}
    )
    Nr, Nc = size(state)
    return Dict(
        nb => state.A[mod1(row + nb[1], Nr), mod1(col + nb[2], Nc)]
            for nb in neighbors
    )
end

"""
    benv_tensor(ket::T, bra::T, open_axs::Vector{Int}) where
        {T <: Union{PEPSTensor, PEPOTensor, PEPSOrth, PEPOOrth}}
    benv_tensor(ket::T, bra::T, open_axs::Vector{Int}, hairs::Vector{H}) where
        {T <: Union{PEPSTensor, PEPOTensor, PEPSOrth, PEPOOrth}, H <: Union{Nothing, Hair}}

Contract the physical axes and the virtual axes of `ket` with `bra` to obtain the tensor on the boundary of the bond environment. 
Virtual axes specified by `open_axs` (in ascending order) are not contracted. 

# Examples

- West "hair" tensor when `ket`, `bra` are `PEPSTensor` or `PEPOTensor`
    (`open_axs = [2]`) 
    ```
                                   :
                 ╱|                | ╱|
        ┌-----ket----- 2    ┌-----ket----- 2
        |    ╱ |  |         |    ╱ |  |
        |   |  |  |         |   |  |  |
        |   |  | ╱          |   |  | ╱
        └---|-bra----- 1    └---|-bra----- 1
            |╱                  |╱ |
                                   :
    ```
    For `PEPOTensor`, the remaining physical indices are traced out.

- Northwest corner tensor when `ket`, `bra` are `PEPSOrth` or `PEPOOrth`
    (`open_axs = [2, 3]`, `hairs = [h, nothing]`)
    ```
                                   :
                 ╱|                | ╱|
        ┌-----ket----- 2    ┌-----ket----- 2
        |    ╱    h         |    ╱    h
        |   4     |         |   4     |
        |         |         |         |
        |        ╱          |        ╱
        └-----bra----- 1    └-----bra----- 1
             ╱                   ╱ |
            3                   3  :
    ```

- West edge tensor when `ket`, `bra` are `PEPSTensor` or `PEPOTensor`
    (`open_axs = [1, 2, 3]`)
    ```
                   2               :   2
                 ╱                 | ╱
        ┌-----ket----- 4    ┌-----ket----- 4
        |    ╱ |            |    ╱ |
        |   6  |   1        |   6  |   1
        |      | ╱          |      | ╱
        └-----bra----- 3    └-----bra----- 3
             ╱                   ╱ |
            5                   5  :
    ```
"""
function benv_tensor(
        ket::T, bra::T, open_axs::Vector{Int}
    ) where {T <: StateTensor}
    # no hair tensors to be attached to virtual legs
    return benv_tensor(ket, bra, open_axs, fill(nothing, 4 - length(open_axs)))
end
function benv_tensor(
        ket::T, bra::T, open_axs::Vector{Int}, hairs::Vector{H}
    ) where {T <: StateTensor, H <: Union{Nothing, Hair}}
    @assert length(hairs) == 4 - length(open_axs)
    ket2 = if T <: PEPOTensor
        first(fuse_physicalspaces(ket))
    elseif T <: PEPSOrth
        deepcopy(ket)
    else # PEPSTensor, PEPOOrth
        twistdual(ket, 1)
    end
    nax = numind(ket2)
    axs, open_axs2 = if T <: PEPSOrth
        (1:4), open_axs
    else # PEPSTensor, PEPOTensor (with physical legs fused), PEPOOrth
        (2:5), open_axs .+ 1
    end
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
    bra2 = (T <: PEPOTensor) ? first(fuse_physicalspaces(bra)) : bra
    return ncon([bra2, ket2], indexlist, [true, false])
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
        $(fname)(ket::StateTensor) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket::StateTensor, h1, h2, h3) = benv_tensor(ket, ket, $open_axs, [h1, h2, h3])
    end
end

# construction of corners
for (dir, open_axs) in open_axs_cor
    fname = Symbol("cor_", dir)
    @eval begin
        $(fname)(ket::StateTensor) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket::StateTensor, h1, h2) = benv_tensor(ket, ket, $open_axs, [h1, h2])
    end
end

# construction of edges
for (dir, open_axs) in open_axs_edge
    fname = Symbol("edge_", dir)
    @eval begin
        $(fname)(ket::StateTensor) = benv_tensor(ket, ket, $open_axs)
        $(fname)(ket::StateTensor, h) = benv_tensor(ket, ket, $open_axs, [h])
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
        ket::T, bra::T = ket
    ) where {E, S, T <: StateTensor}
    if T <: PEPSOrth
        return @tensoropt ctl2[:] := ctl[D11 D10 D21 D20] * et[-5 -6 D31 D30 D11 D10] *
            el[D21 D20 D41 D40 -1 -2] * conj(bra[D31 -7 -3 D41]) * ket[D30 -8 -4 D40]
    else
        ket2 = (T <: PEPOTensor) ? first(fuse_physicalspaces(ket)) : twistdual(ket, 1)
        bra2 = (T <: PEPOTensor) ? first(fuse_physicalspaces(bra)) : bra
        return @tensoropt ctl2[:] := ctl[D11 D10 D21 D20] * et[-5 -6 D31 D30 D11 D10] *
            el[D21 D20 D41 D40 -1 -2] * conj(bra2[d D31 -7 -3 D41]) * ket2[d D30 -8 -4 D40]
    end
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
        ket::T, bra::T = ket
    ) where {E, S, T <: StateTensor}
    if T <: PEPSOrth
        return @tensoropt cbr2[:] := cbr[D31 D30 D41 D40] * eb[D21 D20 D41 D40 -7 -8] *
            er[-3 -4 D31 D30 D11 D10] * conj(bra[-1 D11 D21 -5]) * ket[-2 D10 D20 -6]
    else
        ket2 = (T <: PEPOTensor) ? first(fuse_physicalspaces(ket)) : twistdual(ket, 1)
        bra2 = (T <: PEPOTensor) ? first(fuse_physicalspaces(bra)) : bra
        return @tensoropt cbr2[:] := cbr[D31 D30 D41 D40] * eb[D21 D20 D41 D40 -7 -8] *
            er[-3 -4 D31 D30 D11 D10] * conj(bra2[d -1 D11 D21 -5]) * ket2[d -2 D10 D20 -6]
    end
end
