#= 
The construction of bond environment for Neighborhood Tensor Update (NTU) 
is adapted from YASTN (https://github.com/yastn/yastn).
Copyright 2024 The YASTN Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0
=#

const BondEnv{T,S} = AbstractTensorMap{T,S,2,2} where {T<:Number,S<:ElementarySpace}
const Hair{T,S} = AbstractTensor{T,S,2} where {T<:Number,S<:ElementarySpace}
const PEPSOrth{T,S} = AbstractTensor{T,S,4} where {T<:Number,S<:ElementarySpace}

"""
Extract tensors in an infinite PEPS 
at positions `neighbors` relative to `(row, col)`
"""
function collect_neighbors(
    peps::InfinitePEPS, row::Int, col::Int, neighbors::Vector{Tuple{Int,Int}}
)
    Nr, Nc = size(peps)
    return Dict(
        nb => peps.A[mod1(row + nb[1], Nr), mod1(col + nb[2], Nc)] for nb in neighbors
    )
end

"""
Extract tensors in an infinite PEPS with weight 
at positions `neighbors` relative to `(row, col)`
"""
function collect_neighbors(
    peps::InfiniteWeightPEPS,
    row::Int,
    col::Int,
    neighbors::Vector{Tuple{Int,Int,Vector{Int}}},
    add_bwt::Bool=true,
)
    Nr, Nc = size(peps)
    axs = Tuple(1:4)
    return Dict(
        (x, y) => begin
            r, c = mod1(row + x, Nr), mod1(col + y, Nc)
            sqrts = if add_bwt
                Tuple(!(ax in open_axs) for ax in 1:4)
            else
                ntuple(Returns(true), 4)
            end
            _absorb_weights(peps.vertices[r, c], peps.weights, r, c, axs, sqrts, false)
        end for (x, y, open_axs) in neighbors
    )
end

"""
    benv_tensor(open_axs::Vector{Int}, ket::T, bra::T) where {T<:Union{PEPSTensor,PEPSOrth}}
    benv_tensor(open_axs::Vector{Int}, ket::T, bra::T, axts::Vector{H}) where {T<:Union{PEPSTensor,PEPSOrth},H<:Union{Nothing,Hair}}

Contract the physical axes (for PEPSTensor) and the virtual axes of `ket` with `bra` to obtain the tensor on the boundary of the bond environment. Virtual axes specified by `open_axs` (in ascending order) are not contracted. 

# Examples

- Left "hair" tensor when `ket`, `bra` are `PEPSTensor`
(`open_axs = [2]`) 
```
             ╱|
    ┌-----bra----- 1
    |    ╱ |  |
    |   |  |  |
    |   |  | ╱
    └---|-ket----- 2
        |╱
```

- Upper-left corner tensor when `ket`, `bra` are `PEPSOrth`
(`open_axs = [2, 3]`, `axts = [t, nothing]`)
(fermion signs should not be cancelled when contracting `t`)
```
             ╱|
    ┌-----bra----- 1
    |    ╱    t
    |   3     |
    |         |
    |        ╱
    └-----ket----- 2
         ╱
        4
```

- Left edge tensor when `ket`, `bra` are `PEPSTensor`
(`open_axs = [1, 2, 3]`)
```
               1
             ╱
    ┌-----bra----- 3
    |    ╱ |
    |   5  |   2
    |      | ╱
    └-----ket----- 4
          ╱
        6
```
"""
function benv_tensor(
    open_axs::Vector{Int}, ket::T, bra::T
) where {T<:Union{PEPSTensor,PEPSOrth}}
    @assert issorted(open_axs)
    @assert all(1 <= ax <= 4 for ax in open_axs)
    open_axs2 = (T <: PEPSTensor ? open_axs .+ 1 : open_axs)
    axs = 1:(T <: PEPSTensor ? 5 : 4)
    codomain_axes = Tuple(ax for ax in axs if ax ∉ open_axs2)
    domain_axes = Tuple(open_axs2)
    perm = (codomain_axes, domain_axes)
    t = adjoint(permute(bra, perm)) * permute(ket, perm)
    n = length(open_axs2)
    return permute(t, Tuple(Iterators.flatten(zip(1:n, (n + 1):(2n)))))
end
function benv_tensor(
    open_axs::Vector{Int}, ket::T, bra::T, axts::Vector{H}
) where {T<:Union{PEPSTensor,PEPSOrth},H<:Union{Nothing,Hair}}
    @assert length(axts) == 4 - length(open_axs)
    ket2 = deepcopy(ket)
    axs = (T <: PEPSTensor) ? (2:5) : (1:4)
    open_axs2 = (T <: PEPSTensor ? open_axs .+ 1 : open_axs)
    for (axt, ax) in zip(axts, Tuple(ax for ax in axs if ax ∉ open_axs2))
        if axt === nothing
            twist!(ket2, ax)
            continue
        end
        @assert space(axt, 1) == space(axt, 2)'
        ket_indices = collect(-1:-1:((T <: PEPSTensor) ? -5 : -4))
        ket_indices[ax] = 1
        # apply `axt` to virtual indices of `ket` to be contracted
        ket2 = ncon([axt, ket2], [[-ax, 1], ket_indices])
    end
    nax = (T <: PEPSTensor) ? 5 : 4
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

            |           | /         |
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
        $(fname)(ket::PEPSTensor) = benv_tensor($open_axs, ket, ket)
        $(fname)(ket::PEPSOrth) = benv_tensor($open_axs, ket, ket)
        $(fname)(ket::PEPSTensor, h1, h2, h3) =
            benv_tensor($open_axs, ket, ket, [h1, h2, h3])
        $(fname)(ket::PEPSOrth, h1, h2, h3) = benv_tensor($open_axs, ket, ket, [h1, h2, h3])
    end
end

# construction of corners
for (dir, open_axs) in open_axs_cor
    fname = Symbol("cor_", dir)
    @eval begin
        $(fname)(ket::PEPSTensor) = benv_tensor($open_axs, ket, ket)
        $(fname)(ket::PEPSOrth) = benv_tensor($open_axs, ket, ket)
        $(fname)(ket::PEPSTensor, h1, h2) = benv_tensor($open_axs, ket, ket, [h1, h2])
        $(fname)(ket::PEPSOrth, h1, h2) = benv_tensor($open_axs, ket, ket, [h1, h2])
    end
end

# construction of edges
for (dir, open_axs) in open_axs_edge
    fname = Symbol("edge_", dir)
    @eval begin
        $(fname)(ket::PEPSTensor) = benv_tensor($open_axs, ket, ket)
        $(fname)(ket::PEPSOrth) = benv_tensor($open_axs, ket, ket)
        $(fname)(ket::PEPSTensor, h) = benv_tensor($open_axs, ket, ket, [h])
        $(fname)(ket::PEPSOrth, h) = benv_tensor($open_axs, ket, ket, [h])
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
    ctl::AbstractTensor{T,S,4},
    et::AbstractTensor{T,S,6},
    el::AbstractTensor{T,S,6},
    ket::PEPSOrth{T,S},
    bra::PEPSOrth{T,S}=ket,
) where {T<:Number,S<:ElementarySpace}
    return @tensor ctl2[:] :=
        ctl[D11 D10 D21 D20] *
        et[-5 -6 D31 D30 D11 D10] *
        el[D21 D20 D41 D40 -1 -2] *
        conj(bra[D31 -7 -3 D41]) *
        ket[D30 -8 -4 D40]
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
    cbr::AbstractTensor{T,S,4},
    eb::AbstractTensor{T,S,6},
    er::AbstractTensor{T,S,6},
    ket::PEPSOrth{T,S},
    bra::PEPSOrth{T,S}=ket,
) where {T<:Number,S<:ElementarySpace}
    return @tensor cbr2[:] :=
        cbr[D31 D30 D41 D40] *
        eb[D21 D20 D41 D40 -7 -8] *
        er[-3 -4 D31 D30 D11 D10] *
        conj(bra[-1 D11 D21 -5]) *
        ket[-2 D10 D20 -6]
end
