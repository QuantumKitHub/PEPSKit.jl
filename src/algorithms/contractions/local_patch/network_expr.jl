# Local patch contraction expression generators
# ---------------------------------------------

# Assembly helpers
# ----------------

"""
$(SIGNATURES)

Wrap an assembled product in `@autoopt @tensor`, with `lhs` on the left if given and as a
scalar otherwise.
"""
function _tensor_expr(prod, lhs = nothing)
    return isnothing(lhs) ? :(@autoopt @tensor $prod) : :(@autoopt @tensor $lhs := $prod)
end


# Contraction expression generators
# ---------------------------------

## Boundary: environment

"""
    boundary_contraction_expr(::Type{Env}, rowrange, colrange)

Build the contraction expressions for the environment factors surrounding the patch spanned
by `rowrange` and `colrange`, dispatching on the type of the environment.

The returned factors must consume exactly the perimeter labels the bulk exposes -
`virtuallabel(NORTH, layer, j)`, `virtuallabel(EAST, layer, i)`,
`virtuallabel(SOUTH, layer, j)` and `virtuallabel(WEST, layer, i)`, one per layer - and must
close every label they introduce themselves among their own factors.

!!! note
    The expression generator assumes the environment variable name is `env`.
"""
boundary_contraction_expr(env::Type, rowrange, colrange) = throw(
    ArgumentError("No patch boundary contraction defined for environments of type $env.")
)

function boundary_contraction_expr(
        ::Type{<:CTMRGEnv{C, T}}, rowrange, colrange
    ) where {C, T}
    # the edges carry one virtual leg per layer of the sandwich, on top of the two environment
    # indices threading the ring, so the height follows from the edge tensor type
    height = numout(T) - 1
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    C_NW = :(corner(env, NORTHWEST, $(rmin - 1), $(cmin - 1)))
    corner_NW = tensorexpr(C_NW, envlabel(WEST, 0), envlabel(NORTH, 0))

    C_NE = :(corner(env, NORTHEAST, $(rmin - 1), $(cmax + 1)))
    corner_NE = tensorexpr(C_NE, envlabel(NORTH, gridsize[2]), envlabel(EAST, 0))

    C_SE = :(corner(env, SOUTHEAST, $(rmax + 1), $(cmax + 1)))
    corner_SE = tensorexpr(C_SE, envlabel(EAST, gridsize[1]), envlabel(SOUTH, gridsize[2]))

    C_SW = :(corner(env, SOUTHWEST, $(rmax + 1), $(cmin - 1)))
    corner_SW = tensorexpr(C_SW, envlabel(SOUTH, 0), envlabel(WEST, gridsize[1]))

    edges_N = map(1:gridsize[2]) do i
        E_N = :(edge(env, NORTH, $(rmin - 1), $(cmin + i - 1)))
        return tensorexpr(
            E_N,
            (envlabel(NORTH, i - 1), virtuallabel.(NORTH, ntuple(identity, height), i)...),
            envlabel(NORTH, i),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        E_E = :(edge(env, EAST, $(rmin + i - 1), $(cmax + 1)))
        return tensorexpr(
            E_E,
            (envlabel(EAST, i - 1), virtuallabel.(EAST, ntuple(identity, height), i)...),
            envlabel(EAST, i),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        E_S = :(edge(env, SOUTH, $(rmax + 1), $(cmin + i - 1)))
        return tensorexpr(
            E_S,
            (envlabel(SOUTH, i), virtuallabel.(SOUTH, ntuple(identity, height), i)...),
            envlabel(SOUTH, i - 1),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        E_W = :(edge(env, WEST, $(rmin + i - 1), $(cmin - 1)))
        return tensorexpr(
            E_W,
            (envlabel(WEST, i), virtuallabel.(WEST, ntuple(identity, height), i)...),
            envlabel(WEST, i - 1),
        )
    end

    return [
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
    ]
end


## Bulk: state

"""
    bulk_contraction_expr(::Type{State}, rowrange, colrange, open)

Build the contraction expressions for the factors inside the patch spanned by `rowrange`
and `colrange`, dispatching on the type of the state.

`open` is the vector of sites whose physical legs are left open, or `nothing` when every
physical leg is contracted within the bulk. Open sites are labelled
`physicallabel(:O, layer, slot)`, where `slot` is the position of the site in `open`; closed
sites share a label between the layers. Physical labels are this generator's business alone.

Interior bonds must be closed among the returned factors, which must expose their
perimeter-facing virtual legs under the labels [`boundary_contraction_expr`](@ref) consumes.

!!! note
    The expression generator assumes the state variable name is `state`.
"""
bulk_contraction_expr(state::Type, rowrange, colrange, open) = throw(
    ArgumentError("No patch bulk contraction defined for states of type $state.")
)

function bulk_contraction_expr(
        ::Type{<:Tuple{InfinitePEPS, InfinitePEPS}}, rowrange, colrange, open
    )
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    layers = map(1:2) do side
        return map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
            inds_id = _open_slot(open, rmin + i - 1, cmin + j - 1)
            physical_label = if isnothing(inds_id)
                physicallabel(i, j)
            else
                physicallabel(:O, side, inds_id)
            end
            return tensorexpr(
                :(state[$(side)][$(rmin + i - 1), $(cmin + j - 1)]),
                (physical_label,),
                _bulk_virtuallabels(i, j, side, gridsize),
            )
        end
    end

    ket, bra = layers
    return [ket..., map(x -> Expr(:call, :conj, x), bra)...]
end

function bulk_contraction_expr(::Type{<:InfinitePEPO}, rowrange, colrange, open)
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    # a single layer is not wrapped in a tuple, so it is indexed directly
    layer = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = _open_slot(open, rmin + i - 1, cmin + j - 1)
        physical_label_out = if isnothing(inds_id)
            physicallabel(i, j)     # traced over the layer
        else
            physicallabel(:O, 1, inds_id)
        end
        physical_label_in = if isnothing(inds_id)
            physicallabel(i, j)
        else
            physicallabel(:O, 2, inds_id)
        end
        return tensorexpr(
            :(twistdual(state[$(rmin + i - 1), $(cmin + j - 1)], 2)),
            (physical_label_out, physical_label_in),
            _bulk_virtuallabels(i, j, 1, gridsize),
        )
    end

    return vec(layer)
end

function bulk_contraction_expr(
        ::Type{<:Tuple{InfinitePEPO, InfinitePEPO}}, rowrange, colrange, open
    )
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    layers = map(1:2) do side
        return map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
            inds_id = _open_slot(open, rmin + i - 1, cmin + j - 1)
            physical_label_out = if isnothing(inds_id)
                physicallabel(:out, i, j)
            else
                physicallabel(:O, side, inds_id)
            end
            # the two layers are linked through a shared label on the open sites
            physical_label_in = if isnothing(inds_id)
                physicallabel(:in, i, j)
            else
                physicallabel(:Oopen, inds_id)
            end
            tensor_name = if side == 2
                :(state[2][$(rmin + i - 1), $(cmin + j - 1)])
            else
                :(twistdual(state[1][$(rmin + i - 1), $(cmin + j - 1)], (1, 2)))
            end
            return tensorexpr(
                tensor_name, (physical_label_out, physical_label_in),
                _bulk_virtuallabels(i, j, side, gridsize),
            )
        end
    end

    ket, bra = layers
    return [ket..., map(x -> Expr(:call, :conj, x), bra)...]
end

function bulk_contraction_expr(
        state::Type{<:Tuple{Vararg{InfinitePEPO}}}, rowrange, colrange, open
    )
    return throw(
        ArgumentError(
            "Cannot contract a patch of $(length(state.parameters)) PEPO layers; only a \
            single layer or a two-layer sandwich are supported."
        )
    )
end


## Operator

"""
    operator_contraction_expr(::Type{Operator}, nsites)

Build the contraction expressions for the operator factors inserted into a patch acting on
`nsites` sites, dispatching on the type of the operator.

The factors refer to the enclosing generated function's `operator` argument by name, and
attach to the open physical legs of the bulk: `physicallabel(:O, 2, k)` is the bra index of
site `k` and `physicallabel(:O, 1, k)` its ket index. Every state type labels its open legs
with that same pair, so this generator does not depend on the state. Any label it introduces
itself must be closed among its own factors.

!!! note
    The expression generator assumes the operator variable name is `operator`.
"""
operator_contraction_expr(operator::Type, nsites) = throw(
    ArgumentError("No patch operator contraction defined for operators of type $operator.")
)

function operator_contraction_expr(::Type{<:AbstractTensorMap}, nsites)
    return [
        tensorexpr(
            :operator,
            ntuple(i -> physicallabel(:O, 2, i), nsites),
            ntuple(i -> physicallabel(:O, 1, i), nsites),
        ),
    ]
end

# The MPO bond is small - the operator's Schmidt rank, e.g. 3 for Heisenberg XYZ - so it is
# labelled as a physical rather than a virtual dimension, which gives `@autoopt` the right
# order of magnitude when it searches for a contraction order.
mpolabel(args...) = physicallabel(:mpo, args...)

function operator_contraction_expr(::Type{<:MPOTerm}, nsites)
    # one factor per site, linked by a chain of bonds running left to right, rank-2 at the
    # ends of the chain and rank-4 in the bulk:
    #   W₁  : bra₁ ← ket₁ ⊗ b₁
    #   Wᵢ  : bᵢ₋₁ ⊗ braᵢ ← ketᵢ ⊗ bᵢ
    #   W_N : b_{N-1} ⊗ bra_N ← ket_N
    return map(1:nsites) do i
        bra, ket = physicallabel(:O, 2, i), physicallabel(:O, 1, i)
        out, ins = if nsites == 1
            (bra,), (ket,)
        elseif i == 1
            (bra,), (ket, mpolabel(1))
        elseif i == nsites
            (mpolabel(nsites - 1), bra), (ket,)
        else
            (mpolabel(i - 1), bra), (ket, mpolabel(i))
        end
        return tensorexpr(:(operator[$i]), out, ins)
    end
end

# A tensor product term is the special case in which every bond has dimension 1, so its
# factors carry no bonds at all: each simply sits between the ket and bra index of its site.
# This method is more specific than the `MPOTerm` one above, so it takes precedence.
function operator_contraction_expr(::Type{<:TensorProductTerm}, nsites)
    return map(1:nsites) do i
        return tensorexpr(
            :(operator[$i]), (physicallabel(:O, 2, i),), (physicallabel(:O, 1, i),)
        )
    end
end
