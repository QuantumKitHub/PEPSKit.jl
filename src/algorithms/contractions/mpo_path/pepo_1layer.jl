"""
Check that the physical legs of a first OBC-MPO tensor match the PEPO site's physical space.
"""
function _check_pepo_first_physicalspace(A, op)
    physicalspace(A) == space(op, 1) == space(op, 2)' ||
        throw(SpaceMismatch("first MPO tensor physical space does not match PEPO site"))
    return nothing
end

"""
Check that the physical legs of a last OBC-MPO tensor match the PEPO site's physical space.
"""
function _check_pepo_last_physicalspace(A, op)
    physicalspace(A) == space(op, 2) == space(op, 3)' ||
        throw(SpaceMismatch("last MPO tensor physical space does not match PEPO site"))
    return nothing
end

"""
Check that the physical legs of a middle MPO tensor match the PEPO site's physical space.
"""
function _check_pepo_middle_physicalspace(A, op)
    physicalspace(A) == space(op, 2) == space(op, 3)' ||
        throw(SpaceMismatch("middle MPO tensor physical space does not match PEPO site"))
    return nothing
end

"""
Convert a symbolic cardinal path direction to the corresponding PEPO virtual-leg index.
"""
function _mpo_path_direction(direction::Symbol)
    direction === :north && return NORTH
    direction === :east && return EAST
    direction === :south && return SOUTH
    direction === :west && return WEST
    throw(ArgumentError("invalid MPO path direction: $direction"))
end

"""
Return the tensor-expression label for the PEPO virtual leg in a cardinal direction.
"""
function _mpo_path_virtual_label(direction::Symbol)
    return (:N, :E, :S, :W)[_mpo_path_direction(direction)]
end

"""
Build the `@tensor` labels from `(direction, suffix)` pairs
used to fuse MPO virtual strings.

- Each direction is `:north`, `:east`, `:south`, or `:west`.
- Suffix `:l` marks an incoming MPO bond and `:r` an outgoing one.

Examples:

- `((:east, :r),)` produces `[W S; N Er]`.
- `((:west, :l), (:north, :r))` produces `[Wl S; Nr E]`.
"""
function _mpo_path_result_expr(directions)
    labels = [:N, :E, :S, :W]
    for (direction, suffix) in directions
        index = _mpo_path_direction(direction)
        labels[index] = Symbol(labels[index], suffix)
    end
    return tensorexpr(:t, (labels[WEST], labels[SOUTH]), (labels[NORTH], labels[EAST]))
end

"""
Act the first tensor `op` of an OBC-MPO on PEPO tensor `A` and fuse the
outgoing MPO string with the virtual space of `A` along `direction`.
"""
@generated function mpo_path_first(A::PEPOTensor, op, ::Val{direction}) where {direction}
    direction_index = _mpo_path_direction(direction)
    virtual_label = _mpo_path_virtual_label(direction)
    fused_label = Symbol(virtual_label, :r)

    result_e = _mpo_path_result_expr(((direction, :r),))
    op_e = tensorexpr(:op, :dout, (:din, :r))
    A_e = tensorexpr(:A′, (:din, :dout), (:N, :E, :S, :W))
    F_e = tensorexpr(:F, fused_label, (virtual_label, :r))
    rhs = Expr(:call, :*, op_e, A_e, F_e)
    contraction = macroexpand(
        @__MODULE__, :(return @tensoropt $result_e := $rhs)
    )

    return quote
        _check_pepo_first_physicalspace(A, op)
        A′ = twistdual(A, 2)
        F = fuser(storagetype(A), domain(A, $direction_index)', space(op, 3))
        $contraction
    end
end

"""
Act the last tensor `op` of an OBC-MPO on PEPO tensor `A` and fuse the
incoming MPO string with the virtual space of `A` along `direction`.
"""
@generated function mpo_path_last(A::PEPOTensor, op, ::Val{direction}) where {direction}
    direction_index = _mpo_path_direction(direction)
    virtual_label = _mpo_path_virtual_label(direction)
    fused_label = Symbol(virtual_label, :l)

    result_e = _mpo_path_result_expr(((direction, :l),))
    F_e = Expr(:call, :conj, tensorexpr(:F, fused_label, (virtual_label, :l)))
    op_e = tensorexpr(:op, (:l, :dout), :din)
    A_e = tensorexpr(:A′, (:din, :dout), (:N, :E, :S, :W))
    rhs = Expr(:call, :*, F_e, op_e, A_e)
    contraction = macroexpand(
        @__MODULE__, :(return @tensoropt $result_e := $rhs)
    )

    return quote
        _check_pepo_last_physicalspace(A, op)
        A′ = twistdual(A, 2)
        F = fuser(storagetype(A), domain(A, $direction_index), space(op, 1)')
        $contraction
    end
end

"""
Act the middle tensor `op` of an MPO on PEPO tensor `A` and fuse the
incoming and the outgoing MPO string with the virtual space of `A` along
`directions = (incoming, outgoing)`.
"""
@generated function mpo_path_middle(
        A::PEPOTensor, op, ::Val{directions}
    ) where {directions}
    incoming, outgoing = directions
    incoming == outgoing &&
        throw(ArgumentError("MPO path should enter and exit in different directions"))

    incoming_index = _mpo_path_direction(incoming)
    outgoing_index = _mpo_path_direction(outgoing)
    incoming_label = _mpo_path_virtual_label(incoming)
    outgoing_label = _mpo_path_virtual_label(outgoing)
    fused_incoming_label = Symbol(incoming_label, :l)
    fused_outgoing_label = Symbol(outgoing_label, :r)

    result_e = _mpo_path_result_expr(((incoming, :l), (outgoing, :r)))
    Fin_e = Expr(
        :call, :conj,
        tensorexpr(:Fin, fused_incoming_label, (incoming_label, :l)),
    )
    op_e = tensorexpr(:op, (:l, :dout), (:din, :r))
    A_e = tensorexpr(:A′, (:din, :dout), (:N, :E, :S, :W))
    Fout_e = tensorexpr(
        :Fout, fused_outgoing_label, (outgoing_label, :r)
    )
    rhs = Expr(:call, :*, Fin_e, op_e, A_e, Fout_e)
    contraction = macroexpand(
        @__MODULE__, :(return @tensoropt $result_e := $rhs)
    )

    return quote
        _check_pepo_middle_physicalspace(A, op)
        A′ = twistdual(A, 2)
        Fin = fuser(storagetype(A), domain(A, $incoming_index), space(op, 1)')
        Fout = fuser(storagetype(A), domain(A, $outgoing_index)', space(op, 4))
        $contraction
    end
end

"""
Route an MPO virtual string with `stringspace` through a PEPO tensor `A`
along `directions = (incoming, outgoing)`.
"""
@generated function mpo_path_string(
        A::PEPOTensor, stringspace::ElementarySpace, ::Val{directions}
    ) where {directions}
    incoming, outgoing = directions
    incoming == outgoing &&
        throw(ArgumentError("MPO path should enter and exit in different directions"))

    incoming_index = _mpo_path_direction(incoming)
    outgoing_index = _mpo_path_direction(outgoing)
    incoming_label = _mpo_path_virtual_label(incoming)
    outgoing_label = _mpo_path_virtual_label(outgoing)
    fused_incoming_label = Symbol(incoming_label, :l)
    fused_outgoing_label = Symbol(outgoing_label, :r)

    result_e = _mpo_path_result_expr(((incoming, :l), (outgoing, :r)))
    Fin_e = Expr(
        :call, :conj,
        tensorexpr(:Fin, fused_incoming_label, (incoming_label, :l)),
    )
    O_e = tensorexpr(:O, (:W, :S), (:N, :E))
    I_e = tensorexpr(:I, :l, :r)
    Fout_e = tensorexpr(
        :Fout, fused_outgoing_label, (outgoing_label, :r)
    )
    rhs = Expr(:call, :*, Fin_e, O_e, I_e, Fout_e)
    contraction = macroexpand(
        @__MODULE__, :(return @tensoropt $result_e := $rhs)
    )

    return quote
        O = trace_physicalspaces(A)
        I = id(storagetype(A), stringspace)
        Fin = fuser(storagetype(A), domain(A, $incoming_index), stringspace')
        Fout = fuser(storagetype(A), domain(A, $outgoing_index)', stringspace')
        $contraction
    end
end
