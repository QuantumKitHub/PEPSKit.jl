"Reduced bond factor `R[q; physical bond]`, with one outgoing and two incoming legs."
const ReducedBondTensor{T, S} = AbstractTensorMap{T, S, 1, 2}

function _bond_tensor_axes(axes::Tuple, dir, gate_ax)
    Np = length(axes) - 4
    @assert 1 <= dir <= 4
    @assert 1 <= gate_ax <= Np
    reduced = (gate_ax, Np + dir)
    return TupleTools.deleteat(axes, reduced), reduced
end

"""
    bond_tensor_end(A, dir; gate_ax = 1, allocator = DefaultAllocator(), kwargs...)

Use `left_orth` to factor out the physical axis `gate_ax` and
virtual leg `dir` of `A`, returning `(X, R)` with `R[q; physical bond]`.

For a PEPS tensor `A` with `dir = EAST`, `gate_ax = 1`:
```
        n                       n
        │                       │
    w ─ A ─ e       =       w ─ X ← q ← R ─ e
        │ ╲                     │         ╲
        s  p                    s          p
```
"""
function bond_tensor_end(A::AbstractTensorMap{T, S, Np, 4}, dir::Int; gate_ax::Integer = 1, allocator = DefaultAllocator(), kwargs...) where {T, S, Np}
    open, reduced = _bond_tensor_axes(allind(A), dir, gate_ax)
    partition = (open, reduced)
    cp = allocator_checkpoint!(allocator)
    A′ = tensoralloc_add(scalartype(A), A, partition, false, Val(true), allocator)
    tensoradd!(A′, A, partition, false, One(), Zero(), DefaultBackend(), allocator)
    X, R = left_orth!(A′; kwargs...)
    tensorfree!(A′, allocator)
    allocator_reset!(allocator, cp)
    order = TupleTools.insertafter(codomainind(X), Np + dir - 2, domainind(X))
    # TODO: can we cleanly avoid this?
    X = permute(X, order)
    return X, R
end

"""
    undo_bond_tensor_end(R, X, dir; gate_ax = 1, allocator = DefaultAllocator())

Undo `bond_tensor_end`, reconstructing directly in the original frame.
`R` has layout `R[q; physical bond]`; `X` and the result use the original frame.
"""
function undo_bond_tensor_end(R::ReducedBondTensor, X::AbstractTensorMap, dir::Int; gate_ax::Integer = 1, allocator = DefaultAllocator())
    Np = numind(X) - 3
    open, reduced = _bond_tensor_axes(ntuple(identity, Np + 4), dir, gate_ax)
    contracted = Np - 1 + dir
    free = TupleTools.deleteat(allind(X), contracted)
    # Restore the original axes from the contraction's (open..., physical, bond) order.
    order = invperm((open..., reduced...))
    output = (order[1:Np], order[(Np + 1):end])
    cp = allocator_checkpoint!(allocator)
    A = tensorcontract(
        X, (free, (contracted,)), false, R, ((1,), (2, 3)), false,
        output, One(), DefaultBackend(), allocator
    )
    allocator_reset!(allocator, cp)
    return A
end

"""
    bond_tensor_mid(A, dir; allocator = DefaultAllocator(), kwargs...)

Use QR decomposition to factor out virtual leg `dir` of `A`,
returning `(X, R)` with the physical axes kept in `X`.

For a PEPS tensor `A` with `dir = EAST`:
```
        n                       n
        │                       │
    w ─ A ─ e       =       w ─ X ← q ← R ─ e
        │ ╲                     │ ╲       ╲
        s  p                    s  p       (1)
```
The `(1)` leg of `R[q; auxiliary bond]` is a trivial physical leg.
"""
function bond_tensor_mid(A::Union{PEPSTensor, PEPOTensor}, dir::Int; allocator = DefaultAllocator(), kwargs...)
    # insert the auxiliary trivial physical leg in front of existing ones
    # so it can be factored into R using `bond_tensor_end` with `gate_ax = 1`
    X, R = bond_tensor_end(insertleftunit(A, Val(1)), dir; allocator, kwargs...)
    return permute(X, (codomainind(A), domainind(A))), R
end

"""
    undo_bond_tensor_mid(R, X, dir; allocator = DefaultAllocator())

Reconstruct a middle tensor with `undo_bond_tensor_end`, then remove the trivial
physical leg introduced by `bond_tensor_mid`.
"""
function undo_bond_tensor_mid(R::ReducedBondTensor, X::Union{PEPSTensor, PEPOTensor}, dir::Int; allocator = DefaultAllocator())
    A = undo_bond_tensor_end(R, X, dir; allocator)
    return removeunit(A, Val(1))
end
