"""
Attach the ancilla space `Va` and merge with physical space `Vp` of
the physical operator `op`, so the new physical space becomes `Vp ⊗ Va`.
"""
function attach_ancilla(
        op::AbstractTensorMap{T, S, N, N}; Va0::Union{ElementarySpace, Nothing} = nothing
    ) where {T, S <: ElementarySpace, N}
    Vp = first(codomain(op))
    Va = isnothing(Va0) ? Vp' : Va0
    id_Va = TensorKit.id(Va)
    op_a = op ⊗ reduce(⊗, fill(id_Va, N))
    perm = vcat(1:2:2N, 2:2:2N)
    perm = invperm(Tuple(vcat(perm, perm .+ 2N)))
    op_a = permute(op_a, ((perm[1:2N], perm[(2N + 1):end])))
    # fuse physical and ancilla legs
    fuser = isometry(T, fuse(Vp, Va), Vp ⊗ Va)
    f_idxs = collect([-n, 2n - 1, 2n] for n in 1:2N)
    op_a = ncon(
        vcat(op_a, fill(fuser, 2N)...),
        vcat([collect(1:4N)], f_idxs),
        vcat(fill(false, N + 1), fill(true, N))
    )
    return permute(op_a, (Tuple(1:N), Tuple((N + 1):2N)))
end

"""
Attach the ancilla space to all terms in the LocalOperator `O`
"""
function attach_ancilla(O::LocalOperator; Va0 = nothing)
    return LocalOperator(
        collect(fuse(Vp, isnothing(Va0) ? Vp' : Va0) for Vp in physicalspace(O)),
        (sites => attach_ancilla(op; Va0) for (sites, op) in O.terms)...
    )
end
