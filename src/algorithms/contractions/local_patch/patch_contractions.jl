# Low-level patch contractions using generated contraction expressions
# --------------------------------------------------------------------

"""
$(SIGNATURES)

Contract the rectangular patch of a network encoded by `state`, spanned by `inds`, with
`operator` inserted on those sites, leaving no physical leg open, and return the resulting
scalar.

The sites are carried as `Val` parameters so that the patch geometry is available while the
contraction is generated. The expression is assembled from [`boundary_contraction_expr`](@ref),
[`bulk_contraction_expr`](@ref) and [`operator_contraction_expr`](@ref), which dispatch on the
types of `env`, `state` and `operator` respectively, so this single method covers every
combination those three have methods for.
"""
@generated function _contract_local_operator(
        inds::NTuple{N, Val}, operator, state, env
    ) where {N}
    sites = _patch_inds(inds)
    rowrange, colrange = _patch_ranges(sites)

    multiplication_ex = Expr(
        :call, :*,
        boundary_contraction_expr(env, rowrange, colrange)...,
        bulk_contraction_expr(state, rowrange, colrange, sites)...,
        operator_contraction_expr(operator, N)...,
    )

    returnex = _tensor_expr(multiplication_ex)
    return macroexpand(@__MODULE__, returnex)
end

"""
$(SIGNATURES)

Contract the rectangular patch of a network encoded by `state`, spanned by `inds`, with no
operator inserted, pairing the physical legs of the layers on every site, and return the
resulting scalar.

Assembled as [`_contract_local_operator`](@ref), except that
[`bulk_contraction_expr`](@ref) is passed `nothing` in place of the open sites, so no physical
leg is left open and no operator factor is needed. Note this is the norm of the patch within
the given environment, not the physical norm of the state.
"""
@generated function _contract_local_norm(
        inds::NTuple{N, Val}, state, env
    ) where {N}
    sites = _patch_inds(inds)
    rowrange, colrange = _patch_ranges(sites)

    multiplication_ex = Expr(
        :call, :*,
        boundary_contraction_expr(env, rowrange, colrange)...,
        bulk_contraction_expr(state, rowrange, colrange, nothing)...,   # legs paired, not open
    )

    returnex = _tensor_expr(multiplication_ex)
    return macroexpand(@__MODULE__, returnex)
end
