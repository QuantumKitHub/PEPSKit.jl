# Generic reduced density matrix patch contraction
# ------------------------------------------------

"""
$(SIGNATURES)

Contract the rectangular patch of a network encoded by `state`, spanned by `inds`, leaving
the physical legs of those sites open, and return the resulting reduced density matrix, normalized by its supertrace.

Assembled as [`_contract_local_operator`](@ref) but without an operator factor, so the open
legs become the indices of `ρ`: `physicallabel(:O, 1, k)` in its codomain and
`physicallabel(:O, 2, k)` in its domain, for each site `k`. Normalization uses `str` rather
than `tr`, since the supertrace carries the fermionic signs.
"""
@generated function _contract_densitymatrix(
        inds::NTuple{N, Val}, state, env
    ) where {N}
    sites = _patch_inds(inds)
    rowrange, colrange = _patch_ranges(sites)

    multiplication_ex = Expr(
        :call, :*,
        boundary_contraction_expr(env, rowrange, colrange)...,
        bulk_contraction_expr(state, rowrange, colrange, sites)...,
    )
    result = tensorexpr(
        :ρ,
        ntuple(i -> physicallabel(:O, 1, i), N),
        ntuple(i -> physicallabel(:O, 2, i), N),
    )

    multex = _tensor_expr(multiplication_ex, result)
    return quote
        $(macroexpand(@__MODULE__, multex))
        return ρ / str(ρ)
    end
end
