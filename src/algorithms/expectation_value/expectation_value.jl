# Expectation value of a LocalOperator
# ------------------------------------

"""
    expectation_value(state, O::LocalOperator, env)
    expectation_value(bra, O::LocalOperator, ket, env)

Compute the expectation value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ or tr(O * state) / tr(state) of a [`LocalOperator`](@ref) `O`.
This can be done either for a PEPS, or alternatively for a density matrix PEPO.
In the latter case the first signature corresponds to a single layer PEPO contraction, while
the second signature yields a bilayer contraction instead.
"""
function MPSKit.expectation_value(
        bra::S, O::LocalOperator, ket::S, env
    ) where {S <: InfiniteState}
    checklattice(bra, O, ket)
    term_vals = dtmap(collect(O.terms)) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        return local_expectation_value(inds, bra, operator, ket, env)
    end
    return sum(term_vals)
end
MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, env) = expectation_value(peps, O, peps, env)
function MPSKit.expectation_value(state::InfinitePEPO, O::LocalOperator, env)
    checklattice(state, O)
    term_vals = dtmap(collect(O.terms)) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        return local_expectation_value(inds, state, operator, env)
    end
    return sum(term_vals)
end


# Expectation value of an individual local term
# ---------------------------------------------

"""
    local_expectation_value(inds, bra, operator, ket, env)
    local_expectation_value(inds, state, operator, env)

Compute the contribution of a single term of a [`LocalOperator`](@ref) to the expectation
value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ or tr(O * state) / tr(state), where `operator` is the local
term acting on the sites `inds`.

The implementation is overloaded based on the type of operator to be evaluated
"""
function local_expectation_value end

# AbstractTensorMap evaluation goes through reduced density matrix
function local_expectation_value(inds, bra, operator::AbstractTensorMap, ket, env)
    ρ = reduced_densitymatrix(inds, ket, bra, env)
    return trmul(operator, ρ)
end
function local_expectation_value(inds, state, operator::AbstractTensorMap, env)
    ρ = reduced_densitymatrix(inds, state, env)
    return trmul(operator, ρ)
end

"""
$(SIGNATURES)

Compute the contribution of an [`MPOTerm`](@ref) - and hence also of a
[`TensorProductTerm`](@ref) - given as one tensor per site
in `inds`, to the expectation value ⟨bra|O|ket⟩ / ⟨bra|ket⟩.

Rather than forming the dense operator and tracing it against a reduced density matrix, the
factors are inserted into the patch contraction directly, and the result is divided by the
local norm of the same patch.
"""
function local_expectation_value(inds, bra, operator::MPOTerm, ket, env)
    return contract_local_operator(inds, operator, ket, bra, env) /
        contract_local_norm(inds, ket, bra, env)
end

# Expectation value of a local partition function tensor
# ------------------------------------------------------

"""
    expectation_value(pf::InfinitePartitionFunction, inds => O, env::CTMRGEnv)

Compute the expectation value corresponding to inserting a local tensor(s) `O` at
position `inds` in the partition function `pf` and contracting the whole using a given CTMRG
environment `env`.

Here `inds` can be specified as either a `Tuple{Int,Int}` or a `CartesianIndex{2}`, and `O`
should be a rank-4 tensor conforming to the [`PartitionFunctionTensor`](@ref) indexing
convention.
"""
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction,
        op::Pair{CartesianIndex{2}, <:AbstractTensorMap{T, S, 2, 2}},
        env,
    ) where {T, S}
    return contract_local_tensor(op[1], op[2], env) /
        contract_local_tensor(op[1], pf[op[1]], env)
end
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction, op::Pair{Tuple{Int, Int}}, env
    )
    return expectation_value(pf, CartesianIndex(op[1]) => op[2], env)
end
