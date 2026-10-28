# Expectation value of a LocalOperator
# ------------------------------------

"""
    expectation_value(state, O::LocalOperator, env, alg = ExactContract())
    expectation_value(bra, O::LocalOperator, ket, env, alg = ExactContract())

Compute the expectation value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ or tr(O * state) / tr(state) of a [`LocalOperator`](@ref) `O`.
This can be done either for a PEPS, or alternatively for a density matrix PEPO.
In the latter case the first signature corresponds to a single layer PEPO contraction, while the second signature yields a bilayer contraction instead.
The default [`ExactContract`](@ref) contracts each patch exactly.
For single-layer PEPOs with a CTMRG environment, use [`BoundaryMPSContract`](@ref) to select approximate boundary-MPS contractions.
"""
function MPSKit.expectation_value(
        bra::S, O::LocalOperator, ket::S, env, alg::ExactContract = ExactContract(),
    ) where {S <: InfiniteState}
    return _expectation_value(bra, O, env, alg, ket)
end
MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, env, alg::ExactContract = ExactContract()) =
    expectation_value(peps, O, peps, env, alg)
MPSKit.expectation_value(state::InfinitePEPO, O::LocalOperator, env, alg::ContractionAlgorithm = ExactContract()) =
    _expectation_value(state, O, env, alg)

"""
Validate the lattice, prepare the contraction algorithm, and sum independently normalized local terms.
"""
function _expectation_value(state::InfiniteState, O::LocalOperator, env, alg::ContractionAlgorithm, ket::InfiniteState...)
    checklattice(state, O, ket...)
    alg = _prepare_contract(state, env, alg)
    isempty(O.terms) && return zero(promote_type(scalartype(state), scalartype(env)))
    term_vals = dtmap(collect(O.terms)) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        return local_expectation_value(inds, state, operator, ket..., env, alg)
    end
    return sum(term_vals)
end

"""
Prepare environment-dependent contraction settings, leaving exact contraction unchanged.
"""
_prepare_contract(state::InfiniteState, env, alg::ExactContract) = alg


# Expectation value of an individual local term
# ---------------------------------------------

"""
    local_expectation_value(inds, bra, operator, ket, env, alg = ExactContract())
    local_expectation_value(inds, state, operator, env, alg = ExactContract())

Compute the contribution of a single term of a [`LocalOperator`](@ref) to the expectation
value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ or tr(O * state) / tr(state), where `operator` is the local
term acting on the sites `inds`.

Dispatch on the operator and contraction algorithm selects how each term is evaluated.
"""
function local_expectation_value end

# AbstractTensorMap evaluation goes through reduced density matrix
function local_expectation_value(
        inds, bra::S, operator::AbstractTensorMap, ket::S, env, ::ExactContract = ExactContract(),
    ) where {S <: InfiniteState}
    ρ = reduced_densitymatrix(inds, ket, bra, env)
    return trmul(operator, ρ)
end
function local_expectation_value(
        inds, state::InfinitePEPO, operator::AbstractTensorMap, env, ::ExactContract = ExactContract(),
    )
    ρ = reduced_densitymatrix(inds, state, env)
    return trmul(operator, ρ)
end

"""
$(SIGNATURES)

Compute the contribution of an [`MPOTerm`](@ref), including a [`TensorProductTerm`](@ref), given as one tensor per site in `inds`.
For a PEPS or purified PEPO, compute ``⟨bra|O|ket⟩ / ⟨bra|ket⟩``; for a single-layer density matrix PEPO, compute ``tr(O * state) / tr(state)``.
Insert the factors into the patch contraction directly and divide by the local norm of the same patch.
"""
function local_expectation_value(
        inds, bra::S, operator::MPOTerm, ket::S, env, ::ExactContract = ExactContract(),
    ) where {S <: InfiniteState}
    return contract_local_operator(inds, operator, ket, bra, env) /
        contract_local_norm(inds, ket, bra, env)
end
function local_expectation_value(
        inds, state::InfinitePEPO, operator::MPOTerm, env, ::ExactContract = ExactContract(),
    )
    return contract_local_operator(inds, operator, state, env) /
        contract_local_norm(inds, state, env)
end

# TODO: Implement boundary_contraction_expr for BPEnv to contract its messages directly.
local_expectation_value(inds, bra::S, operator::MPOTerm, ket::S, env::BPEnv, alg::ExactContract = ExactContract()) where {S <: InfiniteState} =
    local_expectation_value(inds, bra, operator, ket, CTMRGEnv(env), alg)

# Expectation value of a local partition function tensor
# ------------------------------------------------------

"""
    expectation_value(pf::InfinitePartitionFunction, inds => O, env::CTMRGEnv, alg = ExactContract())

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
        env, ::ExactContract = ExactContract(),
    ) where {T, S}
    return contract_local_tensor(op[1], op[2], env) /
        contract_local_tensor(op[1], pf[op[1]], env)
end
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction, op::Pair{Tuple{Int, Int}}, env, alg::ExactContract = ExactContract(),
    )
    return expectation_value(pf, CartesianIndex(op[1]) => op[2], env, alg)
end
