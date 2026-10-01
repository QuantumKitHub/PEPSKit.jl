# Approximate finite-window expectation values
# --------------------------------------------

"""
$(SIGNATURES)

Approximately measure a `LocalOperator` in a single-layer PEPO using finite boundary MPS zipup sweeps.
Each term is normalized in its own enclosing window, and the resulting contributions are summed.
Dense terms are reordered into a column-wise snake and decomposed into MPOs without truncation.
Explicit [`MPOTerm`](@ref) factors retain their given order and are connected by non-self-intersecting nearest-neighbor paths.
Routing tries horizontal-first and then vertical-first shortest connections; more complicated explicit MPO routes raise a not-implemented error.
Single-site terms are evaluated exactly, and an empty operator returns zero.

- By default, `direction = :auto` selects north-to-south sweeps for wide and square windows, and east-to-west for tall windows, separately for each term.
    Specify `direction = :rows` or `:columns` to override this choice.
- The zipup truncation is controlled by `trunc`, which defaults to `truncrank(χ)` with `χ` the largest CTMRG boundary dimension.
    This keyword does not truncate the operator decomposition.
- After each zipup step, the result is refined by a single-site DMRG approximation step with `maxiter` sweeps.
    Set `maxiter = 0` to disable this refinement.
"""
function expectation_value_approx(
        ρ::InfinitePEPO, O::LocalOperator, env::CTMRGEnv;
        trunc = _approx_trunc(env), maxiter::Int = 1, direction::Symbol = :auto,
    )
    _check_window_inputs(ρ, direction)
    checklattice(ρ, O)
    alg = WindowApprox(Zipup(; trunc), _approx_dmrg(maxiter))
    isempty(O.terms) && return zero(promote_type(scalartype(ρ), scalartype(env)))
    term_vals = map(collect(O.terms)) do (sites, term)
        return _local_expectation_value_approx(sites, ρ, term, env, alg, direction)
    end
    return sum(term_vals)
end

"""
Evaluate a dense or MPO term, using exact single-site contractions and routing larger terms.
"""
function _local_expectation_value_approx(
        sites::Vector{CartesianIndex{2}}, ρ::InfinitePEPO,
        term::Union{AbstractTensorMap, MPOTerm}, env::CTMRGEnv,
        alg::WindowApprox, direction::Symbol,
    )
    if _local_term_iszero(term)
        return zero(promote_type(scalartype(ρ), scalartype(env), _local_term_scalartype(term)))
    end
    if length(sites) == 1
        op = term isa AbstractTensorMap ? term : only(term)
        return local_expectation_value(sites, ρ, op, env)
    end
    routed = _route_mpo_term(sites, term, physicalspace(ρ))
    return _expectation_value_approx(ρ, routed, env, alg, direction)
end

"""
Return the smallest row and column ranges containing a collection of lattice sites.
"""
function _window_ranges(sites)
    rows = getindex.(sites, 1)
    cols = getindex.(sites, 2)
    return UnitRange(extrema(rows)...), UnitRange(extrema(cols)...)
end

"""
Return the largest CTMRG boundary-space dimension appearing in the corner tensors.
"""
function _ctmrg_boundary_chi(env::CTMRGEnv)
    χ = 0
    for C in env.corners
        χ = max(χ, dim(space(C, 1)), dim(space(C, 2)))
    end
    return χ
end

"""
Construct the default rank truncation from the largest CTMRG boundary dimension.
"""
_approx_trunc(env::CTMRGEnv) = truncrank(_ctmrg_boundary_chi(env))

"""
Construct the optional one-site DMRG refinement, or disable refinement for zero iterations.
"""
function _approx_dmrg(maxiter::Int)
    maxiter >= 0 || throw(ArgumentError("maxiter should be nonnegative"))
    return iszero(maxiter) ? nothing : DMRG(; maxiter, verbosity = 0)
end
