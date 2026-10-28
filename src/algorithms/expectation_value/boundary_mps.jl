# Boundary-MPS contraction of individual local terms
# --------------------------------------------------

"""
Validate a single-layer PEPO and resolve the default zipup truncation from its CTMRG environment.
"""
function _prepare_contract(ρ::InfinitePEPO, env::CTMRGEnv, alg::BoundaryMPSContract)
    size(ρ, 3) == 1 || throw(DimensionMismatch("only single-layer PEPO contractions are supported"))
    isnothing(alg.zipup) || return alg
    return BoundaryMPSContract(Zipup(; trunc = _default_truncation(env, alg)), alg.dmrg; direction = alg.direction)
end

"""
Evaluate a dense or MPO term, using exact single-site contractions and routing larger terms.
"""
function local_expectation_value(
        sites::Vector{CartesianIndex{2}}, ρ::InfinitePEPO,
        term::Union{AbstractTensorMap, MPOTerm}, env::CTMRGEnv,
        alg::BoundaryMPSContract,
    )
    if _local_term_iszero(term)
        return zero(promote_type(scalartype(ρ), scalartype(env), scalartype(term)))
    end
    if length(sites) == 1
        return local_expectation_value(sites, ρ, term, env, ExactContract())
    end
    routed = _route_mpo_term(sites, term, physicalspace(ρ))
    return _expectation_value(ρ, routed, env, alg)
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
_default_truncation(env::CTMRGEnv, ::BoundaryMPSContract) = truncrank(_ctmrg_boundary_chi(env))
