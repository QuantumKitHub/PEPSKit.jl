# Approximate finite-window expectation values
# --------------------------------------------

"""
$(SIGNATURES)

Approximately measure the expectation value of an open-boundary `MPOObservable` in a single-layer PEPO using finite boundary MPS/MPO zipup sweeps.
The zipup truncation is controlled by `trunc`, which defaults to `truncrank(χ)` with `χ` the largest CTMRG boundary dimension.
After each zipup step, the result is refined by a single-site DMRG approximation step with `maxiter` sweeps.
Set `maxiter = 0` to disable this refinement.
"""
function expectation_value_approx(
        ρ::InfinitePEPO, observable::MPOObservable, env::CTMRGEnv;
        trunc = _approx_trunc(env), maxiter::Int = 1, direction::Symbol = :auto,
    )
    return _expectation_value_approx(
        ρ, observable, env,
        WindowApprox(Zipup(; trunc), _approx_dmrg(maxiter)), direction
    )
end

"""
Return the row and column ranges enclosing an MPO observable's complete routed path.
"""
function _window_ranges(observable::MPOObservable)
    return _window_ranges(observable.path)
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
