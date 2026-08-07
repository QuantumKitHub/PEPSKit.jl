abstract type PreconditionAlgorithm end

const PRECONDITION_ALGORITHM_SYMBOLS = IdDict{Symbol, Type{<:PreconditionAlgorithm}}()

"""
    PreconditionAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `PreconditionAlgorithm` algorithm struct.
"""
function PreconditionAlgorithm(;
        alg = Defaults.precondition_alg,
        tol = Defaults.precondition_tol,
        maxiter = Defaults.precondition_maxiter,
        verbosity = Defaults.precondition_verbosity,
        krylovdim = Defaults.precondition_krylovdim,
        regularization = Defaults.precondition_regularization,
    )
    # replace symbol with PreconditionAlgorithm alg type
    haskey(PRECONDITION_ALGORITHM_SYMBOLS, alg) ||
        throw(ArgumentError("unknown PreconditionAlgorithm algorithm: $alg"))
    alg_type = PRECONDITION_ALGORITHM_SYMBOLS[alg]

    return alg_type(GMRES(; tol, maxiter, verbosity, krylovdim), regularization)
end

"""
$(TYPEDEF)

Preconditioner for PEPS ground-state optimization based on the leading term of the PEPS
metric tensor, i.e. the local norm matrix ``N`` obtained by contracting the full CTMRG
environment around a given unit-cell site while leaving the ket and bra PEPS tensor at that
site open.

Instead of following the raw Euclidean gradient ``g``, the optimizer is fed the solution
``g̃`` of the regularized local linear problem
```math
( N / ⟨ψ|ψ⟩ + δ ) g̃ = g
```
which is solved separately for every tensor in the unit cell using an iterative solver.
The regularization ``δ`` prevents the (generally singular) metric from being inverted
exactly, and decays during the optimization according to
```math
δ = \\texttt{regularization} × ‖∇f‖^2 / \\texttt{iter}^2
```
so that the preconditioner acts conservatively far from the minimum and becomes
increasingly aggressive as the optimization converges.

## Fields

$(TYPEDFIELDS)

## Constructors

    LocalPreconditioner(; kwargs...)

Construct a local preconditioner algorithm struct based on keyword arguments. The supported
keywords are:

* `tol::Real=$(Defaults.precondition_tol)` : Convergence tolerance of the local linear problem.
* `maxiter::Int=$(Defaults.precondition_maxiter)` : Maximal number of iterations of the local linear problem. Note that the default performs a single iteration, such that the metric is only inverted approximately.
* `verbosity::Int=$(Defaults.precondition_verbosity)` : Preconditioner output verbosity, ≤0 by default to disable too verbose printing.
* `krylovdim::Int=$(Defaults.precondition_krylovdim)` : Krylov dimension of the local linear problem.
* `regularization::Real=$(Defaults.precondition_regularization)` : Prefactor setting the regularization strength ``δ``, see above.

Reference: [Phys. Rev. B 113, 125111](https://doi.org/10.1103/h396-yc28), [arXiv:2511.09546](https://arxiv.org/abs/2511.09546)
"""
struct LocalPreconditioner{A} <: PreconditionAlgorithm
    "solver algorithm used for the local linear problem"
    solver_alg::A

    "prefactor setting the regularization strength of the local linear problem"
    regularization::Float64
end
LocalPreconditioner(; kwargs...) = PreconditionAlgorithm(; alg = :LocalPreconditioner, kwargs...)
PRECONDITION_ALGORITHM_SYMBOLS[:LocalPreconditioner] = LocalPreconditioner

# `LocalPreconditioner` has no top-level `tol` field (it lives on `solver_alg`), so the
# default `MPSKit.DynamicTols._updatetol` (which sets `alg.tol`) doesn't apply
_updatetol(alg::LocalPreconditioner, tol::Real) = @set alg.solver_alg.tol = tol

"""
$(SIGNATURES)

Apply the regularized local metric ``N / ⟨ψ|ψ⟩ + δ`` to a single PEPS tensor `g_rc`, where
``N`` is obtained by contracting the CTMRG environment `env` around the unit-cell site
`(r, c)` and `norm_pref` is the corresponding local norm ``⟨ψ|ψ⟩``.
"""
function apply_local_preconditioner(g_rc::PEPSTensor, env::CTMRGEnv, δ, (r, c), norm_pref)
    @autoopt @tensor g_rc_prec[d; D_N_below D_E_below D_S_below D_W_below] :=
        g_rc[d; D_N_above D_E_above D_S_above D_W_above] *
        corner(env, NORTHWEST, r - 1, c - 1)[χ_WNW; χ_NNW] *
        edge(env, NORTH, r - 1, c)[χ_NNW D_N_above D_N_below; χ_NNE] *
        corner(env, NORTHEAST, r - 1, c + 1)[χ_NNE; χ_ENE] *
        edge(env, EAST, r, c + 1)[χ_ENE D_E_above D_E_below; χ_ESE] *
        corner(env, SOUTHEAST, r + 1, c + 1)[χ_ESE; χ_SSE] *
        edge(env, SOUTH, r + 1, c)[χ_SSE D_S_above D_S_below; χ_SSW] *
        corner(env, SOUTHWEST, r + 1, c - 1)[χ_SSW; χ_WSW] *
        edge(env, WEST, r, c - 1)[χ_WSW D_W_above D_W_below; χ_WNW]
    # TODO: figure out if this needs additional twists?
    return g_rc_prec / norm_pref + δ * g_rc
end

"""
    peps_precondition(x, g, tol_state::Base.RefValue, alg)

Precondition the PEPS gradient `g` at the point `x = (peps, env)` according to the
preconditioner algorithm `alg`. Passing `alg = nothing` disables preconditioning and returns
`g` unchanged. See [`LocalPreconditioner`](@ref) for details on the local metric
preconditioner, and note that its regularization strength is set based on the current
gradient norm and iteration count tracked in `tol_state`.
"""
peps_precondition(x, g, tol_state::Base.RefValue, alg::Nothing) = g

function peps_precondition(x, g, tol_state::Base.RefValue, alg::LocalPreconditioner)
    peps, env = x
    δ = alg.regularization * tol_state[].gradnorm^2 / max(tol_state[].iter, 1)
    g_prec_unitcell = map(eachcoordinate(g)) do (r, c)
        nf = _contract_site((r, c), InfiniteSquareNetwork(peps), env)
        g_rc_prec, = linsolve(g[r, c], g[r, c], alg.solver_alg) do g_in
            return apply_local_preconditioner(g_in, env, δ, (r, c), nf)
        end
        return g_rc_prec
    end
    return InfinitePEPS(g_prec_unitcell)
end
