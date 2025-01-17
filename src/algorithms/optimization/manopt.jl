using Printf
using Manifolds: Manifolds
using Manifolds:
    AbstractManifold,
    AbstractRetractionMethod,
    Euclidean,
    default_retraction_method,
    retract
using Manopt

"""
    mutable struct RecordTruncationError <: RecordAction

Record the maximal truncation error of all `boundary_alg` runs of the corresponding
optimization iteration.
"""
mutable struct RecordTruncationError <: RecordAction
    recorded_values::Vector{Float64}
    RecordTruncationError() = new(Vector{Float64}())
end
function (r::RecordTruncationError)(
    p::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int
)
    cache = Manopt.get_cost_function(get_objective(p))
    return Manopt.record_or_reset!(r, cache.env_info.truncation_error, i)
end

"""
    mutable struct RecordConditionNumber <: RecordAction

Record the maximal condition number of all `boundary_alg` runs of the corresponding
optimization iteration.
"""
mutable struct RecordConditionNumber <: RecordAction
    recorded_values::Vector{Float64}
    RecordConditionNumber() = new(Vector{Float64}())
end
function (r::RecordConditionNumber)(
    p::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int
)
    cache = Manopt.get_cost_function(get_objective(p))
    return Manopt.record_or_reset!(r, cache.env_info.condition_number, i)
end

"""
    mutable struct RecordUnitCellGradientNorm <: RecordAction
        
Record the PEPS gradient norms unit cell entry-wise, i.e. an array 
of norms `norm.(peps.A)`.
"""
mutable struct RecordUnitCellGradientNorm <: RecordAction
    recorded_values::Vector{Matrix{Float64}}
    RecordUnitCellGradientNorm() = new(Vector{Matrix{Float64}}())
end
function (r::RecordUnitCellGradientNorm)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    cache = Manopt.get_cost_function(get_objective(p))
    grad = cache.from_vec(get_gradient(s))
    return Manopt.record_or_reset!(r, norm.(grad.A), i)
end

"""
    mutable struct DebugPEPSOptimize <: DebugAction
    
Custom `DebugAction` printing for PEPS optimization runs.

The debug info is output using `@info` and by default prints the optimization iteration,
the cost function value, the gradient norm, the last step size and the time elapsed during
the current iteration.
"""
mutable struct DebugPEPSOptimize <: DebugAction
    last_time::UInt64
    DebugPEPSOptimize() = new(time_ns())
end
function (d::DebugPEPSOptimize)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    time_new = time_ns()
    cost = get_cost(p, get_iterate(s))
    if k == 0
        @info @sprintf("Initial f(x) = %.8f", cost)
    elseif k > 0
        gradient_norm = norm(get_manifold(p), get_iterate(s), get_gradient(s))
        stepsize = get_last_stepsize(p, s, k)
        time = (time_new - d.last_time) * 1e-9
        @info @sprintf(
            "Optimization %d: f(x) = %.8f   ‖∂f‖ = %.8f   step = %.4f   time = %.2f sec",
            k,
            cost,
            gradient_norm,
            stepsize,
            time
        )
    end
    d.last_time = time_new  # update time stamp
    return nothing
end

"""
    SymmetrizeExponentialRetraction <: AbstractRetractionMethod
    
Exponential retraction followed by a symmetrization step.
"""
struct SymmetrizeExponentialRetraction <: AbstractRetractionMethod
    symmetrization::SymmetrizationStyle
    from_vec::Function
end

function Manifolds.retract!(
    M::Euclidean, p, q, X, t::Number, sr::SymmetrizeExponentialRetraction
)
    v = Manifolds.retract!(M, p, q, X, t)
    v_symm_peps = symmetrize!(sr.from_vec(v), sr.symmetrization)
    return to_vec(v_symm_peps)
end
