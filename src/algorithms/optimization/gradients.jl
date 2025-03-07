"""
    GradientAlgorithm

Abstract super type for gradient algorithms.
"""
abstract type GradientAlgorithm end

"""
    struct SimpleGradient(; svd_alg=Defaults.svd_alg,
                                      trscheme=Defaults.trscheme, verbosity=0)

Projector algorithm implementing projectors from SVDing the half-infinite CTMRG environment.
"""
@kwdef struct SimpleGradient <: GradientAlgorithm
    boundary_alg::CTMRGAlgorithm = Defaults.boundary_alg
    maxiter::Int = 10
end

"""
    struct FixedPointGradient(; svd_alg=Defaults.svd_alg,
                                      trscheme=Defaults.trscheme, verbosity=0)

Projector algorithm implementing projectors from SVDing the full 4x4 CTMRG environment.
"""
@kwdef struct FixedPointGradient{G} <: GradientAlgorithm
    boundary_alg::CTMRGAlgorithm = Defaults.boundary_alg
    gradient_alg::G = Defaults.gradient_alg
end

"""
    compute_gradient(
        operator, peps::InfinitePEPS, env::CTMRGEnv, alg::FixedPointGradient
        )
    Compute gradient with respect to PEPS for optimization
"""
function compute_gradient(
    operator, peps::InfinitePEPS, env::CTMRGEnv, alg::FixedPointGradient
)
    local info
    E, gs = withgradient(peps) do ψ
        env′, info = hook_pullback(
            leading_boundary, env, ψ, alg.boundary_alg; alg_rrule=alg.gradient_alg
        )
        return cost_function(ψ, env′, operator)
    end
    return E, only(gs), env′, info  # Return info along with energy and gradient
end

function compute_gradient(operator, peps::InfinitePEPS, env::CTMRGEnv, alg::SimpleGradient)
    local info
    env′ = deepcopy(env)
    E, gs = withgradient(peps) do ψ
        for _ in 1:(alg.maxiter)
            env′, info = ctmrg_iteration(InfiniteSquareNetwork(ψ), env′, alg.boundary_alg)
        end
        return cost_function(ψ, env′, operator)
    end
    return E, only(gs), env′, info  # Return info along with energy and gradient
end
