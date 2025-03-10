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
    maxiter::Int = 20
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
function compute_gradient(operator, peps::InfinitePEPS, alg::FixedPointGradient)
    local info
    env′ = CTMRGEnv(ones, ComplexF64, peps, oneunit(spacetype(peps.A[1])))
    env′, = leading_boundary(env′, peps, alg.boundary_alg)
    E, gs = withgradient(peps) do ψ
        env′, info = hook_pullback(
            leading_boundary, env′, ψ, alg.boundary_alg; alg_rrule=alg.gradient_alg
        )
        return cost_function(ψ, env′, operator)
    end
    return E, only(gs), info  # Return info along with energy and gradient
end

function compute_gradient(operator, peps::InfinitePEPS, alg::SimpleGradient)
    local info
    env′ = CTMRGEnv(ones, ComplexF64, peps, oneunit(spacetype(peps.A[1])))
    env′, = leading_boundary(env′, peps, alg.boundary_alg)
    E, gs = withgradient(peps) do ψ
        for _ in 1:(alg.maxiter)
            env′, info = ctmrg_iteration(InfiniteSquareNetwork(ψ), env′, alg.boundary_alg)
        end
        return cost_function(ψ, env′, operator)
    end
    return E, only(gs), info  # Return info along with energy and gradient
end
