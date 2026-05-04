using Test
using Random
using Accessors
using Zygote
using TensorKit, KrylovKit, PEPSKit
using PEPSKit:
    ctmrg_iteration, compute_gauge_fix_gauge, fix_phases, ScramblingEnvGauge

algs = [
    (:fixed, SimultaneousCTMRG(; projector_alg = :halfinfinite)),
    (:fixed, SimultaneousCTMRG(; projector_alg = :fullinfinite)), # TODO: why are the errors quite a bit larger for :fullinfinite?
]
Dbond, χenv = 2, 16
alg_gauge = ScramblingEnvGauge()
errtol = 1.0e-3

@testset "$iterscheme and $ctm_alg" for (iterscheme, ctm_alg) in algs
    Random.seed!(123521938519)
    state = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond))
    env, = leading_boundary(CTMRGEnv(state, ComplexSpace(χenv)), state, ctm_alg)

    # follow code of _rrule
    if iterscheme == :fixed
        env_conv, info = ctmrg_iteration(InfiniteSquareNetwork(state), env, ctm_alg)
        signs, corner_phases, edge_phases = compute_gauge_fix_gauge(env_conv, env, alg_gauge)

        _, env_vjp = pullback(state, env_conv) do A, x
            e, = ctmrg_iteration(InfiniteSquareNetwork(A), x, ctm_alg)
            return fix_phases(e, signs, corner_phases, edge_phases)
        end
    end

    # get Jacobians of single iteration
    ∂f∂A(x)::typeof(state) = env_vjp(x)[1]
    ∂f∂x(x)::typeof(env) = env_vjp(x)[2]

    # compute real and complex errors
    env_in = CTMRGEnv(state, ComplexSpace(16))
    α_real = randn(Float64)
    α_complex = randn(ComplexF64)

    real_err_∂A = norm(scale(∂f∂A(env_in), α_real) - ∂f∂A(scale(env_in, α_real)))
    real_err_∂x = norm(scale(∂f∂x(env_in), α_real) - ∂f∂x(scale(env_in, α_real)))
    complex_err_∂A = norm(scale(∂f∂A(env_in), α_complex) - ∂f∂A(scale(env_in, α_complex)))
    complex_err_∂x = norm(scale(∂f∂x(env_in), α_complex) - ∂f∂x(scale(env_in, α_complex)))

    @test real_err_∂A < errtol
    @test real_err_∂x < errtol
    @test complex_err_∂A > 1.0e-3
    @test complex_err_∂x > 1.0e-3
end
