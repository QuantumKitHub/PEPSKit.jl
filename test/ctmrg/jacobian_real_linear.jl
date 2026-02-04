using Test
using Random
using Accessors
using Zygote
using TensorKit, KrylovKit, PEPSKit
using PEPSKit:
    ctmrg_iteration, fix_relative_phases, fix_global_phases, ScramblingEnvGauge

algs = [
    (:fixed, SimultaneousCTMRG(; projector_alg = :halfinfinite)),
    (:diffgauge, SequentialCTMRG(; projector_alg = :halfinfinite)),
    (:diffgauge, SimultaneousCTMRG(; projector_alg = :halfinfinite)),
    # TODO: FullInfiniteProjector errors since even real_err_∂A, real_err_∂x are finite?
    # (:fixed, SimultaneousCTMRG(; projector_alg=FullInfiniteProjector)),
    # (:diffgauge, SequentialCTMRG(; projector_alg=FullInfiniteProjector)),
    # (:diffgauge, SimultaneousCTMRG(; projector_alg=FullInfiniteProjector)),
]
Dbond, χenv = 2, 16
alg_gauge = ScramblingEnvGauge()

@testset "$iterscheme and $ctm_alg" for (iterscheme, ctm_alg) in algs
    Random.seed!(123521938519)
    state = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond))
    env, = leading_boundary(CTMRGEnv(state, ComplexSpace(χenv)), state, ctm_alg)

    # follow code of _rrule
    if iterscheme == :fixed
        env_conv, info = ctmrg_iteration(InfiniteSquareNetwork(state), env, ctm_alg)
        env_fixed, signs = gauge_fix(env_conv, alg_gauge, env)
        alg_fixed = gauge_fix(ctm_alg, signs, info)

        _, env_vjp = pullback(state, env_fixed) do A, x
            e, = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)
            return PEPSKit.fix_global_phases(e, x)
        end
    elseif iterscheme == :diffgauge
        _, env_vjp = pullback(state, env) do A, x
            return gauge_fix(ctmrg_iteration(InfiniteSquareNetwork(A), x, ctm_alg)[1], alg_gauge, x)[1]
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

    @test real_err_∂A < 1.0e-9
    @test real_err_∂x < 1.0e-9
    @test complex_err_∂A > 1.0e-3
    @test complex_err_∂x > 1.0e-3
end
