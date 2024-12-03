using Test
using Random
using Zygote
using TensorKit, KrylovKit, PEPSKit
using PEPSKit: ctmrg_iter, gauge_fix, fix_relative_phases, fix_global_phases

iterschemes = [:diffgauge, :fixed]
ctm_alg = CTMRG()
state = InfinitePEPS(2, 2)
envs = leading_boundary(CTMRGEnv(state, ComplexSpace(16)), state, ctm_alg)
@testset "$iterscheme" for iterscheme in iterschemes
    # follow code of _rrule
    if iterscheme == :fixed
        envsconv, info = ctmrg_iter(state, envs, ctm_alg)
        envsfix, signs = gauge_fix(envs, envsconv)
        Ufix, Vfix = fix_relative_phases(info.U, info.V, signs)
        svd_alg_fixed = SVDAdjoint(;
            fwd_alg=PEPSKit.FixedSVD(Ufix, info.S, Vfix),
            rrule_alg=ctm_alg.projector_alg.svd_alg.rrule_alg,
        )
        alg_fixed = CTMRG(;
            svd_alg=svd_alg_fixed, trscheme=notrunc(), ctmrgscheme=:simultaneous
        )

        _, env_vjp = pullback(state, envsfix) do A, x
            e, = PEPSKit.ctmrg_iter(A, x, alg_fixed)
            return PEPSKit.fix_global_phases(x, e)
        end
    elseif iterscheme == :diffgauge
        _, env_vjp = pullback(state, envs) do A, x
            return gauge_fix(x, ctmrg_iter(A, x, ctm_alg)[1])[1]
        end
    end

    # get Jacobians of single iteration
    ∂f∂A(x)::typeof(state) = env_vjp(x)[1]
    ∂f∂x(x)::typeof(envs) = env_vjp(x)[2]

    # compute real and complex errors
    env_in = CTMRGEnv(state, ComplexSpace(16))
    α_real = randn(Float64)
    α_complex = randn(ComplexF64)

    real_err_∂A = norm(scale(∂f∂A(env_in), α_real) - ∂f∂A(scale(env_in, α_real)))
    real_err_∂x = norm(scale(∂f∂x(env_in), α_real) - ∂f∂x(scale(env_in, α_real)))
    complex_err_∂A = norm(scale(∂f∂A(env_in), α_complex) - ∂f∂A(scale(env_in, α_complex)))
    complex_err_∂x = norm(scale(∂f∂x(env_in), α_complex) - ∂f∂x(scale(env_in, α_complex)))

    @test real_err_∂A < 1e-10
    @test real_err_∂x < 1e-10
    @test complex_err_∂A > 1e-3
    @test complex_err_∂x > 1e-3
end
