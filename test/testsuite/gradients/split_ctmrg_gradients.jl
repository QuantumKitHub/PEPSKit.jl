using Test, Random, LinearAlgebra
using TensorKit, PEPSKit, Zygote, Adapt
using ChainRulesCore: ProjectTo
using PEPSKit: ctmrg_iteration, hook_pullback

"""Check public environment gradients against an analytic corner derivative."""
function _split_environment_gradient(env::SplitCTMRGEnv)
    """Read one corner so all other environment derivatives vanish."""
    loss(x) = real(dot(env.corners[1], x.corners[1]))
    gradient = only(Zygote.withgradient(loss, env).grad)
    @test gradient isa typeof(env)
    @test only(Zygote.gradient(loss, env)) isa typeof(env)
    @test gradient.corners[1] ≈ env.corners[1]
    @test all(C -> iszero(norm(C)), gradient.corners[2:end])
    @test iszero(norm(gradient.edges_ket)) && iszero(norm(gradient.edges_bra))
    return nothing
end

function gradients_split_gauge(AT)
    return @testset "Split outer and interlayer gauge ($AT)" begin
        Random.seed!(32591)
        psi = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell = (2, 3)))
        env = SplitCTMRGEnv(psi, ComplexSpace(2); interlayer_space = ComplexSpace(3))
        pairs = map(PEPSKit.eachcoordinate(env, 1:4)) do (dir, r, c)
            firstlayer = env.ket_first[dir] ? :ket : :bra
            factor = edge(env, firstlayer, dir, r, c)
            outer, interlayer = codomain(factor)[1], domain(factor)[1]
            return first(left_orth(adapt(AT, randn(ComplexF64, outer ← outer)))),
                first(left_orth(adapt(AT, randn(ComplexF64, interlayer ← interlayer))))
        end
        signs = (; outer = map(first, pairs), interlayer = map(last, pairs))
        transformed = PEPSKit.fix_relative_phases(env, signs)
        restored = gauge_fix(transformed, env, PEPSKit.ScramblingEnvGauge())
        @test norm(restored - env) / norm(env) < 1.0e-10
        _split_environment_gradient(env)
        S = Vect[FermionParity](0 => 1, 1 => 1)
        fermionic = adapt(AT, InfinitePEPS(S, S; unitcell = (2, 3)))
        _split_environment_gradient(SplitCTMRGEnv(fermionic, S))
    end
end

"""Prepare a small entangled PEPS and split environment for gradient checks."""
function _split_gradient_inputs(AT)
    Random.seed!(84109)
    psi = adapt(
        AT, product_peps(
            randn, ComplexF64, ComplexSpace(2), ComplexSpace(2);
            noise_amp = 0.2, state_vector = fill([1.0, 0.0], (1, 1))
        )
    )
    env = SplitCTMRGEnv(psi, ComplexSpace(2))
    H = adapt(AT, heisenberg_XYZ(InfiniteSquare()))
    alg = SplitCTMRG(; verbosity = 0, tol = 1.0e-10, maxiter = 100)
    return psi, env, H, alg
end

function gradients_split_sweep(AT)
    return @testset "Split sweep pullback ($AT)" begin
        psi, env, H, alg = _split_gradient_inputs(AT)
        loss(A, x) = real(expectation_value(A, H, ctmrg_iteration(InfiniteSquareNetwork(A), x, alg)[1]))
        _, back = Zygote.pullback(loss, psi, env)
        dpsi, denv = back(1.0)
        denv = ProjectTo(env)(denv)
        direction = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(2)))
        direction_env = SplitCTMRGEnv(psi, ComplexSpace(2))
        epsilon = 1.0e-6
        finite_state = (loss(psi + epsilon * direction, env) - loss(psi - epsilon * direction, env)) / (2 * epsilon)
        finite_env = (loss(psi, env + epsilon * direction_env) - loss(psi, env - epsilon * direction_env)) / (2 * epsilon)
        @test real(dot(dpsi, direction)) ≈ finite_state atol = 1.0e-5 rtol = 1.0e-4
        @test real(inner(denv, direction_env)) ≈ finite_env atol = 1.0e-5 rtol = 1.0e-4
    end
end

function gradients_split_fixedpoint(AT)
    return @testset "Split fixed-point energy gradient ($AT)" begin
        psi, initial, H, alg = _split_gradient_inputs(AT)
        env, info = leading_boundary(initial, psi, alg)
        @test info.converged
        next, = ctmrg_iteration(InfiniteSquareNetwork(psi), env, alg)
        aligned = gauge_fix(next, env, PEPSKit.ScramblingEnvGauge())
        @test norm(aligned - env) / norm(env) < 1.0e-7
        gradalg = FixedPointGradient(; solver_alg = (; alg = :GMRES, tol = 1.0e-8, maxiter = 100))
        E, back = Zygote.pullback(psi) do A
            boundary, = hook_pullback(leading_boundary, env, A, alg; alg_rrule = gradalg)
            return cost_function(A, boundary, H)
        end
        gs = back(1.0)
        @test only(back(2.0)) ≈ 2 * only(gs) atol = 1.0e-6 rtol = 1.0e-5
        @test only(back(1.0)) ≈ only(gs) atol = 1.0e-6 rtol = 1.0e-5
        direction = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(2)))
        epsilon = 1.0e-5
        energy(A) = cost_function(A, leading_boundary(env, A, alg)[1], H)
        finite = (energy(psi + epsilon * direction) - energy(psi - epsilon * direction)) / (2 * epsilon)
        @test real(dot(only(gs), direction)) ≈ finite atol = 1.0e-4 rtol = 1.0e-3
        optalg = PEPSOptimize(;
            boundary_alg = alg,
            optimizer_alg = (; maxiter = 2, verbosity = 0)
        )
        optimized_psi, final, optimized, = fixedpoint(H, psi, deepcopy(env), optalg)
        @test optimized < E
        @test cost_function(optimized_psi, final, H) ≈ optimized
        next, = ctmrg_iteration(InfiniteSquareNetwork(optimized_psi), final, alg)
        aligned = gauge_fix(next, final, PEPSKit.ScramblingEnvGauge())
        @test norm(aligned - final) / norm(final) < 1.0e-7
    end
end
