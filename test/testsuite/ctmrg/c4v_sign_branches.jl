using Test
using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using Adapt
using PEPSKit: ctmrg_iteration, compute_gauge_fix_gauge, fix_relative_phases, ScramblingEnvGaugeC4v

# For half-integer U(1) charges, (-1)^{2Q} on the ket bonds negates the sandwich, so fixed
# points come in pairs, one of which flips (C, E) to (-C, -E) every iteration. Both should converge.

sd = 20260924
H = heisenberg_XXZ(ComplexF64, U1Irrep, InfiniteSquare(); spin = 1 // 2)
V = U1Space(0 => 1, 1 // 2 => 1, -1 // 2 => 1)
Venv = U1Space(0 => 2, 1 // 2 => 2, -1 // 2 => 2, 1 => 1, -1 => 1)
tol = 1.0e-10
maxiter = 400

# (-1)^{2Q} on a U(1) space
function charge_parity(W)
    X = id(W)
    for (c, b) in blocks(X)
        b .*= (-1)^Int(2 * c.charge)
    end
    return X
end

# sign by which one gauge-fixed iteration multiplies the edge
function iteration_sign(network, env, alg)
    env′, = ctmrg_iteration(network, env, alg)
    signs, = compute_gauge_fix_gauge(env′, env, ScramblingEnvGaugeC4v())
    env′ = fix_relative_phases(env′, signs)
    return sign(real(dot(env.edges[1], env′.edges[1])))
end

function sign_twin(env)
    E = env.edges[1]
    g = charge_parity(space(E, 2))
    @tensor Ê[χ_out D_ket D_bra; χ_in] := E[χ_out D_ket′ D_bra; χ_in] * g[D_ket; D_ket′]
    return CTMRGEnv(env.corners[1], Ê)
end

function ctmrg_c4v_sign_branches(AT)
    return @testset "C4v CTMRG converges on both sign branches of a half-integer U(1) state ($AT)" begin
        Random.seed!(sd)
        peps = InfinitePEPS(randn, Float64, physicalspace(H)[1, 1], V)
        symmetrize!(peps, RotateReflect())
        peps = adapt(AT, InfinitePEPS(map(a -> a / norm(a), peps.A)))
        network = InfiniteSquareNetwork(peps)
        alg = C4vCTMRG(; tol, maxiter, projector_alg = :C4vEighProjector)

        env, info = leading_boundary(initialize_random_c4v_env(peps, Venv), peps, alg)
        @test info.converged

        env_twin = sign_twin(env)
        @test iteration_sign(network, env_twin, alg) == -iteration_sign(network, env, alg)
        env_twin, info_twin = leading_boundary(env_twin, peps, alg)
        @test info_twin.converged
        @test expectation_value(peps, H, env_twin) ≈ expectation_value(peps, H, env) atol = 1.0e-8
    end
end
