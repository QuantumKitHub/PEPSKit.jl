using Test
using Random
using PEPSKit
using TensorKit
using Adapt
using PEPSKit: EnlargedCorner, c4v_enlarge, c4v_projector!
using PEPSKit: _contract_site, _contract_corners, _contract_vertical_edges,
    _contract_horizontal_edges
using PEPSKit: NORTH, EAST, SOUTH, WEST, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST

stype = ComplexF64

# Unpack a C4v environment into the equivalent generic CTMRGEnv to use as a reference
function _expand_c4v_env(env::C4vCTMRGEnv)
    C = convert(TensorMap, only(env.corners))
    return CTMRGEnv(fill(C, 4, 1, 1), copy(env.edges))
end

function _random_c4v_peps(AT, P, V)
    peps = InfinitePEPS(randn, stype, P, V)
    BraidingStyle(sectortype(P)) isa Bosonic && symmetrize!(peps, RotateReflect())
    return adapt(AT, InfinitePEPS(map(a -> a / norm(a), peps.A)))
end

function _random_local_operator(P)
    O₁ = randn(stype, P ← P)
    O₂ = randn(stype, P ⊗ P ← P ⊗ P)
    return LocalOperator(
        fill(P, 1, 1),
        (CartesianIndex(1, 1),) => O₁,
        (CartesianIndex(1, 1), CartesianIndex(1, 2)) => O₂,
        (CartesianIndex(1, 1), CartesianIndex(2, 1)) => O₂,
    )
end

function test_c4v_contractions(n::InfiniteSquareNetwork, env::C4vCTMRGEnv)
    env_ref = _expand_c4v_env(env)
    ind = (1, 1)

    @testset "Network value contractions" begin
        @test _contract_site(ind, n, env) ≈ _contract_site(ind, n, env_ref)
        @test _contract_corners(ind, env) ≈ _contract_corners(ind, env_ref)
        @test _contract_vertical_edges(ind, env) ≈ _contract_vertical_edges(ind, env_ref)
        @test _contract_horizontal_edges(ind, env) ≈ _contract_horizontal_edges(ind, env_ref)
        @test network_value(n, env) ≈ network_value(n, env_ref)
        @test_throws AssertionError _contract_site((1, 2), n, env)
    end

    @testset "Enlarged corners" begin
        for dir in (NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST)
            Q = EnlargedCorner(n, env, (dir, 1, 1))
            Q_ref = EnlargedCorner(n, env_ref, (dir, 1, 1))
            @test TensorMap(Q) ≈ TensorMap(Q_ref)
        end
        for projector_alg in (C4vEighProjector(), C4vQRProjector())
            @test c4v_enlarge(n, env, projector_alg) ≈ c4v_enlarge(n, env_ref, projector_alg)
        end
    end

    @testset "Rotations" begin
        nv = network_value(n, env)
        for rot in (rotl90, rotr90, rot180)
            env_rot = rot(env)
            env_ref_rot = rot(env_ref)
            @test env_rot isa C4vCTMRGEnv
            @test size(env_rot.edges) == (4, 1, 1)
            @test all(env_rot.edges .≈ env_ref_rot.edges)
            @test network_value(rot(n), env_rot) ≈ nv
        end
        @test rotl90(rotr90(env)) ≈ env
        @test rot180(rot180(env)) ≈ env
    end

    @testset "VectorInterface" begin
        α, β = randn(stype), randn(stype)
        @test norm(env)^2 ≈ norm(only(env.corners))^2 + sum(e -> norm(e)^2, env.edges)
        @test inner(env, env) ≈ norm(env)^2
        @test norm(env)^2 ≈ norm(env_ref)^2 - 3 * norm(only(env.corners))^2
        @test norm(scale(env, α)) ≈ abs(α) * norm(env)
        @test scale(env, α) ≈ α * env
        @test add(env, env, α, β) ≈ (α + β) * env
        @test env + env ≈ 2 * env
        @test norm(env - env) ≈ 0 atol = 1.0e-12
        z = zerovector(env)
        @test z isa C4vCTMRGEnv
        @test norm(z) == 0
        add!(z, env, α, one(stype))
        @test z ≈ α * env
        scale!(z, inv(α))
        @test z ≈ env
    end

    return nothing
end

function test_c4v_peps_observables(peps::InfinitePEPS, env::C4vCTMRGEnv)
    env_ref = _expand_c4v_env(env)
    O = adapt(storagetype(peps), _random_local_operator(physicalspace(peps, 1, 1)))
    @test expectation_value(peps, O, env) ≈ expectation_value(peps, O, env_ref)

    num_vals = 2
    ξ_h, ξ_v, λ_h, λ_v = correlation_length(peps, env; num_vals)
    ξ_h_ref, ξ_v_ref, λ_h_ref, λ_v_ref = correlation_length(peps, env_ref; num_vals)
    @test ξ_h ≈ ξ_h_ref
    @test ξ_v ≈ ξ_v_ref
    # eigsolve may return more than `num_vals` eigenvalues depending on the random start
    # vector, so only the requested leading eigenvalues are comparable; compare magnitudes
    # since the order within a complex-conjugate pair is not fixed
    leading(λs) = map(λ -> abs.(λ[1:num_vals]), λs)
    @test leading(λ_h) ≈ leading(λ_h_ref)
    @test leading(λ_v) ≈ leading(λ_v_ref)
    return nothing
end

function test_c4v_ctmrg_contractions(AT, P, V, Venv)
    @testset "C4v PEPS contractions" begin
        peps = _random_c4v_peps(AT, P, V)
        n = InfiniteSquareNetwork(peps)
        env = initialize_random_c4v_env(peps, Venv)

        @test storagetype(n) <: AT
        test_c4v_contractions(n, env)
        test_c4v_peps_observables(peps, env)
    end

    @testset "C4v PEPS contractions with singlet initialization" begin
        peps = _random_c4v_peps(AT, P, V)
        n = InfiniteSquareNetwork(peps)
        env = initialize_singlet_c4v_env(peps, Venv)

        test_c4v_contractions(n, env)
    end

    @testset "C4v PEPS contractions with generic constructor" begin
        peps = _random_c4v_peps(AT, P, V)
        n = InfiniteSquareNetwork(peps)
        env = C4vCTMRGEnv(randn, stype, peps, Venv)

        test_c4v_contractions(n, env)
    end

    @testset "C4v PartitionFunction contractions" begin
        pf = adapt(AT, InfinitePartitionFunction(randn, stype, V))
        n = InfiniteSquareNetwork(pf)
        env = initialize_random_c4v_env(pf, Venv)

        @test storagetype(n) <: AT
        test_c4v_contractions(n, env)
    end

    @testset "C4v environment after one CTMRG iteration" begin
        peps = _random_c4v_peps(AT, P, V)
        n = InfiniteSquareNetwork(peps)
        env₀ = initialize_random_c4v_env(peps, Venv)
        for projector_alg in (:C4vEighProjector, :C4vQRProjector)
            alg = C4vCTMRG(; projector_alg)
            env, = PEPSKit.ctmrg_iteration(n, env₀, alg)
            @test env isa C4vCTMRGEnv
            test_c4v_contractions(n, env)
        end
    end

    return nothing
end

function c4v_ctmrg_contractions_cartesian_spaces(AT)
    Random.seed!(20261028)
    return @testset "C4v contractions with Cartesian spaces ($AT)" begin
        test_c4v_ctmrg_contractions(AT, ℂ^2, ℂ^3, ℂ^8)
    end
end

function c4v_ctmrg_contractions_u1_spaces(AT)
    Random.seed!(20261028)
    return @testset "C4v contractions with U1 spaces ($AT)" begin
        P = U1Space(-1 => 1, 1 => 1)
        V = U1Space(-1 => 1, 0 => 1, 1 => 1)
        Venv = U1Space(-2 => 1, -1 => 2, 0 => 3, 1 => 2, 2 => 1)
        test_c4v_ctmrg_contractions(AT, P, V, Venv)
    end
end

function c4v_ctmrg_contractions_fermionic_spaces(AT)
    Random.seed!(20261028)
    return @testset "C4v contractions with fermionic spaces ($AT)" begin
        S = Vect[FermionParity]
        test_c4v_ctmrg_contractions(AT, S(0 => 1, 1 => 1), S(0 => 2, 1 => 1), S(0 => 4, 1 => 4))
    end
end
