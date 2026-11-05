using Test
using Random
using StableRNGs
using TensorKit
using MPSKit
using PEPSKit
using OptimKit
using Zygote
using LinearAlgebra
using KrylovKit

using PEPSKit: unitcell, norm_preserving_retract

sd = 123456
Random.seed!(sd)

D = 2
χ = 7
T = ComplexF64

ctmrg_tol = 1.0e-8
ctmrg_maxiter = 300
ctmrg_verbosity = 2

const ising_βc_triangular = BigFloat(BigFloat(asinh(BigFloat(sqrt(BigFloat(1.0) / BigFloat(3.0))))) / BigFloat(2.0))
const f_onsager_triangular::BigFloat = -3.20253248660790791834355252025862951439

function classical_ising_triangular(β)
    t = Float64[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1

    H = [1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3; -4 -5 -6] := O[1 2 3; 4 5 6] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4] * nt[-5; 5] * nt[-6; 6]
    @tensor o2[-1 -2 -3; -4 -5 -6] := o[1 2 3; 4 5 6] * H[-1; 1] * H[-2; 2] * H[-3; 3] * H[-4; 4] * H[-5; 5] * H[-6; 6]
    return TensorMap(o2, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
end

@testset "CTM_triangular - Coordinates" begin
    neighbors = ((3, 1), (3, 2), (1, 2), (2, 1), (2, 4), (1, 4))
    for dir in 1:6
        @test PEPSKit._coordinates(dir, 1, 1, (3, 4)) == neighbors[dir]
    end
end

@testset "CTM_triangular - Network value phases" begin
    V = ComplexSpace(1)
    for sandwich in (false, true)
        network = if sandwich
            A = normalize(ones(Float64, ComplexSpace(2), V^3 ⊗ (V')^3))
            InfiniteTriangularNetwork(InfinitePEPSTriangular(A; unitcell = (2, 2)))
        else
            InfiniteTriangularNetwork(fill(ones(Float64, V^3, V^3), 2, 2))
        end
        env = CTMRGEnvTriangular(ones, ComplexF64, sandwich ? V ⊗ V' : V, V; unitcell = size(network))
        @test network_value(network, env) ≈ 1
        # Arbitrary boundary phases must cancel across the unit cell.
        for field in (:C, :Ea, :Eb), phase in (-1, cis(2π / 3))
            env′ = deepcopy(env)
            getproperty(env′, field)[3, 1, 1] *= phase
            @test network_value(network, env′) ≈ 1
        end
    end

    # Two negative scalar tensors give a positive value per unit cell.
    network = InfiniteTriangularNetwork(fill(-ones(Float64, V^3, V^3), 1, 2))
    env = CTMRGEnvTriangular(ones, ComplexF64, V, V; unitcell = size(network))
    @test network_value(network, env) ≈ 1

    for α in (1.1, 1.1 + 0.1im)
        g = only(Zygote.gradient(n -> real(network_value(n, env)), α * network))
        @test all(t -> t ≈ -conj(α) * ones(Float64, V^3, V^3), unitcell(g))
    end

    # Taking the cube root must not require cubing the full unit-cell magnitude.
    for scale in (1.0e-30, 1.0e30)
        network = InfiniteTriangularNetwork(fill(scale * ones(Float64, V^3, V^3), 2, 2))
        env = CTMRGEnvTriangular(ones, ComplexF64, V, V; unitcell = size(network))
        @test network_value(network, env) ≈ scale^length(network)
    end
end

@testset "CTM_triangular - Direct edge split" begin
    Random.seed!(19461)
    vspace = ComplexSpace(2)
    envspace = ComplexSpace(3)
    for (V, inds) in ((envspace ⊗ vspace, (3, 2)), (envspace ⊗ vspace ⊗ vspace', (4, 2, 3))), conditioning in (false, true), ill_conditioned in (false, true)
        mat, ρL, ρR = (randn(ComplexF64, V, V) + 5 * id(V) for _ in 1:3)
        if ill_conditioned
            block(ρL, Trivial()) .= Diagonal([ones(dim(V) - 1); 1.0e-14])
            block(ρR, Trivial()) .= Diagonal([ones(dim(V) - 1); 1.0e-14])
        end
        L, R = ρL / norm(ρL), ρR / norm(ρR)
        if conditioning
            _, SL, VL = svd_full(L)
            UR, SR, _ = svd_full(R)
            L = PEPSKit.sdiag_pow(SL, 0.5) * VL
            R = UR * PEPSKit.sdiag_pow(SR, 0.5)
        end
        ρ = L * mat * R
        for trunc in (notrunc(), truncrank(3), truncspace(envspace))
            Ea, Eb = PEPSKit._truncate_edge(mat, ρL, ρR, trunc; conditioning)
            reconstructed = Eb * permute(Ea, ((1,), inds))
            U, S, Vᴴ = svd_trunc(ρ / norm(ρ); trunc)
            @test space(Ea, 1) == space(S, 1)
            @test L * reconstructed * R ≈ norm(ρ)^2 * U * S * Vᴴ
            if trunc == notrunc() && !ill_conditioned
                @test reconstructed / norm(reconstructed) ≈ mat / norm(mat)
            end
        end
    end
end

@testset "CTM_triangular - SVD pullback settings" begin
    Random.seed!(21735)
    V = ℂ^2 ⊗ ℂ^2
    for T in (Float64, ComplexF64), pullback in (:FullPullback, :TruncPullback)
        tensors = ntuple(_ -> randn(T, V, V) + 5 * id(V), 3)
        directions = map(t -> normalize(randn(T, space(t))), tensors)
        target = randn(T, V, V)
        decomposition_alg = SVDAdjoint(; fwd_alg = (; alg = :QRIteration), rrule_alg = (; alg = pullback, verbosity = 0))
        loss = function (mat, L, R)
            Ea, Eb, = PEPSKit._truncate_edge(mat, L, R, truncrank(2), decomposition_alg)
            return real(dot(target, Eb * permute(Ea, ((1,), (3, 2)))))
        end
        result = @test_logs min_level = Base.CoreLogging.Warn withgradient(loss, tensors...)
        ε = 1.0e-5
        for i in 1:3
            plus = Base.setindex(tensors, tensors[i] + ε * directions[i], i)
            minus = Base.setindex(tensors, tensors[i] - ε * directions[i], i)
            @test real(dot(result.grad[i], directions[i])) ≈ (loss(plus...) - loss(minus...)) / (2ε) rtol = 1.0e-5 atol = 1.0e-7
        end
        @test decomposition_alg.fwd_alg isa PEPSKit.MatrixAlgebraKit.QRIteration
    end
end

@testset "CTM_triangular - Direct edge coordinates" begin
    Random.seed!(68191)
    sz = (3, 4)
    vspace = ComplexSpace(1)
    offsets = ((-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1))
    for sandwich in (false, true)
        tensors = map(Iterators.product(1:sz[1], 1:sz[2])) do _
            if sandwich
                V = vspace^3 ⊗ (vspace')^3
                return randn(ComplexF64, vspace, V), randn(ComplexF64, vspace, V)
            end
            return randn(ComplexF64, vspace^3, vspace^3)
        end
        network = InfiniteTriangularNetwork(tensors)
        env = CTMRGEnvTriangular(randn, ComplexF64, sandwich ? vspace ⊗ vspace' : vspace, vspace; unitcell = sz)
        at(A, d, r, c) = only(A[mod1(d, 6), mod1(r, sz[1]), mod1(c, sz[2])].data)
        function site(r, c)
            t = tensors[mod1(r, sz[1]), mod1(c, sz[2])]
            return sandwich ? only(t[1].data) * conj(only(t[2].data)) : only(t.data)
        end
        for dir in 1:6, r in 1:sz[1], c in 1:sz[2]
            δL, δR = offsets[mod1(dir + 4, 6)], offsets[mod1(dir + 3, 6)]
            rL, cL = r + δL[1], c + δL[2]
            rR, cR = r + δR[1], c + δR[2]
            left = site(rL, cL) * at(env.Ea, dir + 3, rL + δR[1], cL + δR[2])
            right = site(rR, cR) * at(env.Eb, dir + 3, rR + δL[1], cR + δL[2])
            for i in 0:2
                δ = offsets[mod1(dir - i, 6)]
                left *= at(env.C, dir - i, rL + δ[1], cL + δ[2])
                δ = offsets[mod1(dir + i + 1, 6)]
                right *= at(env.C, dir + i + 1, rR + δ[1], cR + δ[2])
            end
            ρL, ρR = PEPSKit.build_halfinfinite_environments(network, env, dir, r, c)
            @test only(ρL.data) ≈ left
            @test only(ρR.data) ≈ right
        end
    end
end

@testset "CTM_triangular - Simultaneous edge update" begin
    Random.seed!(38471)
    vspace = ComplexSpace(2)
    ket = randn(ComplexF64, ComplexSpace(2), vspace^3 ⊗ (vspace')^3)
    pf = randn(ComplexF64, vspace^3, vspace^3)
    for (sandwich, V) in ((pf, vspace), ((ket, ket), vspace ⊗ vspace'))
        network = InfiniteTriangularNetwork(fill(sandwich, (1, 1)))
        env = CTMRGEnvTriangular(randn, ComplexF64, V, ComplexSpace(2))
        trunc = PEPSKit.FixedSpaceTruncation()
        Pas, Pbs, = PEPSKit.calculate_projectors(network, env, trunc, :twothirds)
        mats, preliminary = PEPSKit.semi_renormalize(network, env, Pas, Pbs, trunc)
        for conditioning in (false, true)
            previous = deepcopy(preliminary)
            edges = map(PEPSKit.eachcoordinate(network, 1:6)) do (dir, r, c)
                ρL, ρR = PEPSKit.build_halfinfinite_environments(network, preliminary, dir, r, c)
                trunc′ = PEPSKit._truncation_strategy(trunc, preliminary.Ea[dir, r, c])
                return PEPSKit._truncate_edge(mats[dir, r, c], ρL, ρR, trunc′; conditioning)
            end
            expected = PEPSKit.normalize_edges(CTMRGEnvTriangular(preliminary.C, getindex.(edges, 1), getindex.(edges, 2)))
            updated, = PEPSKit.renormalize_edges(network, preliminary, mats, trunc; conditioning)
            @test all(isapprox.(updated.Ea, expected.Ea))
            @test all(isapprox.(updated.Eb, expected.Eb))
            @test preliminary.Ea == previous.Ea
            @test preliminary.Eb == previous.Eb
        end
    end
end

@testset "CTM_triangular - Random tensor" begin
    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(79413165445)
            alg = SimultaneousCTMRGTriangular(;
                tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity,
                conditioning, projector_alg
            )
            pspace = ComplexSpace(2)
            vspace = ComplexSpace(D)
            envspace = ComplexSpace(χ)

            ket = randn(T, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
            bra = copy(ket)
            pf = randn(T, vspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace ⊗ vspace)
            sandwiches = [pf, (ket, bra)]
            # sandwiches = [(ket, bra), pf]
            unitcell = (2, 2)

            for (sandwich, V) in zip(sandwiches, [vspace, vspace ⊗ vspace'])
                # for (sandwich, vspace) in zip(sandwiches, [vspace ⊗ vspace', vspace])
                network = InfiniteTriangularNetwork(fill(sandwich, unitcell))
                env₀ = CTMRGEnvTriangular(randn, T, V, envspace; unitcell)
                env, info = leading_boundary(env₀, network, alg)
                @test info.convergence_metric < 1.0 # this is not much of a test
            end
        end
    end
end

@testset "CTM_triangular - Classical Ising" begin
    χ_local = 20
    T_local = Float64

    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(156484561351)

            alg = SimultaneousCTMRGTriangular(;
                tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity,
                conditioning, projector_alg
            )
            sz = (1, 1)
            T = classical_ising_triangular(ising_βc_triangular)
            pf = InfiniteTriangularNetwork(fill(T, sz))

            vspace = codomain(T)[1]
            envspace = ComplexSpace(χ_local)
            env₀ = CTMRGEnvTriangular(randn, T_local, vspace, envspace; unitcell = sz)
            env, info = leading_boundary(env₀, pf, alg)

            nw_value = network_value(pf, env)
            lz = real(log(nw_value))
            fs = lz * -1 / ising_βc_triangular
            @test fs ≈ f_onsager_triangular rtol = 1.0e-4
        end
    end
end


## Gradient tests for triangular CTMRG
# ------------------------------------

Pspaces = [ComplexSpace(2)]
Vspaces = [ComplexSpace(D)]
Espaces = [ComplexSpace(χ)]
# TODO: actually add triangular lattice models...
models = [transverse_field_ising(InfiniteSquare(); J = 1.0, g = 1.0)]
names = ["Heisenberg"]

gradtol = 1.0e-4
ctmrg_verbosity = 0
ctmrg_algs = [[:triangular]]
projector_algs = [[:twothirds, :full]]
conditionings = [[true, false]]
# svd_rrule_algs = nothing # TODO: allow this
gradient_algs = [[nothing]]
# gradient_iterschemes = [[:fixed, :diffgauge]] # TODO: support this
steps = -0.01:0.005:0.01

# have to use a custom retraction, since the state type of hardcoded...
function peps_triangular_retract(x, η, α)
    peps = x[1]
    env = deepcopy(x[2])

    retractions = norm_preserving_retract.(unitcell(peps), unitcell(η), α)
    peps´ = typeof(peps)(map(first, retractions))
    ξ = typeof(peps)(map(last, retractions))

    return (peps´, env), ξ
end


@testset "Triangular CTMRG energy gradient test for $(names[i]) model" verbose = true for i in
    eachindex(
        models
    )
    Pspace = Pspaces[i]
    Vspace = Vspaces[i]
    Espace = Espaces[i]
    # calgs = ctmrg_algs[i]
    palgs = projector_algs[i]
    conds = conditionings[i]
    # salgs = svd_rrule_algs[i]
    galgs = gradient_algs[i]
    # gischemes = gradient_iterschemes[i]

    # TODO: allow passing actual LocalOperator
    H = models[i]
    O1 = last(first(filter(t -> length(first(t)) == 1, H.terms)))
    O2 = last(first(filter(t -> length(first(t)) == 2, H.terms)))

    @testset "ctmrg_alg=:simultaneous_triangular, projector_alg=:$projector_alg, conditioning=$conditioning, gradient_alg=:$gradient_alg" for (
            projector_alg, conditioning, gradient_alg,
        ) in Iterators.product(palgs, conds, galgs)

        @info "optimtest of ctmrg_alg=:simultaneous_triangular, projector_alg=:$projector_alg, conditioning=$conditioning, gradient_alg=:$gradient_alg on $(names[i])"
        Random.seed!(42039482030)
        dir = InfinitePEPSTriangular(Pspace, Vspace)
        psi = InfinitePEPSTriangular(Pspace, Vspace)

        # TODO: actually go through CTMRGAlgorithm selection
        contrete_ctmrg_alg = SimultaneousCTMRGTriangular(;
            projector_alg, conditioning, verbosity = ctmrg_verbosity,
            trunctype = :FixedSpaceTruncation
        )
        # instantiate because hook_pullback doesn't go through the keyword selector...
        concrete_gradient_alg = if isnothing(gradient_alg)
            nothing
        else
            PEPSKit.GradientAlgorithm(; alg = gradient_alg, tol = gradtol)
        end
        env, = leading_boundary(
            CTMRGEnvTriangular(randn, ComplexF64, Vspace ⊗ Vspace', Espace),
            psi, contrete_ctmrg_alg
        )
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha = steps,
            retract = peps_triangular_retract,
            inner = PEPSKit.real_inner,
        ) do (peps, env)
            E, g = Zygote.withgradient(peps) do psi
                env2, = PEPSKit.hook_pullback(
                    leading_boundary,
                    env,
                    psi,
                    contrete_ctmrg_alg;
                    alg_rrule = concrete_gradient_alg,
                )
                return PEPSKit.energy(psi, env2, O1, O2)
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1.0e-2
    end
end

@testset "CTM_triangular - Truncation diagnostics" begin
    V = ComplexSpace(2) ⊗ ComplexSpace(2)
    for T in (Float64, ComplexF64), conditioning in (false, true), scale in (1.0e-30, 1.0, 1.0e30)
        for values in ([4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 1.0e-9, 2.0e-10], [4.0, 3.0, 1.0e-14, 2.0e-15], [4.0, 3.0, 0.0, 0.0])
            mat = TensorMap(Matrix(Diagonal(T.(scale .* values))), V, V)
            for rank in 1:5
                _, _, error = PEPSKit._truncate_edge(mat, id(V), id(V), truncrank(rank); conditioning)
                kept = values[1:min(rank, end)]
                discarded = [values[(min(rank, end) + 1):end]; filter(<(eps(Float64)^(3 / 4) * maximum(kept)), kept)]
                @test error ≈ norm(discarded) / norm(kept) rtol = 1.0e-12
            end
            _, _, error = PEPSKit._truncate_edge(mat, id(V), id(V), notrunc(); conditioning)
            discarded = filter(<(eps(Float64)^(3 / 4) * maximum(values)), values)
            @test error ≈ norm(discarded) / norm(values) rtol = 1.0e-12
        end
    end

    # Non-Abelian sector dimensions must weight both the cutoff loss and norm(S).
    S = DiagonalTensorMap([4.0, 3.0, 1.0e-14], SU2Space(0 => 1, 1 // 2 => 2))
    isqrtS = PEPSKit.sdiag_pow(S, -0.5)
    error = PEPSKit._relative_truncation_error(S, isqrtS, 2.0e-15)
    @test error ≈ sqrt((2.0e-15)^2 + 2 * (1.0e-14)^2) / sqrt(4.0^2 + 2 * 3.0^2)
    for S in (DiagonalTensorMap(zeros(2), ComplexSpace(2)), DiagonalTensorMap(Float64[], ComplexSpace(0)))
        @test iszero(PEPSKit._relative_truncation_error(S, S, 0.0))
        @test isinf(PEPSKit._relative_truncation_error(S, S, 1.0))
    end
end

@testset "CTM_triangular - Convergence spectra" begin
    Random.seed!(71834)
    vspace = ComplexSpace(2)
    alg = SimultaneousCTMRGTriangular(; verbosity = 0)
    for V in (vspace, vspace ⊗ vspace')
        env = CTMRGEnvTriangular(randn, ComplexF64, V, vspace; unitcell = (1, 2))
        CS, TS = PEPSKit.convergence_spectra(env, alg)
        @test size(CS) == (6, 1, 2)
        @test size(TS) == (12, 1, 2)
        @test first(PEPSKit.calc_convergence(env, CS, TS, alg)) == 0
        for field in (:C, :Ea, :Eb)
            changed = deepcopy(env)
            tensors = getproperty(changed, field)
            expected = norm(tensors[6, 1, 2])
            tensors[6, 1, 2] *= 2
            @test first(PEPSKit.calc_convergence(changed, CS, TS, alg)) ≈ expected
        end
    end
end

@testset "CTM_triangular - Iteration diagnostics" begin
    Random.seed!(88193)
    vspace = ComplexSpace(2)
    sz = (1, 2)
    for sandwich in (false, true)
        tensors = map(Iterators.product(1:sz[1], 1:sz[2])) do _
            if sandwich
                ket = randn(ComplexF64, vspace, vspace^3 ⊗ (vspace')^3)
                return ket, copy(ket)
            end
            return randn(ComplexF64, vspace^3, vspace^3)
        end
        network = InfiniteTriangularNetwork(tensors)
        env₀ = CTMRGEnvTriangular(randn, ComplexF64, sandwich ? vspace ⊗ vspace' : vspace, vspace; unitcell = sz)
        for projector_alg in (:twothirds, :full), conditioning in (false, true), trunctype in (:truncrank, :FixedSpaceTruncation)
            alg = SimultaneousCTMRGTriangular(; projector_alg, conditioning, trunctype, verbosity = 0)
            env₁, info₁ = PEPSKit.ctmrg_iteration(network, env₀, alg)
            _, info₂ = PEPSKit.ctmrg_iteration(network, env₁, alg)
            for (maxiter, miniter, tol, expected) in ((1, 1, Inf, info₁), (2, 2, Inf, info₂), (1, 1, 0.0, info₁))
                alg = SimultaneousCTMRGTriangular(; projector_alg, conditioning, trunctype, verbosity = 0, maxiter, miniter, tol)
                env, info = leading_boundary(env₀, network, alg)
                CS, TS = PEPSKit.convergence_spectra(maxiter == 1 ? env₀ : env₁, alg)
                @test info.convergence_error ≈ first(PEPSKit.calc_convergence(env, CS, TS, alg))
                metrics = info.contraction_metrics
                @test metrics == expected.contraction_metrics
                @test all(isfinite, metrics)
                @test metrics.corner_truncation_error > 0
                @test metrics.edge_truncation_error > 0
                @test metrics.truncation_error == max(metrics.corner_truncation_error, metrics.edge_truncation_error)
                @test info.converged == isinf(tol)
                @test info.convergence_error == info.convergence_metric
            end
        end
    end
end

@testset "Triangular algorithm selection" begin
    V = ComplexSpace(1)
    state = InfinitePEPSTriangular(ones(ComplexF64, ComplexSpace(2), V^3 ⊗ (V')^3))
    env₀ = CTMRGEnvTriangular(ones, ComplexF64, V ⊗ V', V)
    kwargs = (; tol = 1.0e-9, maxiter = 2, miniter = 2, verbosity = 0, conditioning = false, trunctype = :FixedSpaceTruncation)
    alg = SimultaneousCTMRGTriangular(; kwargs...)
    @test PEPSKit.select_algorithm(leading_boundary, env₀; kwargs...) == alg
    env, info = leading_boundary(env₀, state, alg)
    for network in (state, InfiniteTriangularNetwork(state))
        env′, info′ = leading_boundary(env₀, network; alg = :SimultaneousCTMRGTriangular, kwargs...)
        @test (env′.C, env′.Ea, env′.Eb) == (env.C, env.Ea, env.Eb)
        @test info′ == info
    end
    @test_throws ArgumentError leading_boundary(env₀, state; alg = :unknown)

    decomposition_alg = SVDAdjoint(; fwd_alg = (; alg = :QRIteration), rrule_alg = (; verbosity = 1))
    configured = PEPSKit.select_algorithm(leading_boundary, env₀; kwargs..., decomposition_alg)
    @test configured.decomposition_alg === decomposition_alg
    @test SimultaneousCTMRGTriangular(; decomposition_alg = (; alg = :QRIteration)).decomposition_alg.fwd_alg isa PEPSKit.MatrixAlgebraKit.QRIteration
    @test PEPSKit.PEPSOptimize(; boundary_alg = (; alg = :SimultaneousCTMRGTriangular, decomposition_alg), precondition_alg = nothing).boundary_alg.decomposition_alg === decomposition_alg
    env′, info′ = leading_boundary(env₀, state, configured)
    @test all(isapprox.(env′.C, env.C))
    @test all(isapprox.(env′.Ea, env.Ea))
    @test all(isapprox.(env′.Eb, env.Eb))
    @test info′.converged == info.converged
    @test SimultaneousCTMRGTriangular().projector_alg === :full
    @test SimultaneousCTMRGTriangular(; projector_alg = :twothirds).projector_alg === :twothirds
    @test PEPSKit._set_fixed_truncation(configured).decomposition_alg === decomposition_alg
end

@testset "Triangular fixedpoint interface" begin
    P, V = ComplexSpace(2), ComplexSpace(1)
    state = InfinitePEPSTriangular(ones(Float64, P, V^3 ⊗ (V')^3))
    env = CTMRGEnvTriangular(ones, Float64, V ⊗ V', V)
    alg = PEPSOptimize(SimultaneousCTMRGTriangular(), nothing, OptimKit.LBFGS(), nothing, true, nothing)
    @test isnothing(PEPSKit.check_input(fixedpoint, state, env, alg))
    @test_throws ArgumentError fixedpoint(LocalOperator(physicalspace(state)), state, nothing, alg)
end

@testset "Triangular native optimization" begin
    P, V = ComplexSpace(2), ComplexSpace(1)
    Sz = TensorMap([1.0 0; 0 -1] / 2, P, P)
    for T in (Float64, ComplexF64)
        tensors = [TensorMap(reshape(T[1, a], 2, 1, 1, 1, 1, 1, 1), P, V^3 ⊗ (V')^3) for a in [0.6 1.3]]
        state = InfinitePEPSTriangular(tensors)
        original = deepcopy(state)
        env = CTMRGEnvTriangular(ones, T, V ⊗ V', V; unitcell = size(state))
        H = LocalOperator(physicalspace(state), [((1, c),) => Sz for c in 1:2])
        boundary_alg = (; miniter = 1, maxiter = 1)
        kwargs = (; boundary_alg, gradient_alg = nothing, precondition_alg = nothing, optimizer_alg = (; maxiter = 2, tol = 0), verbosity = 0)
        alg = PEPSKit.select_algorithm(fixedpoint, env; kwargs...)
        @test PEPSKit.parent_alg(alg.boundary_alg) isa SimultaneousCTMRGTriangular
        direct_alg = PEPSOptimize(; boundary_alg = (; alg = :SimultaneousCTMRGTriangular, miniter = 1, maxiter = 1), gradient_alg = nothing, precondition_alg = nothing)
        @test direct_alg.boundary_alg isa SimultaneousCTMRGTriangular
        @test direct_alg.boundary_alg.maxiter == 1
        @test PEPSKit.adapt_solver(alg.boundary_alg; iter = 2, g_global = 0.1) isa SimultaneousCTMRGTriangular
        iterations = Int[]
        finalize! = (x, f, g, iter) -> (push!(iterations, iter); (x, f, g))
        initial_cost = cost_function(state, env, H)
        ψ, env′, E, info = if T === Float64
            fixedpoint(H, state, env, alg; finalize!)
        else
            fixedpoint(H, state, env; kwargs..., finalize!)
        end
        @test ψ isa typeof(state)
        @test env′ isa typeof(env)
        @test E < initial_cost
        @test E ≈ cost_function(ψ, env′, H)
        @test all(isapprox.(norm.(unitcell(ψ)), 1))
        @test state == original
        @test iterations == [1, 2]
        @test info.costs[end] ≈ E
        @test info.last_gradient isa typeof(state)
        @test all(isfinite, info.gradnorms)
        @test length(info.contraction_metrics) == length(info.gradnorms_unitcell) == length(info.times) == 2

        if T === ComplexF64
            ψfixed, envfixed, Efixed, infofixed = fixedpoint(
                H, state, env; kwargs..., gradient_alg = (; tol = 1.0e-8),
                boundary_alg = (; miniter = 2, maxiter = 10, tol = 1.0e-10),
            )
            @test Efixed ≈ E
            @test infofixed.costs ≈ info.costs
            @test infofixed.gradnorms ≈ info.gradnorms
            @test Efixed ≈ cost_function(ψfixed, envfixed, H)
        else
            @test_throws ArgumentError fixedpoint(H, state, env; kwargs..., gradient_alg = (;))
        end
        @test_throws ArgumentError fixedpoint(H, state, env; kwargs..., gradient_alg = (; alg = :ImplicitGradient))
        @test_throws ArgumentError fixedpoint(H, state, nothing, alg)
        @test_throws ArgumentError fixedpoint(H, state, env; boundary_alg)
        @test_throws ArgumentError fixedpoint(H, state, env; boundary_alg, gradient_alg = nothing)
        @test_throws ArgumentError fixedpoint(H, state, env; kwargs..., symmetrization = Rotate())
        @test_throws ArgumentError fixedpoint(H, state, env; kwargs..., boundary_alg = SimultaneousCTMRG())
    end
end

@testset "Triangular AF Heisenberg optimization" begin
    rng = StableRNG(48631)
    V = ComplexSpace(2)
    state = InfinitePEPSTriangular((args...) -> randn(rng, args...), ComplexF64, V, V)
    Sx = TensorMap(ComplexF64[0 1; 1 0] / 2, V, V)
    Sy = TensorMap(ComplexF64[0 -im; im 0] / 2, V, V)
    Sz = TensorMap(ComplexF64[1 0; 0 -1] / 2, V, V)
    # Ferromagnetic nearest-neighbor Heisenberg energy per site.
    h = -(Sx ⊗ Sx + Sy ⊗ Sy + Sz ⊗ Sz)
    offsets = (CartesianIndex(0, 1), CartesianIndex(1, -1), CartesianIndex(1, 0))
    H = LocalOperator(physicalspace(state), [(CartesianIndex(1, 1), CartesianIndex(1, 1) + δ) => h for δ in offsets])
    env₀ = CTMRGEnvTriangular(ones, ComplexF64, V ⊗ V', ComplexSpace(4))
    boundary_alg = SimultaneousCTMRGTriangular(; projector_alg = :full, tol = 1.0e-10, maxiter = 100, miniter = 4, verbosity = 0)
    gradient_alg = PEPSKit.FixedPointGradient(; solver_alg = (; alg = :GMRES, krylovdim = 30), tol = 1.0e-8, maxiter = 10, verbosity = 0)
    env₀, info₀ = leading_boundary(env₀, state, boundary_alg)
    @test info₀.converged
    # A mixed one-site density matrix rules out a padded product state.
    m = [real(expectation_value(state, LocalOperator(physicalspace(state), ((1, 1),) => S), env₀)) for S in (Sx, Sy, Sz)]
    @test 0.5 ≤ 0.5 + 2sum(abs2, m) < 1 - 1.0e-4
    E₀ = cost_function(state, env₀, H)
    ψ, env, E, info = fixedpoint(
        H, state, env₀; boundary_alg, gradient_alg, precondition_alg = nothing,
        optimizer_alg = (; maxiter = 1, tol = 0), verbosity = 0,
    )
    @test E < E₀
    @test info.costs[end] ≈ E
    @test length(info.contraction_metrics) == 1
    @test all(isfinite, info.gradnorms)
    @test info.last_gradient isa typeof(ψ)
    @test isfinite(norm(info.last_gradient)) && norm(info.last_gradient) > 0
    @test all(isapprox.(norm.(unitcell(ψ)), 1))
    env′, info′ = leading_boundary(env, ψ, boundary_alg)
    @test info′.converged
    @test E ≈ cost_function(ψ, env′, H) atol = 1.0e-8
    # Check the returned nearest-neighbor gradient along one normalized direction.
    direction = info.last_gradient / norm(info.last_gradient)
    ε = 1.0e-4
    energies = map((-ε, ε)) do step
        ψ′ = ψ + step * direction
        env′, info′ = leading_boundary(env, ψ′, boundary_alg)
        @test info′.converged
        cost_function(ψ′, env′, H)
    end
    @test real(dot(info.last_gradient, direction)) ≈ (energies[2] - energies[1]) / (2ε) rtol = 1.0e-4 atol = 1.0e-6
end

@testset "Triangular gauge fixing" begin
    Random.seed!(63412)
    for (V, χ, sandwich) in (
            (ComplexSpace(2), ComplexSpace(2), false),
            (ComplexSpace(2), ComplexSpace(2), true),
            (Z2Space(0 => 1, 1 => 1), Z2Space(0 => 1, 1 => 1), true),
        )
        sz = (2, 3) # diagonal boundaries traverse both unit-cell axes
        env = CTMRGEnvTriangular(randn, ComplexF64, sandwich ? V ⊗ V' : V, χ; unitcell = sz)
        signs = map((:Ea, :Eb)) do _
            map(env.C) do t
                first(left_orth(randn(ComplexF64, space(t, 1), space(t, 1)); positive = true))
            end
        end
        rotated = PEPSKit.fix_relative_phases(env, (; Ea = signs[1], Eb = signs[2]))
        original = deepcopy(env)
        # Check that the gauges cancel in the physical boundary contractions.
        network = if sandwich
            InfiniteTriangularNetwork(InfinitePEPSTriangular([randn(ComplexF64, V, V^3 ⊗ (V')^3) for _ in 1:sz[1], _ in 1:sz[2]]))
        else
            InfiniteTriangularNetwork([randn(ComplexF64, V^3, V^3) for _ in 1:sz[1], _ in 1:sz[2]])
        end
        for contract in (PEPSKit._contract_corners, PEPSKit._contract_edges_0, PEPSKit._contract_edges_60, PEPSKit._contract_edges_120)
            @test contract((1, 2), network, rotated) ≈ contract((1, 2), network, env)
        end
        rotated = CTMRGEnvTriangular(im .* rotated.C, -rotated.Ea, -im .* rotated.Eb)
        fixed = PEPSKit.gauge_fix(rotated, env, PEPSKit.ScramblingEnvGauge())
        @test PEPSKit.calc_elementwise_convergence(env, fixed) < 1.0e-10
        @test (env.C, env.Ea, env.Eb) == (original.C, original.Ea, original.Eb)
        different = CTMRGEnvTriangular(ones, ComplexF64, sandwich ? V ⊗ V' : V, χ; unitcell = (1, 1))
        @test_throws ArgumentError PEPSKit.gauge_fix(different, env, PEPSKit.ScramblingEnvGauge())
    end
end

@testset "Triangular fixed-point differentiation" begin
    Random.seed!(48531)
    V = ComplexSpace(2)
    fields = [0.02 -0.03]
    β = 0.1
    W = sqrt(Symmetric([exp(β / 2) exp(-β / 2); exp(-β / 2) exp(β / 2)]))
    state = InfinitePEPSTriangular(
        map(fields) do h
            data = [exp(h * (3 - 2s) / 2 + im * h * s) * prod(W[s, i] for i in Tuple(I)) for s in 1:2, I in CartesianIndices((2, 2, 2, 2, 2, 2))]
            TensorMap(data, V, V^3 ⊗ (V')^3)
        end
    )
    Sz = TensorMap(ComplexF64[1 0; 0 -1] / 2, V, V)
    H = LocalOperator(physicalspace(state), [((1, c),) => Sz for c in 1:2])
    direction = InfinitePEPSTriangular(map(t -> Sz * t, unitcell(state)))
    env₀ = CTMRGEnvTriangular(ones, ComplexF64, V ⊗ V', V; unitcell = size(state))
    gradalg = PEPSKit.FixedPointGradient(; solver_alg = (; alg = :GMRES, krylovdim = 30), tol = 1.0e-8, maxiter = 10, verbosity = 0)
    for (projector_alg, conditioning) in ((:twothirds, false), (:full, true))
        alg = SimultaneousCTMRGTriangular(; projector_alg, conditioning, maxiter = 200, miniter = 4, tol = 1.0e-10, verbosity = 0)
        env, info = leading_boundary(env₀, state, alg)
        @test info.converged
        env′, = PEPSKit.ctmrg_iteration(InfiniteTriangularNetwork(state), env, PEPSKit._set_fixed_truncation(alg))
        fixed = PEPSKit.gauge_fix(env′, env, PEPSKit.ScramblingEnvGauge())
        @test PEPSKit.calc_elementwise_convergence(env, fixed) < 1.0e-7
        E, gs = withgradient(state) do A
            env′, = PEPSKit.hook_pullback(leading_boundary, env, A, alg; alg_rrule = gradalg)
            cost_function(A, env′, H)
        end
        g = only(gs)
        @test g isa typeof(state)
        # The custom rule reconverges from env before evaluating the energy.
        env_ref, = leading_boundary(env, state, alg)
        @test E ≈ cost_function(state, env_ref, H)
        ε = 1.0e-4
        energies = map((-ε, ε)) do step
            A = state + step * direction
            env′, info′ = leading_boundary(env, A, alg)
            @test info′.converged
            cost_function(A, env′, H)
        end
        @test real(dot(g, direction)) ≈ (energies[2] - energies[1]) / (2ε) rtol = 1.0e-5
    end
end

@testset "Triangular nearest-neighbor energy" begin
    Random.seed!(28514)
    pspace, vspace = ComplexSpace(2), ComplexSpace(1)
    vspaces = vspace^3 ⊗ (vspace')^3
    Sx = TensorMap(ComplexF64[0 1; 1 0] / 2, pspace, pspace)
    Sy = TensorMap(ComplexF64[0 -im; im 0] / 2, pspace, pspace)
    Sz = TensorMap(ComplexF64[1 0; 0 -1] / 2, pspace, pspace)
    spins = (Sx, Sy, Sz)
    # Unequal site norms expose neighbor-index errors even in identity contractions.
    vectors = [randn(ComplexF64, 2) for r in 1:3, c in 1:4]
    tensors = map(v -> TensorMap(reshape(v, 2, 1, 1, 1, 1, 1, 1), pspace, vspaces), vectors)
    state = InfinitePEPSTriangular(tensors)
    network = InfiniteTriangularNetwork(state)
    env = CTMRGEnvTriangular(ones, ComplexF64, vspace ⊗ vspace', vspace; unitcell = size(state))
    contractions = (PEPSKit._contract_edges_0, PEPSKit._contract_edges_60, PEPSKit._contract_edges_120)
    offsets = ((0, 1), (1, -1), (1, 0))
    pairs = ((Sx, Sz), (Sy, Sx), (Sz, Sy))
    ops = map(p -> p[1] ⊗ p[2], pairs)
    expectation(v, op) = dot(v, reshape(convert(Array, op), 2, 2) * v) / dot(v, v)
    expected = 0.0
    expected_horizontal = 0.0
    terms = Pair[]
    for r in 1:3, c in 1:4
        v = vectors[r, c]
        onsite = real(expectation(v, Sz))
        expected += onsite
        expected_horizontal += onsite
        push!(terms, ((r, c),) => Sz)
        for (i, (contract, (δr, δc))) in enumerate(zip(contractions, offsets))
            w = vectors[mod1(r + δr, 3), mod1(c + δc, 4)]
            n = dot(v, v) * dot(w, w)
            bond = expectation(v, pairs[i][1]) * expectation(w, pairs[i][2])
            @test contract((r, c), network, env) ≈ n
            @test contract((r, c), network, env, id(pspace ⊗ pspace)) ≈ n
            @test contract((r, c), network, env, ops[i]) / n ≈ bond
            expected += real(bond)
            i == 1 && (expected_horizontal += real(bond))
            push!(terms, ((r, c), (r + δr, c + δc)) => ops[i])
        end
    end
    @test PEPSKit.energy(state, env, Sz, ops) ≈ expected
    @test PEPSKit.energy(state, env, Sz, ops[1]) ≈ expected_horizontal

    H = LocalOperator(physicalspace(state), terms)
    @test expectation_value(state, H, env) ≈ expected
    @test expectation_value(state, im * H, env) ≈ im * expected
    @test cost_function(state, env, H) ≈ expected
    @test iszero(expectation_value(state, LocalOperator(physicalspace(state)), env))
    @test iszero(expectation_value(InfinitePEPS(pspace, vspace), LocalOperator(fill(pspace, 1, 1)), nothing))

    # Distinct bra/ket and an asymmetric diagonal bond check operator ordering.
    bra = InfinitePEPSTriangular(map(t -> 2Sx * t, tensors))
    bond = LocalOperator(physicalspace(state), ((1, 1), (2, 0)) => ops[2])
    transition(v, op) = dot(reverse(v), reshape(convert(Array, op), 2, 2) * v) / dot(reverse(v), v)
    @test expectation_value(bra, bond, state, env) ≈ transition(vectors[1, 1], Sy) * transition(vectors[2, 4], Sx)
    @test_throws ArgumentError expectation_value(state, LocalOperator(fill(pspace, 1, 1), ((1, 1),) => Sz), env)
    @test_throws ArgumentError expectation_value(state, LocalOperator(physicalspace(state), ((1, 1), (1, 3)) => ops[1]), env)
    @test_throws ArgumentError expectation_value(state, LocalOperator(physicalspace(state), ((1, 1), (1, 2), (2, 1)) => Sz ⊗ Sz ⊗ Sz), env)

    # Spin-1/2 120° product state: each of the three bonds contributes -1/8 per site.
    rotation(θ) = TensorMap([cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)], pspace, pspace)
    A = TensorMap(reshape([1.0, 0.0], 2, 1, 1, 1, 1, 1, 1), pspace, vspaces)
    B, C = rotation(2π / 3) * A, rotation(4π / 3) * A
    state = InfinitePEPSTriangular([A B C; C A B; B C A])
    env = CTMRGEnvTriangular(ones, Float64, vspace ⊗ vspace', vspace; unitcell = (3, 3))
    h = sum(S ⊗ S for S in spins)
    @test PEPSKit.energy(state, env, zero(Sz), (h, h, h)) / length(state) ≈ -3 / 8
end
