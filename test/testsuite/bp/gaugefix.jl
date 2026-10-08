using Test
using TestExtras: @testinferred
using Random
using TensorKit
using PEPSKit, Adapt
using PEPSKit: compare_weights, random_dual!, twistdual
using PEPSKit: _next, _is_bipartite

function bp_gaugefix_bp_vs_su(AT)
    return @testset "BP vs SU ($AT) ($S, bipartite = $(bipartite), posdef msgs = $h)" for
        (S, bipartite, h) in Iterators.product(
            [U1Irrep, FermionParity], [true, false], [true, false]
        )
        unitcell = bipartite ? (2, 2) : (2, 3)
        elt = ComplexF64
        maxiter, tol = 100, 1.0e-9
        Random.seed!(52840679)
        Pspaces, Nspaces, Espaces = if S == U1Irrep
            map(rand(1:2, unitcell), rand(1:2, unitcell), rand(1:2, unitcell)) do d0, d1, d2
                    Vect[S](0 => d0, 1 => d1, -1 => d2)
            end,
                map(rand(2:4, unitcell), rand(2:4, unitcell), rand(2:4, unitcell)) do d0, d1, d2
                    Vect[S](0 => d0, 1 => d1, -1 => d2)
            end,
                map(rand(2:4, unitcell), rand(2:4, unitcell), rand(2:4, unitcell)) do d0, d1, d2
                    Vect[S](0 => d0, 1 => d1, -1 => d2)
            end
        else
            map(rand(2:3, unitcell), rand(2:3, unitcell)) do d0, d1
                    Vect[S](0 => d0, 1 => d1)
            end,
                map(rand(2:4, unitcell), rand(2:4, unitcell)) do d0, d1
                    Vect[S](0 => d0, 1 => d1)
            end,
                map(rand(2:4, unitcell), rand(2:4, unitcell)) do d0, d1
                    Vect[S](0 => d0, 1 => d1)
            end
        end
        Nspaces, Espaces = random_dual!(Nspaces), random_dual!(Espaces)
        if bipartite
            for c in 1:2
                cp1 = _next(c, 2)
                Pspaces[2, c] = Pspaces[1, cp1]
                Nspaces[2, c] = Nspaces[1, cp1]
                Espaces[2, c] = Espaces[1, cp1]
            end
        end
        peps0 = adapt(AT, InfinitePEPS(randn, elt, Pspaces, Nspaces, Espaces))
        if bipartite
            for c in 1:2
                peps0[2, c] = copy(peps0[1, c + 1])
            end
        end

        # start by gauging with SU
        peps1, wts1 = gauge_fix(peps0, SUGauge(; maxiter, tol))
        for (a0, a1) in zip(peps0.A, peps1.A)
            @test space(a0) == space(a1)
        end
        if bipartite
            @test _is_bipartite(peps1)
            @test _is_bipartite(wts1)
        end
        normalize!.(wts1.data)

        # find BP fixed point and SUWeight
        bp_alg = BeliefPropagation(; maxiter, tol, bipartite, project_hermitian = h)
        env = BPEnv(randn, elt, peps1; posdef = h)
        env, err = leading_boundary(env, peps1, bp_alg)
        if bipartite
            @test _is_bipartite(env)
        end
        wts2 = SUWeight(env)
        normalize!.(wts2.data)
        @test compare_weights(wts1, wts2) < 1.0e-9

        bpg_alg = BPGauge()
        peps2, gauge = @testinferred gauge_fix(peps1, bpg_alg, env)
        @test gauge_transform(peps1, gauge) ≈ peps2
        if bipartite
            @test _is_bipartite(peps2)
        end
        for (a1, a2) in zip(peps1.A, peps2.A)
            @test space(a1) == space(a2)
        end
        for (X, Xinv) in zip(gauge.matrices, gauge.inverses)
            # X, Xinv should contract to identity
            @tensor tmp[-1; -2] := X[-1; 1] * Xinv[1; -2]
            @test tmp ≈ twistdual(TensorKit.id(space(X, 1)), 1)
            # BP should differ from SU only by a unitary gauge transformation
            @test inv(X) ≈ adjoint(X) ≈ Xinv
        end
    end
end

"""Test that the BP gauge reproduces a PEPO and preserves its CTMRG density matrices."""
function bp_gaugefix_pepo(AT)
    return @testset "BP gauge of PEPO ($AT) ($S)" for S in (U1Irrep, FermionParity)
        P = Vect[S](0 => 1, 1 => 1)
        V = S == U1Irrep ? Vect[S](-1 => 1, 0 => 2, 1 => 1) : Vect[S](0 => 2, 1 => 1)
        N, E = random_dual!(fill(V, 2, 3)), random_dual!(fill(V, 2, 3))
        ρ = adapt(AT, InfinitePEPO(fill(P, 2, 3), N, E))
        bp_env = adapt(AT, BPEnv(randn, ComplexF64, ρ))
        ρg, gauge = gauge_fix(ρ, BPGauge(), bp_env)
        @test gauge_transform(ρ, gauge) ≈ ρg
        env = adapt(AT, CTMRGEnv(InfinitePEPS(ρ), V))
        envg = gauge_transform(env, gauge)
        inds = [CartesianIndex(1, 1)]
        @test reduced_densitymatrix(inds, ρ, ρ, env) ≈ reduced_densitymatrix(inds, ρg, ρg, envg)
    end
end
