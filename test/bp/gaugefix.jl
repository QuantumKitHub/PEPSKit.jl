using Test, TestExtras
using Random
using TensorKit
using PEPSKit
using PEPSKit: compare_weights, random_dual!, twistdual
using PEPSKit: _next, _is_bipartite

@testset "BP vs SU ($S, bipartite = $(bipartite), posdef msgs = $h)" for
    (S, bipartite, h) in Iterators.product(
        [U1Irrep, FermionParity], [true, false], [true, false]
    )
    unitcell = bipartite ? (2, 2) : (2, 3)
    elt = ComplexF64
    maxiter, tol = 100, 1.0e-9
    Random.seed!(52840679)
    Pspaces, Nspaces, Espaces = if S == U1Irrep
        map(zip(rand(1:2, unitcell), rand(1:2, unitcell), rand(1:2, unitcell))) do (d0, d1, d2)
                Vect[S](0 => d0, 1 => d1, -1 => d2)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1, d2)
                Vect[S](0 => d0, 1 => d1, -1 => d2)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1, d2)
                Vect[S](0 => d0, 1 => d1, -1 => d2)
        end
    else
        map(zip(rand(2:3, unitcell), rand(2:3, unitcell))) do (d0, d1)
                Vect[S](0 => d0, 1 => d1)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1)
                Vect[S](0 => d0, 1 => d1)
        end,
            map(zip(rand(2:4, unitcell), rand(2:4, unitcell))) do (d0, d1)
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
    peps0 = InfinitePEPS(randn, elt, Pspaces, Nspaces, Espaces)
    if bipartite
        for c in 1:2
            peps0[2, c] = copy(peps0[1, _next(c, 2)])
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
    peps2, XXinv = @constinferred gauge_fix(peps1, bpg_alg, env)
    if bipartite
        @test _is_bipartite(peps2)
    end
    for (a1, a2) in zip(peps1.A, peps2.A)
        @test space(a1) == space(a2)
    end
    for (X, Xinv) in XXinv
        # X, Xinv should contract to identity
        @tensor tmp[-1; -2] := X[-1; 1] * Xinv[1; -2]
        @test tmp ≈ twistdual(TensorKit.id(space(X, 1)), 1)
        # BP should differ from SU only by a unitary gauge transformation
        @test inv(X) ≈ adjoint(X) ≈ Xinv
    end
end
