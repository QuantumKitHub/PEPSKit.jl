using Test, TestExtras
using Random
using TensorKit
using PEPSKit
using PEPSKit: SUGauge, gauge_fix, compare_weights, random_dual!

@testset "Compare BP and SU ($S, Herm msgs = $h)" for (S, h) in Iterators.product([U1Irrep], [true, false])
    unitcell = (2, 3)
    elt = ComplexF64
    maxiter, tol = 100, 1.0e-9

    Random.seed!(0)
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
    peps0 = InfinitePEPS(randn, elt, Pspaces, Nspaces, Espaces)

    # start by gauging with SU
    peps1, wts1 = gauge_fix(peps0, SUGauge(; maxiter, tol))
    for (a0, a1) in zip(peps0.A, peps1.A)
        @test space(a0) == space(a1)
    end
    normalize!.(wts1.data)

    # find BP fixed point and SUWeight
    bp_alg = BeliefPropagation(; maxiter, tol, project_hermitian = h)
    env = BPEnv(h ? ones : randn, elt, peps1)
    env, err = leading_boundary(env, peps1, bp_alg)
    wts2 = SUWeight(env; ishermitian = h)
    normalize!.(wts2.data)
    @test compare_weights(wts1, wts2) < 1.0e-9

    # BP should differ from SU only by a unitary gauge transformation
    bpg_alg = BPGauge(; ishermitian = h)
    peps2, XXinv = @constinferred gauge_fix(peps1, bpg_alg, env)
    for (a1, a2) in zip(peps1.A, peps2.A)
        @test space(a1) == space(a2)
    end
    for (X, Xinv) in XXinv
        @test inv(X) ≈ adjoint(X) ≈ Xinv
    end
end
