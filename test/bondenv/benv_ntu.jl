using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Nr, Nc = 2, 2
Random.seed!(20)

function test_ntu_env(
        state::Union{InfinitePEPS, InfinitePEPO}, env_alg::Alg
    ) where {Alg <: PEPSKit.NeighbourEnv}
    @info "Testing $(typeof(env_alg))"
    for row in 1:Nr, col in 1:Nc
        cp1 = PEPSKit._next(col, Nc)
        A, B = state.A[row, col], state.A[row, cp1]
        X, a, b, Y = PEPSKit._qr_bond(A, B)
        @tensor ab[DX DY; da db] := a[DX da D] * b[D db DY]
        benv = PEPSKit.bondenv_ntu(row, col, X, Y, state, env_alg)
        # this is a result of `_qr_bond`
        @assert [isdual(space(benv, ax)) for ax in 1:numind(benv)] == [0, 0, 1, 1]
        # NTU bond environments are exact and should be positive definite
        @test benv' ≈ benv
        benv = project_hermitian(benv)
        nrm1 = PEPSKit.inner_prod(benv, ab, ab)
        @test nrm1 ≈ real(nrm1)
        D, U = eigh_full(benv)
        @test minimum(D.data) >= 0
        # gauge fixing
        Z = PEPSKit.sdiag_pow(D, 0.5) * U'
        @assert benv ≈ Z' * Z
        Z2, a2, b2, (Linv, Rinv) = PEPSKit.fixgauge_benv(Z, a, b)
        benv2 = Z2' * Z2
        # gauge fixing should reduce condition number
        cond = LinearAlgebra.cond(benv)
        cond2 = LinearAlgebra.cond(benv2)
        @test cond2 <= cond
        @info "benv cond number: (gauge-fixed) $(cond2) ≤ $(cond) (initial)"
        # verify gauge transformation of X, Y
        @tensor a2b2[DX DY; da db] := a2[DX da D] * b2[D db DY]
        nrm2 = PEPSKit.inner_prod(benv2, a2b2, a2b2)
        X2, Y2 = PEPSKit._fixgauge_benvXY(X, Y, Linv, Rinv)
        benv3 = PEPSKit.bondenv_ntu(row, col, X2, Y2, state, env_alg)
        benv3 *= norm(benv2, Inf)
        nrm3 = PEPSKit.inner_prod(benv3, a2b2, a2b2)
        @test benv2 ≈ benv3
        @test nrm1 ≈ nrm2 ≈ nrm3
    end
    return
end

@testset "NTU bond environment ($(sym))" for sym in [U1Irrep, FermionParity]
    Pspace = Vect[sym](0 => 1, 1 => 1)
    V2 = Vect[sym](0 => 4, 1 => 1)
    V3 = Vect[sym](0 => 3, 1 => 2)
    V4 = Vect[sym](0 => 2, 1 => 2)
    V5 = Vect[sym](0 => 2, 1 => 3)
    Pspaces = fill(Pspace, (Nr, Nc))
    Nspaces = [V2 V2; V4 V4]
    Espaces = [V3 V5; V5 V3]

    for state in [
            InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces),
            InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces),
        ]
        normalize!.(state.A, Inf)
        for env_alg in (NNEnv(), NNpEnv(), NNNEnv())
            test_ntu_env(state, env_alg)
        end
    end
end
