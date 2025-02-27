using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Nr, Nc = 2, 2
Random.seed!(20)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
V2 = Vect[FermionParity](0 => 1, 1 => 1)
V3 = Vect[FermionParity](0 => 1, 1 => 2)
V4 = Vect[FermionParity](0 => 2, 1 => 2)
V5 = Vect[FermionParity](0 => 3, 1 => 2)
W1 = Vect[FermionParity](0 => 2, 1 => 3)
W2 = Vect[FermionParity](0 => 4, 1 => 1)
Pspaces = fill(Pspace, (Nr, Nc))
Nspaces = [V2 V2; V4 V4]
Espaces = [V3 V5; V5 V3]

peps = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
# The NTU bond environments are constructed exactly
# and should be positive definite
for env_alg in (NNEnv(), NNNEnv())
    @info "Testing $(typeof(env_alg))"
    for row in 1:Nr, col in 1:Nc
        cp1 = PEPSKit._next(col, Nc)
        A, B = peps.A[row, col], peps.A[row, cp1]
        X, a, b, Y = PEPSKit._qr_bond(A, B)
        benv = PEPSKit.bondenv_ntu(row, col, X, Y, peps, env_alg)
        # benv should be Hermitian
        @test benv' ≈ benv
        benv = (benv + benv') / 2
        # benv should be positive definite
        D, U = eigh(benv)
        @test all(all(x -> x >= 0, diag(b)) for (k, b) in blocks(D))
    end
end
