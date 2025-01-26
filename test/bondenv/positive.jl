using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Nr, Nc = 2, 2
Random.seed!(0)
Pspace = ℂ[FermionParity](0 => 1, 1 => 1)
V2 = ℂ[FermionParity](0 => 1, 1 => 1)
V3 = ℂ[FermionParity](0 => 1, 1 => 2)
V4 = ℂ[FermionParity](0 => 2, 1 => 2)
V5 = ℂ[FermionParity](0 => 3, 1 => 2)
W1 = ℂ[FermionParity](0 => 2, 1 => 3)
W2 = ℂ[FermionParity](0 => 4, 1 => 1)
Pspaces = fill(Pspace, (Nr, Nc))
Nspaces = [V2 V2; V4 V4]
Espaces = [V3 V5; V5 V3]

peps = InfiniteWeightPEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
for I in CartesianIndices(peps.vertices)
    peps.vertices[I] /= norm(peps.vertices[I], Inf)
end
for env_alg in (NTUEnvNN(), NTUEnvNNN(), NTUEnvNNNp())
    for row in 1:Nr, col in 1:Nc
        cp1 = PEPSKit._next(col, Nc)
        A, B = peps.vertices[row, col], peps.vertices[row, cp1]
        X = Tensor(randn, ComplexF64, space(A, 2) ⊗ W1' ⊗ space(A, 4) ⊗ space(A, 5))
        Y = Tensor(randn, ComplexF64, space(B, 2) ⊗ space(B, 3) ⊗ space(B, 4) ⊗ W2')
        env = PEPSKit.bondenv_ntu(row, col, X, Y, peps, env_alg)
        # env should be Hermitian
        @test env' ≈ env
        # env should be positive definite
        D, U = eigh(env)
        @test all(all(x -> x >= 0, diag(b)) for (k, b) in blocks(D))
    end
end
