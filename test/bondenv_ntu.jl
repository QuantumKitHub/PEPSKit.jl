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
Pspaces = fill(Pspace, (Nr, Nc))
Nspaces = [V2 V2; V4 V4]
Espaces = [V3 V5; V5 V3]

peps = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
# normalize
for I in CartesianIndices(peps.A)
    peps.A[I] /= norm(peps.A[I], Inf)
end
gate = id(Pspace) ⊗ id(Pspace)
row, col = 2, 1
cp1 = PEPSKit._next(col, Nc)
A, B = peps.A[row, col], peps.A[row, cp1]
#= 
        2   1               1             2
        | ↗                 |            ↗
    5 - A ← 3   ====>   4 - X ← 2   1 ← aR ← 3
        |                   |
        4                   3

        2   1                 2         2
        | ↗                 ↗           |
    5 ← B - 3   ====>  1 ← bL → 3   1 → Y - 3
        |                               |
        4                               4
=#
X, aR0 = leftorth(A, ((2, 4, 5), (1, 3)); alg=QRpos())
X = permute(X, (1, 4, 2, 3))
Y, bL0 = leftorth(B, ((2, 3, 4), (1, 5)); alg=QRpos())
Y = permute(Y, (1, 2, 3, 4))
env = PEPSKit.bondenv_ntu(row, col, X, Y, peps, NTUEnvNN())
# env should be Hermitian
@test env' ≈ env
# env should be positive definite
D, U = eigh(env)
println(space(D))
@test all(all(x -> x >= 0, diag(b)) for (k, b) in blocks(D))
