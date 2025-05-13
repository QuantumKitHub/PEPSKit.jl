using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Nr, Nc = 2, 2
Random.seed!(20)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
V2 = Vect[FermionParity](0 => 4, 1 => 1)
V3 = Vect[FermionParity](0 => 3, 1 => 2)
V4 = Vect[FermionParity](0 => 2, 1 => 2)
V5 = Vect[FermionParity](0 => 2, 1 => 3)
Pspaces = fill(Pspace, (Nr, Nc))
Nspaces = [V2 V2; V4 V4]
Espaces = [V3 V5; V5 V3]

peps = InfiniteWeightPEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
normalize!.(peps.vertices, Inf)
for wt in peps.weights
    wt.data[:] = normalize(rand(length(wt.data)), Inf)
end
for add_bwt in (true, false)
    for env_alg in (NTUEnvNN(; add_bwt), NTUEnvNNN(; add_bwt), NTUEnvNNNp(; add_bwt))
        @info "Testing $(typeof(env_alg))"
        for row in 1:Nr, col in 1:Nc
            cp1 = PEPSKit._next(col, Nc)
            A, B = peps.vertices[row, col], peps.vertices[row, cp1]
            X, a, b, Y = PEPSKit._qr_bond(A, B)
            @tensor ab[DX DY; da db] := a[DX da D] * b[D db DY]
            benv = PEPSKit.bondenv_ntu(row, col, X, Y, peps, env_alg)
            # NTU bond environments are constructed exactly
            # and should be positive definite
            @test benv' ≈ benv
            @assert [isdual(space(benv, ax)) for ax in 1:numind(benv)] == [0, 0, 1, 1]
            nrm1 = PEPSKit.inner_prod(benv, ab, ab)
            @test nrm1 ≈ real(nrm1)
            D, U = eigh(benv)
            @test all(all(x -> x >= 0, diag(b)) for (k, b) in blocks(D))
            @assert benv ≈ U * D * U'
        end
    end
end
