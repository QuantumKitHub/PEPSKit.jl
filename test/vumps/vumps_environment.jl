using Test
using Random
using PEPSKit
using PEPSKit: initial_A, initial_C, initial_FL, initial_FR
using PEPSKit: ρmap, getL!, getAL, getLsped, left_canonical, right_canonical
using TensorKit
using LinearAlgebra

begin "test utility"
    ds = [ℂ^2]
    Ds = [ℂ^3]
    χs = [ℂ^4]
end
@testset begin
    A = TensorMap(rand, ComplexF64, ℂ^2 * ℂ^3, ℂ^4 * ℂ^5)
    @show storagetype(typeof(A))
    # @tensor C = conj(A[1 2;3]) * A[1 2;3]
    @show space(A, 1) space(A, 2) space(A, 3)  space(A, 4) 
    # @tensor C = A'[3; 1 2] * A[1 2;3]
    # @show sqrt(C) norm(A)
end

@testset "InfiniteTransferPEPS for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    @test itp.top == ipeps.A
    @test itp.bot == [A' for A in ipeps.A]
    @test all(i -> space(i) == (d ← D * D * D' * D'), itp.top)
    @test all(i -> space(i) == (D * D * D' * D' ← d), itp.bot)
end

@testset "initialize A C for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    C = initial_C(A)
    @test size(A) == (Ni, Nj)
    @test size(C) == (Ni, Nj)
    @test all(i -> space(i) == (χ * D * D' ← χ), A)
    @test all(i -> space(i) == (χ ← χ), C)
end

@testset "getL!, getAL and getLsped for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    C = initial_C(A)
    C = ρmap(C, A)

    L = getL!(A, C)
    
    @test all(i -> all(j -> real(j) > 0, diag(i).values[]), L)
    @test all(i -> all(j -> imag(j) ≈ 0, diag(i).values[]), L)
    @test all(i -> space(i) == (χ ← χ), L)

    AL, Le, λ = getAL(A, L)
    @test all(i -> space(i) == (χ * D * D' ← χ), AL)
    @test all(i -> space(i) == (χ ← χ), Le)
    @test all(map((λ, AL, Le, A, L) -> λ * AL * Le ≈ transpose(L*transpose(A, ((1,),(4,3,2))), ((1,4,3),(2,))), λ, AL, Le, A, L))

    L = getLsped(Le, A, AL)
    @test all(i -> space(i) == (χ ← χ), L)
    @test all(i -> all(j -> real(j) > 0, diag(i).values[]), L)
    @test all(i -> all(j -> imag(j) ≈ 0, diag(i).values[]), L)
end

@testset "canonical form for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)

    AL, L, λ = left_canonical(A)
    @test all(i -> space(i) == (χ * D * D' ← χ), AL)
    @test all(i -> space(i) == (χ ← χ), L)
    @test all(AL -> (AL' * AL ≈ isomorphism(χ, χ)), AL)
    @test all(map((A, AL, L, λ) -> λ * AL * L ≈ transpose(L*transpose(A, ((1,),(4,3,2))), ((1,4,3),(2,))), A, AL, L, λ))

    R, AR, λ = right_canonical(A)
    @test all(i -> space(i) == (χ ← D' * D * χ), AR)
    @test all(i -> space(i) == (χ ← χ), R)
    @test all(AR -> (AR * AR' ≈ isomorphism(χ, χ)), AR)
    @test all(map((A, R, AR, λ) -> transpose(λ * R * AR, ((1,2,3),(4,))) ≈ A * R, A, R, AR, λ))
end

@testset "initialize A C for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    FL = initial_FL(AL, itp)
    @test all(i -> space(i) == (χ' * D * D' ← χ), FL)

    FR = initial_FR(AR, itp)
    @test all(i -> space(i) == (χ * D' * D ← χ'), FR)
end









# @testset "(2, 2) PEPS" begin
#     psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
#     T = PEPSKit.TransferPEPSMultiline(psi, 1)

#     mps = PEPSKit.initializeMPS(T, fill(ComplexSpace(20), 2, 2))
#     mps, envs, ϵ = leading_boundary(mps, T, vumps_alg)
#     N = abs(prod(expectation_value(mps, T)))

#     ctm = leading_boundary(CTMRGEnv(psi, ComplexSpace(20)), psi, CTMRG(; verbosity=1))
#     N´ = abs(norm(psi, ctm))

#     @test N ≈ N´ rtol = 1e-2
# end

# @testset "PEPO runthrough" begin
#     function ising_pepo(beta; unitcell=(1, 1, 1))
#         t = ComplexF64[exp(beta) exp(-beta); exp(-beta) exp(beta)]
#         q = sqrt(t)

#         O = zeros(2, 2, 2, 2, 2, 2)
#         O[1, 1, 1, 1, 1, 1] = 1
#         O[2, 2, 2, 2, 2, 2] = 1
#         @tensor o[-1 -2; -3 -4 -5 -6] :=
#             O[1 2; 3 4 5 6] *
#             q[-1; 1] *
#             q[-2; 2] *
#             q[-3; 3] *
#             q[-4; 4] *
#             q[-5; 5] *
#             q[-6; 6]

#         O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

#         return InfinitePEPO(O; unitcell)
#     end

#     psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
#     O = ising_pepo(1)
#     T = InfiniteTransferPEPO(psi, O, 1, 1)

#     mps = PEPSKit.initializeMPS(T, [ComplexSpace(10)])
#     mps, envs, ϵ = leading_boundary(mps, T, vumps_alg)
#     f = abs(prod(expectation_value(mps, T)))
# end
