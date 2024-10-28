using Test
using Random
using PEPSKit
using PEPSKit: initial_A, initial_C, initial_FL, initial_FR
using PEPSKit: ρmap, getL!, getAL, getLsped, _to_tail, _to_front, left_canonical, right_canonical
using PEPSKit: leftenv, FLmap, rightenv, FRmap, ACenv, ACmap, Cenv, Cmap, leftCenv, Lmap, rightCenv, Rmap 
using PEPSKit: LRtoC, ALCtoAC, ACCtoALAR
using TensorKit
using LinearAlgebra

begin "test utility"
    ds = [ℂ^2]
    Ds = [ℂ^3]
    χs = [ℂ^4]
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
    @test all(map((A, AL, L, λ) -> λ * AL * L ≈ _to_front(L * _to_tail(A)), A, AL, L, λ))

    R, AR, λ = right_canonical(A)
    @test all(i -> space(i) == (χ * D * D' ← χ), AR)
    @test all(i -> space(i) == (χ ← χ), R)
    @test all(AR -> (_to_tail(AR) * _to_tail(AR)' ≈ isomorphism(χ, χ)), AR)
    @test all(map((A, R, AR, λ) -> _to_front(λ * R * _to_tail(AR)) ≈ A * R, A, R, AR, λ))
end

@testset "initialize FL FR for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    FL = initial_FL(AL, itp)
    @test all(i -> space(i) == (χ * D' * D ← χ), FL)

    FR = initial_FR(AR, itp)
    @test all(i -> space(i) == (χ * D * D' ← χ), FR)
end

@testset "leftenv and rightenv for unitcell $Ni x $Nj" for Ni in 1:3, Nj in 1:3, (d, D, χ) in zip(ds, Ds, χs), ifobs in [true, false]
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, FL = leftenv(AL, adjoint.(AL), itp; ifobs)
    λR, FR = rightenv(AR, adjoint.(AR), itp; ifobs)

    @test all(i -> space(i) == (χ * D' * D ← χ), FL)
    @test all(i -> space(i) == (χ * D * D' ← χ), FR)

    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        @test λL[i] * FL[i,:] ≈ FLmap(FL[i,:], AL[i,:], adjoint.(AL)[ir,:], itp.top[i,:], itp.bot[i,:]) rtol = 1e-12
        @test λR[i] * FR[i,:] ≈ FRmap(FR[i,:], AR[i,:], adjoint.(AR)[ir,:], itp.top[i,:], itp.bot[i,:]) rtol = 1e-12
    end
end

@testset "ACenv and Cenv for unitcell $Ni x $Nj" for Ni in 1:3, Nj in 1:3, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, FL = leftenv(AL, adjoint.(AL), itp)
    λR, FR = rightenv(AR, adjoint.(AR), itp)

     C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, FR, itp)
     λC,  C =  Cenv( C, FL, FR) 
    @test all(i -> space(i) == (χ * D * D' ← χ), AC)
    @test all(i -> space(i) == (χ ← χ),  C)

    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        @test λAC[j] * AC[:,j] ≈ ACmap(AC[:,j], FL[:,j], FR[:,j], itp.top[:,j], itp.bot[:,j]) rtol = 1e-12
        @test  λC[j] *  C[:,j] ≈  Cmap( C[:,j], FL[:,jr], FR[:,j]) rtol = 1e-10
    end
end

@testset "ACCtoALAR for unitcell $Ni x $Nj" for Ni in 1:3, Nj in 1:3, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, FL = leftenv(AL, adjoint.(AL), itp)
    λR, FR = rightenv(AR, adjoint.(AR), itp)

     C = LRtoC(L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, FR, itp)
     λC,  C =  Cenv( C, FL, FR) 

    AL, AR, errL, errR = ACCtoALAR(AC, C)
    @test all(i -> space(i) == (χ * D * D' ← χ), AL)
    @test all(i -> space(i) == (χ * D * D' ← χ), AR)
    @test all(AL -> (AL' * AL ≈ isomorphism(χ, χ)), AL)
    @test all(AR -> (_to_tail(AR) * _to_tail(AR)' ≈ isomorphism(χ, χ)), AR)
    @test errL isa Real
    @test errR isa Real
end

@testset "leftCenv and rightCenv for unitcell $Ni x $Nj" for Ni in 1:3, Nj in 1:3, (d, D, χ) in zip(ds, Ds, χs), ifobs in [true, false]
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    A = initial_A(itp, χ)
    AL, L, λ = left_canonical(A)
    R, AR, λ = right_canonical(A)

    λL, L =  leftCenv(AL, adjoint.(AL); ifobs)
    λR, R = rightCenv(AR, adjoint.(AR); ifobs) 
    @test all(i -> space(i) == (χ ← χ), R)
    @test all(i -> space(i) == (χ ← χ), L)

    for i in 1:Ni
        ir = ifobs ? mod1(Ni + 2 - i, Ni) : i
        @test λL[i] * L[i,:] ≈ Lmap(L[i,:], AL[i,:], adjoint.(AL)[ir,:]) rtol = 1e-12
        @test λR[i] * R[i,:] ≈ Rmap(R[i,:], AR[i,:], adjoint.(AR)[ir,:]) rtol = 1e-12
    end
end