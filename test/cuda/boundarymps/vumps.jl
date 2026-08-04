using Test
using Random
using PEPSKit
using TensorKit
using MPSKit
using LinearAlgebra
using Adapt, CUDA

Random.seed!(29384293742893)

const vumps_alg = VUMPS(;
    tol = 1.0e-6, alg_eigsolve = MPSKit.Defaults.alg_eigsolve(; ishermitian = false), verbosity = 2
)

@testset "(1, 1) PEPS" begin
    Vpeps = ComplexSpace(2)
    psi = adapt(CuArray, InfinitePEPS(Vpeps, Vpeps))

    T = adapt(CuArray, PEPSKit.InfiniteTransferPEPS(psi, 1, 1))
    foreach(V -> (@test V == Vpeps ⊗ Vpeps'), physicalspace(T))
    mps = adapt(CuArray, initialize_mps(T, [ComplexSpace(20)]))

    mps, env, ϵ = leading_boundary(mps, T, vumps_alg)
    N = abs(sum(expectation_value(mps, T)))

    mps2, = changebonds(mps, T, OptimalExpand(; trunc = truncrank(30)))
    mps2, env2, ϵ = leading_boundary(mps2, T, vumps_alg)
    N2 = abs(sum(expectation_value(mps2, T)))
    @test N ≈ N2 rtol = 1.0e-2

    ctm, = leading_boundary(CTMRGEnv(psi, ComplexSpace(20)), psi)
    N´ = abs(norm(psi, ctm))

    @test N ≈ N´ atol = 1.0e-3
end

@testset "(2, 2) PEPS" begin
    Vpeps = ComplexSpace(2)
    psi = adapt(CuArray, InfinitePEPS(Vpeps, Vpeps; unitcell = (2, 2)))
    T = adapt(CuArray, PEPSKit.MultilineTransferPEPS(psi, 1))
    @test storagetype(T) <: CuArray
    # foreach(V -> (@test V == Vpeps ⊗ Vpeps'), physicalspace(T)) # TODO: MPSKit.physicalspace(::MultilineMPO) isn't implemented...
    mps = adapt(CuArray, initialize_mps(rand, scalartype(T), T, fill(ComplexSpace(20), 2, 2)))
    @test storagetype(mps) <: CuArray
    mps, env, ϵ = leading_boundary(mps, T, vumps_alg)
    N = abs(prod(expectation_value(mps, T)))

    ctm, = leading_boundary(CTMRGEnv(psi, ComplexSpace(20)), psi)
    N´ = abs(norm(psi, ctm))

    @test N ≈ N´ rtol = 1.0e-2
end

#=@testset "Fermionic PEPS" begin
    D = Vect[fℤ₂](0 => 1, 1 => 1)
    d = Vect[fℤ₂](0 => 1, 1 => 1)
    χ = Vect[fℤ₂](0 => 10, 1 => 10)

    psi = adapt(CuArray, InfinitePEPS(D, d; unitcell = (1, 1)))
    n = adapt(CuArray, InfiniteSquareNetwork(psi))
    T = adapt(CuArray, InfiniteTransferPEPS(psi, 1, 1))
    foreach(V -> (@test V == D ⊗ D'), physicalspace(T))

    # compare boundary MPS contraction to CTMRG contraction
    mps = adapt(CuArray, initialize_mps(T, [χ]))
    mps, env, ϵ = leading_boundary(mps, T, vumps_alg)
    N_vumps = abs(prod(expectation_value(mps, T)))

    ctm, = leading_boundary(CTMRGEnv(psi, χ), psi)
    N_ctm = abs(norm(psi, ctm))

    @test N_vumps ≈ N_ctm rtol = 1.0e-2

    # and again after blocking the local sandwiches
    n´ = adapt(CuArray, InfiniteSquareNetwork(map(PEPSKit.mpotensor, PEPSKit.unitcell(n))))
    T´ = adapt(CuArray, InfiniteMPO(map(PEPSKit.mpotensor, T.O)))
    foreach(V -> (@test V == fuse(D, D')), physicalspace(T´))

    mps´ = adapt(CuArray, InfiniteMPS(randn, ComplexF64, [physicalspace(T´, 1)], [χ]))
    mps´, env´, ϵ = leading_boundary(mps´, T´, vumps_alg)
    N_vumps´ = abs(prod(expectation_value(mps´, T´)))

    ctm´, = leading_boundary(CTMRGEnv(n´, χ), n´)
    N_ctm´ = abs(network_value(n´, ctm´))

    @show N_vumps´
    @test N_vumps´ ≈ N_vumps rtol = 1.0e-2
    @test N_vumps´ ≈ N_ctm´ rtol = 1.0e-2
end=#

@testset "PEPO runthrough" begin
    function ising_pepo(beta; unitcell = (1, 1, 1))
        t = ComplexF64[exp(beta) exp(-beta); exp(-beta) exp(beta)]
        q = sqrt(t)

        O = zeros(2, 2, 2, 2, 2, 2)
        O[1, 1, 1, 1, 1, 1] = 1
        O[2, 2, 2, 2, 2, 2] = 1
        @tensor o[-1 -2; -3 -4 -5 -6] :=
            O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

        O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

        return adapt(CuArray, InfinitePEPO(O; unitcell))
    end

    Vpepo = ComplexSpace(2)
    Vpeps = ComplexSpace(2)

    # single-layer PEPO
    O = ising_pepo(1)
    psi = adapt(CuArray, PEPSKit.initializePEPS(O, Vpeps))
    T = adapt(CuArray, InfiniteTransferPEPO(psi, O, 1, 1))
    foreach(V -> (@test V == Vpeps ⊗ Vpepo ⊗ Vpeps'), physicalspace(T))

    mps = adapt(CuArray, initialize_mps(rand, scalartype(T), T, [ComplexSpace(10)]))
    mps, env, ϵ = leading_boundary(mps, T, vumps_alg)
    f = abs(prod(expectation_value(mps, T)))

    # double-layer PEPO
    O2 = repeat(O, 1, 1, 2)
    psi2 = adapt(CuArray, initializePEPS(O2, Vpeps))
    T2 = adapt(CuArray, InfiniteTransferPEPO(psi, O2, 1, 1))
    foreach(V -> (@test V == Vpeps ⊗ Vpepo ⊗ Vpepo ⊗ Vpeps'), physicalspace(T2))

    mps2 = adapt(CuArray, initialize_mps(rand, scalartype(T2), T2, [ComplexSpace(8)]))
    mps2, env2, ϵ = leading_boundary(mps2, T2, vumps_alg)
    f = abs(prod(expectation_value(mps2, T2)))
end
