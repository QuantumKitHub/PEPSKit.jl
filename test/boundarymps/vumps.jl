using Test
using Random
using PEPSKit
using TensorKit
using MPSKit
using LinearAlgebra

Random.seed!(29384293742893)

@testset "(1, 1) PEPS" begin
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))

    T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
    mps = PEPSKit.initializeMPS(T, [ComplexSpace(20)])

    mps, envs, ϵ = leading_boundary(mps, T, VUMPS())
    N = abs(sum(expectation_value(mps, T)))

    ctm = leading_boundary(
        CTMRGEnv(psi; Venv=ComplexSpace(20)), psi, CTMRG(; verbosity=1, fixedspace=true)
    )
    N´ = abs(norm(psi, ctm))

    @test N ≈ N´ atol = 1e-3
end

@testset "(2, 2) PEPS" begin
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
    T = PEPSKit.TransferPEPSMultiline(psi, 1)

    mps = PEPSKit.initializeMPS(T, fill(ComplexSpace(20), 2, 2))
    mps, envs, ϵ = leading_boundary(mps, T, VUMPS())
    N = abs(prod(expectation_value(mps, T)))

    ctm = leading_boundary(
        CTMRGEnv(psi; Venv=ComplexSpace(20)), psi, CTMRG(; verbosity=1, fixedspace=true)
    )
    N´ = abs(norm(psi, ctm))

    @test N ≈ N´ rtol = 1e-3
end

@testset "PEPO runthrough" begin
    function ising_pepo(beta; unitcell=(1, 1, 1))
        t = ComplexF64[exp(beta) exp(-beta); exp(-beta) exp(beta)]
        q = sqrt(t)

        O = zeros(2, 2, 2, 2, 2, 2)
        O[1, 1, 1, 1, 1, 1] = 1
        O[2, 2, 2, 2, 2, 2] = 1
        @tensor o[-1 -2; -3 -4 -5 -6] :=
            O[1 2; 3 4 5 6] *
            q[-1; 1] *
            q[-2; 2] *
            q[-3; 3] *
            q[-4; 4] *
            q[-5; 5] *
            q[-6; 6]

        O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

        return InfinitePEPO(O; unitcell)
    end

    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))
    O = ising_pepo(1)
    T = InfiniteTransferPEPO(psi, O, 1, 1)

    mps = PEPSKit.initializeMPS(T, [ComplexSpace(10)])
    mps, envs, ϵ = leading_boundary(mps, T, VUMPS())
    f = abs(prod(expectation_value(mps, T)))
end
