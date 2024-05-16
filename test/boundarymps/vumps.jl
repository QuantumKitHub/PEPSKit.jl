using Test
using Random
using PEPSKit
using TensorKit
using MPSKit
using LinearAlgebra

Random.seed!(29384293742893)

# (1, 1) unit cell

psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))

T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
mps = PEPSKit.initializeMPS(T, [ComplexSpace(20)])

mps, envs, ϵ = leading_boundary(mps, T, VUMPS())
N = abs(sum(expectation_value(mps, T)))

ctm = leading_boundary(
    psi, CTMRG(; verbosity=1, fixedspace=true), CTMRGEnv(psi; Venv=ComplexSpace(20))
)
N´ = abs(norm(psi, ctm))

@test N ≈ N´ atol = 1e-3


# (2, 2) unit cell

psi2 = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
T2 = PEPSKit.TransferPEPSMultiline(psi2, 1)

mps2 = PEPSKit.initializeMPS(T2, fill(ComplexSpace(20), 2, 2))
mps2, envs2, ϵ = leading_boundary(mps2, T2, VUMPS())
N2 = abs(prod(expectation_value(mps2, T2)))

ctm2 = leading_boundary(
    psi2, CTMRG(; verbosity=1, fixedspace=true), CTMRGEnv(psi2; Venv=ComplexSpace(20))
)
N2´ = abs(norm(psi2, ctm2))

@test N ≈ N´ rtol = 1e-3


# PEPO contraction run through

function ising_pepo(beta; unitcell=(1, 1, 1))
    t = ComplexF64[exp(beta) exp(-beta); exp(-beta) exp(beta)]
    q = sqrt(t)

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    return InfinitePEPO(O; unitcell)
end

O = ising_pepo(1)
T3 = InfiniteTransferPEPO(psi, O, 1, 1)

mps3 = PEPSKit.initializeMPS(T3, [ComplexSpace(10)])
mps3, envs3, ϵ = leading_boundary(mps3, T3, VUMPS())
f = abs(prod(expectation_value(mps3, T3)))

nothing
