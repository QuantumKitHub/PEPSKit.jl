using Test
using Random
using PEPSKit
using TensorKit
using MPSKit
using LinearAlgebra

Random.seed!(29384293742893)
psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))

T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
mps = PEPSKit.initializeMPS(T, [ComplexSpace(8)])

mps, envs, ϵ = leading_boundary(mps, T, VUMPS(; verbosity=2))
N = sum(expectation_value(mps, T))

ctm = leading_boundary(
    psi, CTMRG(; verbosity=1, fixedspace=true), CTMRGEnv(psi; Venv=ComplexSpace(8))
)
N2 = norm(psi, ctm)

@test N ≈ N2 atol = 1e-3
