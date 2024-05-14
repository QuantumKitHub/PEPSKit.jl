using Test
using PEPSKit
using TensorKit
using MPSKit
using LinearAlgebra

psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))

T = PEPSKit.InfiniteTransferPEPS(psi, 1, 1)
mps = PEPSKit.initializeMPS(T, [ComplexSpace(8)])

mps, envs, ϵ = leading_boundary(mps, T, VUMPS())
N = sum(expectation_value(mps, T))

ctm = leading_boundary(
    psi, CTMRG(; verbosity=2, fixedspace=true), CTMRGEnv(psi; Venv=ComplexSpace(8))
)
N2 = norm(psi, ctm)

@test N ≈ N2 atol = 1e-3
