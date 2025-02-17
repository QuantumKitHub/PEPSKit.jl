
using Test
using Random
using PEPSKit
using TensorKit
using LinearAlgebra
using QuadGK
using MPSKit

## Setup

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

## Test

# initialize
beta = 1
O = ising_pepo(1)
O2 = repeat(O, 1, 1, 2)
Random.seed!(81812781143)

# contract
psi = initializePEPS(O, ComplexSpace(2))
n1 = InfiniteSquareNetwork(psi, O)
n2 = InfiniteSquareNetwork(psi, O2)
χenv = ℂ^12
env1_0 = CTMRGEnv(n1, χenv)
env2_0 = CTMRGEnv(n2, χenv)

# cover all different flavors
ctm_styles = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs = [HalfInfiniteProjector, FullInfiniteProjector]

@testset "PEPO-trial using $ctm_style with $projector_alg" for (ctm_style, projector_alg) in
                                                               Iterators.product(
    ctm_styles, projector_algs
)
    ctm_alg = ctm_style(; maxiter=150, projector_alg)
    env1, = leading_boundary(env1_0, n1, ctm_alg)
    env2, = leading_boundary(env2_0, n2, ctm_alg)
end
