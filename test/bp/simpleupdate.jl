using Test
using Random
using LinearAlgebra
using TensorKit
using Test
using TensorKit
using MPSKitModels: S_exchange
using PEPSKit

using Random

Random.seed!(1234)

# -------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------

elt = Float64
Dspace = ComplexSpace(2)

H = real(heisenberg_XYZ(InfiniteSquare(2, 2); Jx = 1.0, Jy = 1.0, Jz = 1.0))
peps = InfinitePEPS(randn, elt, physicalspace(H), fill(Dspace, 2, 2))

O = S_exchange();

# Compute BP messages
# -------------------
messages = BPEnv(peps)
bp_alg = BeliefPropagation(; tol = 1.0e-10, maxiter = 100)
messages, bp_error = leading_boundary(messages, peps, bp_alg)

E₀ = PEPSKit.expectation_value(peps, H, messages)

dt = 0.01
circuit = PEPSKit.trotterize(H, dt)

su_alg = SimpleUpdate(; trunc = truncrank(10))

for _ in 1:100
    normalize!.(peps.A)
    normalize!.(messages.messages)
    peps, messages, ϵ = PEPSKit.apply!(peps, circuit, su_alg, messages)
end

E = PEPSKit.expectation_value(peps′, H, messages′)

leading_boundary(messages′, peps′, bp_alg)
