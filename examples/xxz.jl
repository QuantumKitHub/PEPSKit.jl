using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

using MPSKit: add_physical_charge

## Néel order in the XXZ model

# parameters
J = 1.0
Delta = 1.0
spin = 1//2
symmetry = U1Irrep
lattice = InfiniteSquare(2, 2)

# spaces
Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1)
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)
# staggered auxiliary physical charges -> encode Néel order directly in the ansatz
Saux = [
    U1Irrep(-1//2) U1Irrep(1//2)
    U1Irrep(1//2) U1Irrep(-1//2)
]

# algorithms
boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=100, ls_maxiter=2, ls_maxfg=2)
reuse_env = true

# Hamiltonian
H0 = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin)
H = add_physical_charge(H0, Saux)
Pspaces = H.lattice

# initialize state
Nspaces = fill(Vpeps, size(lattice)...)
Espaces = fill(Vpeps, size(lattice)...)
Random.seed!(2928528935)
ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env₀ = CTMRGEnv(ψ₀, Venv)
env₀, = leading_boundary(env₀, ψ₀; boundary_alg...)

# optimize
ψ, env, E, info = fixedpoint(
    H, ψ₀, env₀; boundary_alg, gradient_alg, optimizer_alg, reuse_env
)
@test E < -0.666
# ends up at E = -0.669..., but takes a while
