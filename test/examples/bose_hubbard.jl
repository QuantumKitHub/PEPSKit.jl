using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

using MPSKit: add_physical_charge

## The Mott-insulating phase of the Bose-Hubbard model at uniform unit filling

# reference value: cylinder-MPS result for circumference Ly=7: E = -0.273284888
E_ref = -0.27

# parameters
t = 1.0
U = 30.0
cutoff = 2
mu = 0.0
symmetry = U1Irrep
n = 1
lattice = InfiniteSquare(1, 1)

# spaces
Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1)
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)

# algorithms
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, verbosity = 2, trscheme = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, maxiter = 10, alg = :eigsolver, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, verbosity = 3, maxiter = 25, ls_maxiter = 2, ls_maxfg = 2)
reuse_env = true

# Hamiltonian
H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff, t, U, n)
Pspaces = physicalspace(H)

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
@test E < E_ref
