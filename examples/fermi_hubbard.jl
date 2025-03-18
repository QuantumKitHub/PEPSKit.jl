using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

using MPSKit: add_physical_charge

## The Fermi-Hubbard model with fℤ₂ ⊠ U1 symmetry, at large U and half filling

# reference: https://www.osti.gov/servlets/purl/1565498
# energy should end up at E_ref ≈ 4 * -0.5244140625 = -2.09765625, but takes a lot of time
E_ref = -2.0

# parameters
t = 1.0
U = 8.0
lattice = InfiniteSquare(2, 2)
fermion = fℤ₂
particle_symmetry = U1Irrep
spin_symmetry = Trivial
S = fermion ⊠ particle_symmetry # symmetry sector

# spaces
D = 1
Vpeps = Vect[S]((0, 0) => 2 * D, (1, 1) => D, (1, -1) => D)
χ = 1
Venv = Vect[S](
    (0, 0) => 4 * χ, (1, -1) => 2 * χ, (1, 1) => 2 * χ, (0, 2) => χ, (0, -2) => χ
)
Saux = S((1, -1)) # impose half filling

# algorithms
boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=100, ls_maxiter=2, ls_maxfg=2)
reuse_env = true

# Hamiltonian
H0 = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, lattice; t, U)
H = add_physical_charge(H0, fill(Saux, size(H0.lattice)...))
Pspaces = H.lattice

# initialize state
Nspaces = fill(Vpeps, size(lattice)...)
Espaces = fill(Vpeps, size(lattice)...)
Random.seed!(2928528937)
ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env₀ = CTMRGEnv(ψ₀, Venv)
env₀, = leading_boundary(env₀, ψ₀; boundary_alg...)

# optimize
ψ, env, E, info = fixedpoint(
    H, ψ₀, env₀; boundary_alg, gradient_alg, optimizer_alg, reuse_env
)
@test E < E_ref
