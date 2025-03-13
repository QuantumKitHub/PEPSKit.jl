using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

using MPSKit: add_physical_charge

## The Mott-insulating phase of the Bose-Hubbard model at uniform unit filling

# parameters
t = 1.0
U = 20.0
cutoff = 2
mu = 0.0
filling = 1
symmetry = U1Irrep
lattice = InfiniteSquare()

# spaces
Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1)
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)

# algorithms
boundary_alg = SimultaneousCTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=100, verbosity=2
)
gradient_alg = EigSolver(;
    solver_alg=Arnoldi(; tol=1e-6, maxiter=10, eager=true), iterscheme=:diffgauge
)
optimizer_alg = LBFGS(; gradtol=1e-4, verbosity=3, maxiter=40, ls_maxiter=2, ls_maxfg=2)
pepsopt_alg = PEPSOptimize(; boundary_alg, optimizer_alg, gradient_alg, reuse_env=true)

# Hamiltonian
H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff=cutoff, t=t, U=U, n=filling)
Pspaces = H.lattice

# initialize state
Nspaces = fill(Vpeps, size(lattice)...)
Espaces = fill(Vpeps, size(lattice)...)
Random.seed!(2928528935)
ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env₀ = CTMRGEnv(ψ₀, Venv)
env₀, = leading_boundary(env₀, ψ₀, boundary_alg)

# optimize
ψ, env, E, info = fixedpoint(H, ψ₀, env₀, pepsopt_alg)
@test E < -0.424
# converging to E ≈ -0.4242...
# TODO: find a benchmark value to compare to
