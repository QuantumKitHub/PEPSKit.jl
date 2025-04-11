using Random
using TensorKit, PEPSKit
Random.seed!(2394823948)

# square lattice Heisenberg Hamiltonian, sublattice rotated to fit on a single-site unit cell
H = heisenberg_XYZ(InfiniteSquare())

# parameters and algorithms
Dbond = 2
χenv = 16
boundary_alg = (; tol=1e-10, trscheme=(; alg=:fixedspace))
optimizer_alg = (; alg=:lbfgs, tol=1e-4, maxiter=100, lbfgs_memory=16)

# initialize PEPS and environment
peps₀ = InfinitePEPS(2, Dbond)
env₀, = leading_boundary(CTMRGEnv(peps₀, ℂ^χenv), peps₀; boundary_alg...)

# ground state search
peps, env, E, = fixedpoint(H, peps₀, env₀; boundary_alg, optimizer_alg, verbosity=1)
@show E
