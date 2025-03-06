using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

using MPSKit: add_physical_charge

function naive_normalize(state::InfinitePEPS)
    normalized_tensors = map(PEPSKit.unitcell(state)) do tensor
        return tensor / norm(tensor)
    end
    return InfinitePEPS(normalized_tensors)
end

## The Fermi-Hubbard model with fℤ₂ ⊠ U1 symmetry, at large U and half filling

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
χ = 2
Venv = Vect[S](
    (0, 0) => 4 * χ, (1, -1) => 2 * χ, (1, 1) => 2 * χ, (0, 2) => χ, (0, -2) => χ
)
Saux = S((1, -1)) # impose half filling

# algorithms
boundary_alg = SimultaneousCTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=100, verbosity=2
)
gradient_alg = EigSolver(;
    solver=Arnoldi(; tol=1e-6, maxiter=10, eager=true), iterscheme=:diffgauge
)
optimization_alg = LBFGS(; gradtol=1e-4, verbosity=3, maxiter=50, ls_maxiter=2, ls_maxfg=2)
pepsopt_alg = PEPSOptimize(;
    boundary_alg=boundary_alg,
    optimizer=optimization_alg,
    gradient_alg=gradient_alg,
    reuse_env=true,
)

# Hamiltonian
H0 = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, lattice; t, U)
H = add_physical_charge(H_t, fill(Saux, size(H_t.lattice)...))
Pspaces = H.lattice

# initialize state
Nspaces = fill(Vpeps, size(lattice)...)
Espaces = fill(Vpeps, size(lattice)...)
Random.seed!(2928528935)
ψ₀ = naive_normalize(InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces))
env₀ = CTMRGEnv(ψ₀, Venv)
env₀, = leading_boundary(env₀, ψ₀, boundary_alg)

# optimize
ψ, env, E, info = fixedpoint(H, ψ₀, env₀, pepsopt_alg)
# should eventually end up at
# TODO: actuatlly test something
