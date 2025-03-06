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
# staggered auxiliary physical charges -> encode Néel order
Saux = [
    U1Irrep(-1//2) U1Irrep(1//2)
    U1Irrep(1//2) U1Irrep(-1//2)
]

# algorithms
boundary_alg = SimultaneousCTMRG(;
    trscheme=FixedSpaceTruncation(), tol=1e-10, miniter=3, maxiter=100, verbosity=2
)
gradient_alg = EigSolver(;
    solver=Arnoldi(; tol=1e-6, maxiter=10, eager=true), iterscheme=:diffgauge
)
optimization_alg = LBFGS(; gradtol=1e-4, verbosity=3, maxiter=50, ls_maxiter=2, ls_maxfg=4)
pepsopt_alg = PEPSOptimize(;
    boundary_alg=boundary_alg,
    optimizer=optimization_alg,
    gradient_alg=gradient_alg,
    reuse_env=true,
)

# Hamiltonian
H0 = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin)
H = add_physical_charge(H0, Saux)
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
@test E < -0.668
# ends up at E = -0.669..., but takes a while
