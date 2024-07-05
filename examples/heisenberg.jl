using LinearAlgebra
using TensorKit, OptimKit
using PEPSKit, KrylovKit

# Square lattice Heisenberg Hamiltonian
# We use the parameters (J₁, J₂, J₃) = (-1, 1, -1) by default to capture
# the ground state in a single-site unit cell. This can be seen from
# sublattice rotating H from parameters (1, 1, 1) to (-1, 1, -1).
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)

# Parameters
χbond = 2
χenv = 20
projector_alg = ProjectorAlg(; trscheme=truncdim(χenv))
ctm_alg = CTMRG(; tol=1e-10, miniter=4, maxiter=100, verbosity=1, projector_alg)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=100),
    reuse_env=true,
    verbosity=2,
)

# Ground state search
# We initialize a random PEPS with bond dimension χbond and from that converge
# a CTMRG environment with dimension χenv on the environment bonds before
# starting the optimization. The ground-state energy should approximately approach
# E/N = −0.6694421, which is a QMC estimate from https://arxiv.org/abs/1101.3281.
# Of course there is a noticable bias for small χbond and χenv.
ψ₀ = InfinitePEPS(2, χbond)
env₀ = leading_boundary(CTMRGEnv(ψ₀; Venv=ℂ^χenv), ψ₀, ctm_alg)
result = fixedpoint(ψ₀, H, opt_alg, env₀)
@show result.E
