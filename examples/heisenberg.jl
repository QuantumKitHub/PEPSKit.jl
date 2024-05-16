using LinearAlgebra
using TensorKit, OptimKit
using PEPSKit, KrylovKit

# Square lattice Heisenberg Hamiltonian
# We use the parameters (J₁, J₂, J₃) = (-1, 1, -1) by default to capture
# the ground state in a single-site unit cell. This can be seen from
# sublattice rotating H from parameters (1, 1, 1) to (-1, 1, -1).
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    physical_space = ComplexSpace(2)
    T = ComplexF64
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return NLocalOperator{NearestNeighbor}(H / 4)
end

# Parameters
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
χbond = 2
χenv = 20
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=1)
alg = PEPSOptimize(;
    boundary_alg=ctmalg,
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
env₀ = leading_boundary(CTMRGEnv(ψ₀; Venv=ℂ^χenv), ψ₀, ctmalg)
result = fixedpoint(ψ₀, H, alg, env₀)
@show result.E
