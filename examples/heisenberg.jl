using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit
using PEPSKit

# Square lattice Heisenberg Hamiltonian
# Sublattice-rotate to get (1, 1, 1) → (-1, 1, -1), transformed to GS with single-site unit cell
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    Sx, Sy, Sz, _ = spinmatrices(1//2)
    Vphys = ℂ^2
    σx = TensorMap(Sx, Vphys, Vphys)
    σy = TensorMap(Sy, Vphys, Vphys)
    σz = TensorMap(Sz, Vphys, Vphys)

    @tensor H[-1 -3; -2 -4] :=
        Jx * σx[-1, -2] * σx[-3, -4] +
        Jy * σy[-1, -2] * σy[-3, -4] +
        Jz * σz[-1, -2] * σz[-3, -4]

    return NLocalOperator{NearestNeighbor}(H)
end

# Parameters
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
χbond = 2
χenv = 20
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2)
alg = PEPSOptimize{LinSolve}(;
    boundary_alg=ctmalg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2),
    fpgrad_tol=1e-6,
    fpgrad_maxiter=100,
    reuse_env=true,
    verbosity=2,
)

# Ground state search
ψ₀ = InfinitePEPS(2, χbond)
env₀ = leading_boundary(ψ₀, ctmalg, CTMRGEnv(ψ₀; Venv=ℂ^χenv))
result = fixedpoint(ψ₀, H, alg, env₀)
@show result.E
