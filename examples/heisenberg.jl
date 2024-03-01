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

    return H
end

# Initialize InfinitePEPS with random & complex entries by default
function init_peps(d, D, Lx, Ly, finit=randn, dtype=ComplexF64)
    Pspaces = fill(ℂ^d, Lx, Ly)
    Nspaces = fill(ℂ^D, Lx, Ly)
    Espaces = fill(ℂ^D, Lx, Ly)
    return InfinitePEPS(finit, dtype, Pspaces, Nspaces, Espaces)
end

# Parameters
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
χbond = 2
χenv = 20
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2)
optalg = PEPSOptimize{LinSolve}(;
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2),
    fpgrad_tol=1e-6,
    fpgrad_maxiter=100,
    verbosity=2,
)

# Ground state search
ψinit = init_peps(2, χbond, 1, 1)
envinit = leading_boundary(ψinit, ctmalg, CTMRGEnv(ψinit; Venv=ℂ^χenv))
result = groundsearch(H, ctmalg, optalg, ψinit, envinit)
@show result.E₀
