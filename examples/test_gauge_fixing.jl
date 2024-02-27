using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit
using PEPSKit

# Square lattice Heisenberg Hamiltonian
function square_lattice_heisenberg(; Jx=-1.0, Jy=1.0, Jz=-1.0)
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

# Initialize PEPS and environment
H = square_lattice_heisenberg()
χbond = 2
χenv = 20
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2)
ψ = init_peps(2, χbond, 1, 1)
env, = leading_boundary(ψ, ctmalg, CTMRGEnv(ψ; Venv=ℂ^χenv))

println("\nBefore gauge-fixing:")
env′, = PEPSKit.ctmrg_iter(ψ, env, ctmalg)
PEPSKit.check_elementwise_conv(env, env′)

println("\nAfter gauge-fixing:")
envfix = PEPSKit.gauge_fix(env, env′)
PEPSKit.check_elementwise_conv(env, envfix)