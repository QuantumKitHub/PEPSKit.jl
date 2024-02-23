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
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-12, miniter=4, maxiter=100, verbosity=1)
ψ = init_peps(2, χbond, 1, 1)
env, = leading_boundary(ψinit, ctmalg, CTMRGEnv(ψinit; Venv=ℂ^χenv))

# Compute CTM gradient in four different ways (set reuse_env=false to not mutate environment)
println("\nFP gradient using naive AD:")
alg_naive = PEPSOptimize{NaiveAD}(; verbosity=2, reuse_env=false)
@time _, g_naive = PEPSKit.ctmrg_gradient((ψ, env), H, ctmalg, alg_naive)
g_naive = InfinitePEPS(g_naive...)  # Convert NamedTuple to InfinitePEPS

println("\nFP gradient using explicit evaluation of the geometric sum:")
alg_geomsum = PEPSOptimize{GeomSum}(;
    fpgrad_tol=1e-6, fpgrad_maxiter=100, verbosity=2, reuse_env=false
)
@time _, g_geomsum = PEPSKit.ctmrg_gradient((ψ, env), H, ctmalg, alg_geomsum)

println("\nFP gradient using manual iteration of the linear problem:")
alg_maniter = PEPSOptimize{ManualIter}(;
    fpgrad_tol=1e-6, fpgrad_maxiter=100, verbosity=2, reuse_env=false
)
@time _, g_maniter = PEPSKit.ctmrg_gradient((ψ, env), H, ctmalg, alg_maniter)

println("\nFP gradient using GMRES to solve the linear problem:")
alg_linsolve = PEPSOptimize{LinSolve}(;
    fpgrad_tol=1e-6, fpgrad_maxiter=100, verbosity=2, reuse_env=false
)
@time _, g_linsolve = PEPSKit.ctmrg_gradient((ψ, env), H, ctmalg, alg_linsolve)

@show norm(g_geomsum - g_naive) / norm(g_naive)
@show norm(g_maniter - g_naive) / norm(g_naive)
@show norm(g_linsolve - g_naive) / norm(g_naive)