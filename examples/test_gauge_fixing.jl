using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit
using PEPSKit

# Initialize InfinitePEPS with random & complex entries by default
function init_peps(d, D, Lx, Ly, finit=randn, dtype=ComplexF64)
    Pspaces = fill(ℂ^d, Lx, Ly)
    Nspaces = fill(ℂ^D, Lx, Ly)
    Espaces = fill(ℂ^D, Lx, Ly)
    return InfinitePEPS(finit, dtype, Pspaces, Nspaces, Espaces)
end

# Initialize PEPS and environment
χbond = 2
χenv = 20
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2)
ψ = init_peps(2, χbond, 2, 2)
env = leading_boundary(ψ, ctmalg, CTMRGEnv(ψ; Venv=ℂ^χenv))

println("\nBefore gauge-fixing:")
env′, = PEPSKit.ctmrg_iter(ψ, env, ctmalg)
PEPSKit.check_elementwise_convergence(env, env′)

println("\nAfter gauge-fixing:")
envfix = PEPSKit.gauge_fix(env, env′);
PEPSKit.check_elementwise_convergence(env, envfix)