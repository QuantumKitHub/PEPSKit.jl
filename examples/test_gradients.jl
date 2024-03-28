using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit
using PEPSKit, KrylovKit

using Zygote

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

    return NLocalOperator{NearestNeighbor}(H)
end

# Initialize PEPS and environment
H = square_lattice_heisenberg()
χbond = 2
χenv = 16
boundary_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=1e-12, miniter=4, maxiter=100, verbosity=2
)
ψ = InfinitePEPS(2, χbond)
env = leading_boundary(ψ, boundary_alg, CTMRGEnv(ψ; Venv=ℂ^χenv))

# Compute CTM gradient in four different ways (set reuse_env=false to not mutate environment)
function compute_grad(gradient_alg)
    @info "FP gradient using $(gradient_alg):"
    alg = PEPSOptimize(; boundary_alg, gradient_alg, reuse_env=false)
    @time _, g = PEPSKit.ctmrg_gradient((ψ, env), H, alg)
    return g
end

g_naive = compute_grad(NaiveAD());
g_geomsum = compute_grad(GeomSum());
g_maniter = compute_grad(ManualIter());
# g_linsolve = compute_grad(KrylovKit.GMRES(; tol=1e-6));

@show norm(g_geomsum - g_naive) / norm(g_naive)
@show norm(g_maniter - g_naive) / norm(g_naive)
# @show norm(g_linsolve - g_naive) / norm(g_naive)
