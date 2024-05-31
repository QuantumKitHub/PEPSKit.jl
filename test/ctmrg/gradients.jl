using Test
using Random
using PEPSKit
using TensorKit
using PEPSKit
using Zygote
using OptimKit
using KrylovKit

# Square lattice Heisenberg Hamiltonian
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    physical_space = ComplexSpace(2)
    T = ComplexF64
    σx = TensorMap(T[0 1; 1 0], physical_space, physical_space)
    σy = TensorMap(T[0 im; -im 0], physical_space, physical_space)
    σz = TensorMap(T[1 0; 0 -1], physical_space, physical_space)
    H = (Jx * σx ⊗ σx) + (Jy * σy ⊗ σy) + (Jz * σz ⊗ σz)
    return NLocalOperator{NearestNeighbor}(H / 4)
end

h = TensorMap(randn, Float64, ComplexSpace(2)^2, ComplexSpace(2)^2)
h += h'

# Initialize PEPS and environment
H = NLocalOperator{NearestNeighbor}(h)
χbond = 2
χenv = 8
boundary_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=1
)
tol = 1e-8

steps = -0.01:0.005:0.01
gradmodes = [nothing, GeomSum(; tol), ManualIter(; tol), KrylovKit.GMRES(; tol)]

@testset "$alg_rrule" for alg_rrule in gradmodes
    # random point, random direction
    Random.seed!(42039482038)
    dir = InfinitePEPS(2, χbond)
    psi = InfinitePEPS(2, χbond)
    env = leading_boundary(CTMRGEnv(psi; Venv=ℂ^χenv), psi, boundary_alg)

    alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
        (psi, env), dir; alpha=steps, retract=PEPSKit.my_retract, inner=PEPSKit.my_inner
    ) do (peps, envs)
        E, g = Zygote.withgradient(peps) do psi
            envs2 = PEPSKit.hook_pullback(leading_boundary, envs, psi, boundary_alg; alg_rrule)
            return costfun(psi, envs2, H)
        end
        return E, only(g)
    end

    @test dfs1 ≈ dfs2 atol = 1e-2
end
