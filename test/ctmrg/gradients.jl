using Test
using Random
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit

## Test models, gradmodes and CTMRG algorithm
# -------------------------------------------
χbond = 2
χenv = 4
Pspaces = [ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1)]
Vspaces = [ComplexSpace(χbond), Vect[FermionParity](0 => χbond / 2, 1 => χbond / 2)]
Espaces = [ComplexSpace(χenv), Vect[FermionParity](0 => χenv / 2, 1 => χenv / 2)]
models = [square_lattice_heisenberg(), square_lattice_pwave()]
names = ["Heisenberg", "p-wave superconductor"]
Random.seed!(42039482030)
tol = 1e-8
boundary_alg = CTMRG(;
    trscheme=truncdim(χenv), tol=tol, miniter=4, maxiter=100, fixedspace=true, verbosity=0
)
gradmodes = [
    nothing, GeomSum(; tol), ManualIter(; tol), KrylovKit.GMRES(; tol=tol, maxiter=10)
]
steps = -0.01:0.005:0.01

## Tests
# ------
@testset "AD CTMRG energy gradients for $(names[i]) model" verbose = true for i in
                                                                              eachindex(
    models
)
    Pspace = Pspaces[i]
    Vspace = Pspaces[i]
    Espace = Espaces[i]
    psi_init = InfinitePEPS(Pspace, Vspace, Vspace)
    @testset "$alg_rrule" for alg_rrule in gradmodes
        dir = InfinitePEPS(Pspace, Vspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace, Vspace)
        env = leading_boundary(CTMRGEnv(psi; Venv=Espace), psi, boundary_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha=steps,
            retract=PEPSKit.my_retract,
            inner=PEPSKit.my_inner,
        ) do (peps, envs)
            E, g = Zygote.withgradient(peps) do psi
                envs2 = PEPSKit.hook_pullback(
                    leading_boundary, envs, psi, boundary_alg, ; alg_rrule
                )
                return costfun(psi, envs2, models[i])
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end
