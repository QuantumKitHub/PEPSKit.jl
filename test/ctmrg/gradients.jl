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
models = [heisenberg_XYZ(InfiniteSquare()), pwave_superconductor(InfiniteSquare())]
names = ["Heisenberg", "p-wave superconductor"]

gradtol = 1e-4
ctmrg_algs = [
    [
        SimultaneousCTMRG(; verbosity=0, projector_alg=:halfinfinite),
        SimultaneousCTMRG(; verbosity=0, projector_alg=:fullinfinite),
    ],
    [SequentialCTMRG(; verbosity=0, projector_alg=:halfinfinite)],
]
gradmodes = [
    [
        nothing,
        GeomSum(; tol=gradtol, iterscheme=:fixed),
        GeomSum(; tol=gradtol, iterscheme=:diffgauge),
        ManualIter(; tol=gradtol, iterscheme=:fixed),
        ManualIter(; tol=gradtol, iterscheme=:diffgauge),
        LinSolver(; solver_alg=BiCGStab(; tol=gradtol), iterscheme=:fixed),
        LinSolver(; solver_alg=BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
        EigSolver(; solver_alg=Arnoldi(; tol=gradtol, eager=true), iterscheme=:fixed),
        EigSolver(; solver_alg=Arnoldi(; tol=gradtol, eager=true), iterscheme=:diffgauge),
    ],
    [  # Only use :diffgauge due to high gauge-sensitivity (perhaps due to small χenv?)
        nothing,
        GeomSum(; tol=gradtol, iterscheme=:diffgauge),
        ManualIter(; tol=gradtol, iterscheme=:diffgauge),
        LinSolver(; solver_alg=BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
        EigSolver(; solver_alg=Arnoldi(; tol=gradtol, eager=true), iterscheme=:diffgauge),
    ],
]
steps = -0.01:0.005:0.01

## Tests
# ------
@testset "AD CTMRG energy gradients for $(names[i]) model" verbose = true for i in
                                                                              eachindex(
    models
)
    Pspace = Pspaces[i]
    Vspace = Vspaces[i]
    Espace = Espaces[i]
    gms = gradmodes[i]
    calgs = ctmrg_algs[i]
    @testset "$ctmrg_alg and $alg_rrule" for (ctmrg_alg, alg_rrule) in
                                             Iterators.product(calgs, gms)
        @info "optimtest of $ctmrg_alg and $alg_rrule on $(names[i])"
        Random.seed!(42039482030)
        dir = InfinitePEPS(Pspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace)
        env, = leading_boundary(CTMRGEnv(psi, Espace), psi, ctmrg_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha=steps,
            retract=PEPSKit.peps_retract,
            inner=PEPSKit.real_inner,
        ) do (peps, env)
            E, g = Zygote.withgradient(peps) do psi
                env2, = PEPSKit.hook_pullback(leading_boundary, env, psi, ctmrg_alg; alg_rrule)
                return cost_function(psi, env2, models[i])
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end
