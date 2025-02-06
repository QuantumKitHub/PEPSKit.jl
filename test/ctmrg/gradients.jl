using Test
using Random
using TensorKit
using KrylovKit
using PEPSKit
using PEPSKit: to_vec, PEPSCostFunctionCache, gradient_function
using Manopt, Manifolds

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
        SimultaneousCTMRG(; verbosity=0, projector_alg=HalfInfiniteProjector),
        SimultaneousCTMRG(; verbosity=0, projector_alg=FullInfiniteProjector),
    ],
    [SequentialCTMRG(; verbosity=0, projector_alg=HalfInfiniteProjector)],
]
gradmodes = [
    [
        nothing,
        GeomSum(; tol=gradtol, iterscheme=:fixed),
        GeomSum(; tol=gradtol, iterscheme=:diffgauge),
        ManualIter(; tol=gradtol, iterscheme=:fixed),
        ManualIter(; tol=gradtol, iterscheme=:diffgauge),
        LinSolver(; solver=KrylovKit.BiCGStab(; tol=gradtol), iterscheme=:fixed),
        LinSolver(; solver=KrylovKit.BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
    ],
    [  # Only use :diffgauge due to high gauge-sensitivity (perhaps due to small χenv?)
        nothing,
        GeomSum(; tol=gradtol, iterscheme=:diffgauge),
        ManualIter(; tol=gradtol, iterscheme=:diffgauge),
        LinSolver(; solver=KrylovKit.BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
    ],
]

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
    @testset "$ctmrg_alg and $gradient_alg" for (ctmrg_alg, gradient_alg) in
                                                Iterators.product(calgs, gms)
        @info "gradient check of $ctmrg_alg and $gradient_alg on $(names[i])"
        Random.seed!(42039482030)
        psi = InfinitePEPS(Pspace, Vspace)
        env, = leading_boundary(CTMRGEnv(psi, Espace), psi, ctmrg_alg)

        psi_vec, from_vec = to_vec(psi)
        opt_alg = PEPSOptimize(; boundary_alg=ctmrg_alg, gradient_alg)
        cache = PEPSCostFunctionCache(models[i], opt_alg, psi_vec, from_vec, deepcopy(env))
        cost = cache
        grad = gradient_function(cache)

        M = Euclidean(length(psi_vec))
        @test check_gradient(M, cost, grad; N=10, exactness_tol=gradtol, limits=(-8, -3))
    end
end
