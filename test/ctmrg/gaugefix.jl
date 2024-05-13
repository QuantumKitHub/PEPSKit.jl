using Test
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iter, gauge_fix, check_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 4)]

@testset "Trivial symmetry ($T) - ($unitcell)" for (T, unitcell) in Iterators.product(scalartypes, unitcells)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(16)

    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    alg = CTMRG(; trscheme=truncdim(dim(ctm_space)), tol=1e-10, miniter=4, maxiter=100, verbosity=2)
    alg_fixed = CTMRG(; trscheme=truncdim(dim(ctm_space)), tol=1e-10, miniter=4, maxiter=100, verbosity=2, fixedspace=true)

    ctm = leading_boundary(psi, alg, ctm)
    ctm2, = ctmrg_iter(psi, ctm, alg_fixed)
    ctm_fixed = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed)
end

@testset "Z2 symmetry ($T) - ($unitcell)" for (T, unitcell) in Iterators.product(scalartypes, unitcells)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 2, 1 => 2)
    ctm_space = Z2Space(0 => 8, 1 => 8)

    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    alg = CTMRG(; trscheme=truncdim(dim(ctm_space)), tol=1e-10, miniter=4, maxiter=100, verbosity=2)
    alg_fixed = CTMRG(; trscheme=truncdim(dim(ctm_space)), tol=1e-10, miniter=4, maxiter=100, verbosity=2, fixedspace=true)

    ctm = leading_boundary(psi, alg, ctm)
    ctm2, = ctmrg_iter(psi, ctm, alg_fixed)
    ctm_fixed = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed)
end

