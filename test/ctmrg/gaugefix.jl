using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iter, gauge_fix, check_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 4)]
χ = 8

function _make_symmetric(psi)
    if ==(size(psi)...)
        return PEPSKit.symmetrize(psi, PEPSKit.Full())
    else
        return PEPSKit.symmetrize(PEPSKit.symmetrize(psi, PEPSKit.Depth()), PEPSKit.Width())
    end
end

@testset "Trivial symmetry ($T) - ($unitcell)" for (T, unitcell) in
                                                   Iterators.product(scalartypes, unitcells)
    Random.seed!(1234567)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(χ)

    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)
    psi = _make_symmetric(psi)
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    verbosity = 1
    alg = CTMRG(;
        trscheme=truncdim(dim(ctm_space)), tol=1e-10, miniter=4, maxiter=200, verbosity
    )
    alg_fixed = CTMRG(; trscheme=truncdim(dim(ctm_space)), verbosity, fixedspace=true)

    ctm = leading_boundary(psi, alg, ctm)
    ctm2, = ctmrg_iter(psi, ctm, alg_fixed)
    ctm_fixed = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed)
end

@testset "Z2 symmetry ($T) - ($unitcell)" for (T, unitcell) in
                                              Iterators.product(scalartypes, unitcells)
    Random.seed!(1234567)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 1, 1 => 1)
    ctm_space = Z2Space(0 => χ ÷ 2, 1 => χ ÷ 2)

    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)
    psi = _make_symmetric(psi)
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    verbosity = 1
    alg = CTMRG(;
        trscheme=truncspace(ctm_space), tol=1e-10, miniter=4, maxiter=200, verbosity
    )
    alg_fixed = CTMRG(; trscheme=truncspace(ctm_space), verbosity, fixedspace=true)

    ctm = leading_boundary(psi, alg, ctm)
    ctm2, = ctmrg_iter(psi, ctm, alg_fixed)
    ctm_fixed = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed)
end
