using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iter, gauge_fix, calc_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
schemes = [:simultaneous, :sequential]
χ = Dict([(1, 1) => 8, (2, 2) => 26, (3, 2) => 26])  # Increase χ to converge non-symmetric environments

@testset "Trivial symmetry ($T) - ($unitcell) - ($ctmrgscheme)" for (
    T, unitcell, ctmrgscheme
) in Iterators.product(
    scalartypes, unitcells, schemes
)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(χ[unitcell])

    Random.seed!(29358293852)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(randn, T, physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi, ctm_space)

    alg = CTMRG(;
        tol=1e-10, maxiter=100, verbosity=2, trscheme=FixedSpaceTruncation(), ctmrgscheme
    )

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg)
    ctm_fixed, = gauge_fix(ctm, ctm2)
    @test calc_elementwise_convergence(ctm, ctm_fixed) ≈ 0 atol=1e-6
end

@testset "Z2 symmetry ($T) - ($unitcell) - ($ctmrgscheme)" for (T, unitcell, ctmrgscheme) in
                                                               Iterators.product(
    scalartypes, unitcells, schemes
)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 1, 1 => 1)
    ctm_space = Z2Space(0 => χ[(1, 1)] ÷ 2, 1 => χ[(1, 1)] ÷ 2)

    Random.seed!(2938293852938)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi, ctm_space)

    alg = CTMRG(;
        tol=1e-10, maxiter=400, verbosity=2, trscheme=FixedSpaceTruncation(), ctmrgscheme
    )

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg)
    ctm_fixed, = gauge_fix(ctm, ctm2)
    @test calc_elementwise_convergence(ctm, ctm_fixed) ≈ 0 atol=1e-6
end
