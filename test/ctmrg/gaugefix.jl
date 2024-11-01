using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iter, gauge_fix, calc_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
maxiter = 200
schemes = [:simultaneous, :sequential]
χ = 6
atol = 1e-4

function _make_symmetric!(psi)
    if ==(size(psi)...)
        return symmetrize!(psi, RotateReflect())
    else
        return symmetrize!(symmetrize!(psi, ReflectDepth()), ReflectWidth())
    end
end

# If I can't make the rng seed behave, I'll just randomly define a peps somehow
function _semi_random_peps!(psi::InfinitePEPS)
    i = 0
    A′ = map(psi.A) do a
        for (_, b) in blocks(a)
            l = length(b)
            b .= reshape(collect((1:l) .+ i), size(b))
            i += l
        end
        return a
    end
    return InfinitePEPS(A′)
end

@testset "Trivial symmetry ($T) - ($unitcell) - ($ctmrgscheme)" for (
    T, unitcell, ctmrgscheme
) in Iterators.product(
    scalartypes, unitcells, schemes
)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(χ)

    psi = InfinitePEPS(undef, T, physical_space, peps_space; unitcell)
    _semi_random_peps!(psi)
    _make_symmetric!(psi)

    Random.seed!(987654321)  # Seed RNG to make random environment consistent
    ctm = CTMRGEnv(psi, ctm_space)

    alg = CTMRG(; maxiter, ctmrgscheme)

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg)
    ctm_fixed, = gauge_fix(ctm, ctm2)
    @test calc_elementwise_convergence(ctm, ctm_fixed) ≈ 0 atol = atol
end

@testset "Z2 symmetry ($T) - ($unitcell) - ($ctmrgscheme)" for (T, unitcell, ctmrgscheme) in
                                                               Iterators.product(
    scalartypes, unitcells, schemes
)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 1, 1 => 1)
    ctm_space = Z2Space(0 => χ ÷ 2, 1 => χ ÷ 2)

    psi = InfinitePEPS(undef, T, physical_space, peps_space; unitcell)
    _semi_random_peps!(psi)
    _make_symmetric!(psi)

    Random.seed!(987654321)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(physical_space, peps_space; unitcell)
    ctm = CTMRGEnv(psi, ctm_space)

    alg = CTMRG(; maxiter, ctmrgscheme)

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg)
    ctm_fixed, = gauge_fix(ctm, ctm2)
    @test calc_elementwise_convergence(ctm, ctm_fixed) ≈ 0 atol = atol
end
