using Test
using Random
using PEPSKit
using TensorKit
using Accessors

using PEPSKit: ctmrg_iter, gauge_fix, check_elementwise_convergence

scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
χ = 6

function _make_symmetric(psi)
    if ==(size(psi)...)
        return PEPSKit.symmetrize(psi, PEPSKit.Full())
    else
        return PEPSKit.symmetrize(PEPSKit.symmetrize(psi, PEPSKit.Depth()), PEPSKit.Width())
    end
end

# If I can't make the rng seed behave, I'll just randomly define a peps somehow
function semi_random_peps!(psi::InfinitePEPS)
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

@testset "Trivial symmetry ($T) - ($unitcell)" for (T, unitcell) in
                                                   Iterators.product(scalartypes, unitcells)
    physical_space = ComplexSpace(2)
    peps_space = ComplexSpace(2)
    ctm_space = ComplexSpace(χ)

    psi = InfinitePEPS(undef, T, physical_space, peps_space; unitcell)
    semi_random_peps!(psi)
    psi = _make_symmetric(psi)

    Random.seed!(987654321)  # Seed RNG to make random environment consistent
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    verbosity = 1
    alg = CTMRG(;
        tol=1e-10, miniter=4, maxiter=400, verbosity, trscheme=truncdim(dim(ctm_space))
    )
    alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation()

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg_fixed)
    ctm_fixed = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed; atol=1e-4)
end

@testset "Z2 symmetry ($T) - ($unitcell)" for (T, unitcell) in
                                              Iterators.product(scalartypes, unitcells)
    physical_space = Z2Space(0 => 1, 1 => 1)
    peps_space = Z2Space(0 => 1, 1 => 1)
    ctm_space = Z2Space(0 => χ ÷ 2, 1 => χ ÷ 2)

    psi = InfinitePEPS(undef, T, physical_space, peps_space; unitcell)
    semi_random_peps!(psi)
    psi = _make_symmetric(psi)

    Random.seed!(123456789)  # Seed RNG to make random environment consistent
    ctm = CTMRGEnv(psi; Venv=ctm_space)

    verbosity = 1
    alg = CTMRG(;
        tol=1e-10, miniter=4, maxiter=400, verbosity, trscheme=truncdim(dim(ctm_space))
    )
    alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation()

    ctm = leading_boundary(ctm, psi, alg)
    ctm2, = ctmrg_iter(psi, ctm, alg_fixed)
    ctm_fixed = gauge_fix(ctm, ctm2)
    @test PEPSKit.check_elementwise_convergence(ctm, ctm_fixed; atol=1e-4)
end
