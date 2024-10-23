using Test
using Random
using PEPSKit: leading_boundary
using TensorKit

begin "test utility"
    ds = [ℂ^2]
    Ds = [ℂ^3]
    χs = [ℂ^4]
end

@testset "initialize environments for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    alg = VUMPS(verbosity=4)

    itp = InfiniteTransferPEPS(ipeps)
    vumpsruntime = VUMPSRuntime(itp, χ, alg)
    @test vumpsruntime isa VUMPSRuntime
end

@testset "vumps runtime for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    alg = VUMPS(maxiter=100, verbosity=2)

    itp = InfiniteTransferPEPS(ipeps)
    leading_boundary(itp, VUMPSRuntime(itp, χ, alg), alg)
end

# @testset "(2, 2) PEPS" begin
#     psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
#     T = PEPSKit.TransferPEPSMultiline(psi, 1)

#     mps = PEPSKit.initializeMPS(T, fill(ComplexSpace(20), 2, 2))
#     mps, envs, ϵ = leading_boundary(mps, T, vumps_alg)
#     N = abs(prod(expectation_value(mps, T)))

#     ctm = leading_boundary(CTMRGEnv(psi, ComplexSpace(20)), psi, CTMRG(; verbosity=1))
#     N´ = abs(norm(psi, ctm))

#     @test N ≈ N´ rtol = 1e-2
# end