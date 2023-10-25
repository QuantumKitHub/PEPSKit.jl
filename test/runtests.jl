using Test, PEPSKit, MPSKit, TensorKit

@testset "Boundary MPS" begin
    peps = InfinitePEPS(2, 3)
    tpeps = InfiniteTransferPEPS(peps, 1, 1)

    mps = initializeMPS(tpeps, 4)

    mps, _, _ = leading_boundary(mps, tpeps, VUMPS())
end

@testset "CTMRG" begin
    peps = InfinitePEPS(2, 3)
    tpeps = InfiniteTransferPEPS(peps, 1, 1)

    mps = initializeMPS(tpeps, 4)

    env = leading_boundary(peps, CTMRG(; tol=1e-10))
end
