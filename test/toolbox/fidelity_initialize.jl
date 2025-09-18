using TensorKit
using PEPSKit
using Test

@testset "Fidelity initialization with approximate!" begin
    pepssrc = InfinitePEPS(2, 2)
    peps_single = single_site_fidelity_initialize(pepssrc, ℂ^6, ℂ^3)
    @test peps_single isa InfinitePEPS
end
