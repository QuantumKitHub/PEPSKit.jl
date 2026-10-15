using Test, Random, LinearAlgebra, TensorKit, PEPSKit

@testset "Norm-preserving optimization geometry" begin
    Random.seed!(1234)
    peps = PEPSKit.peps_normalize(InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell = (1, 2)))
    direction = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell = (1, 2))
    tangent = PEPSKit.peps_project_tangent(direction, peps)
    (retracted, _), velocity = PEPSKit.peps_retract((peps, nothing), tangent, 0.3)
    transported = deepcopy(tangent)
    PEPSKit.peps_transport!(transported, (peps, nothing), tangent, 0.3, (retracted, nothing))
    for i in eachindex(peps)
        @test abs(real(dot(peps[i], tangent[i]))) < 1e-12
        @test norm(retracted[i]) ≈ 1 atol = 1e-12
        @test norm(transported[i]) ≈ norm(tangent[i]) rtol = 1e-12
        @test abs(real(dot(retracted[i], transported[i]))) < 1e-12
        @test transported[i] ≈ velocity[i]
    end
    @test PEPSKit.peps_project_tangent(im * peps, peps) ≈ im * peps
end
