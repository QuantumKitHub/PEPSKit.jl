using PEPSKit,MPSKit,TensorKit,Test

println("------------------------------------")
println("|     States                       |")
println("------------------------------------")
@testset "Infinte peps ($D,$d,$elt)" for (D,d,elt) in [
        (ComplexSpace(10),ComplexSpace(2),ComplexF64),
        (ℂ[SU₂](1=>1,0=>3),ℂ[SU₂](0=>1)*ℂ[SU₂](0=>1),ComplexF32)
        ]

    
end
