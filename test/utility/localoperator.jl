using TensorKit
using PEPSKit
using Test

vds = (ℂ^2, Rep[U₁](1 => 1, -1 => 1), Rep[SU₂](1 / 2 => 1))
@testset "LocalOperator $vd" for vd in vds
    t = randn(ComplexF64, vd ⊗ vd ← vd ⊗ vd)
    physical_spaces = fill(vd, (2, 2))

    terms = ((CartesianIndex(1, 1), CartesianIndex(1, 2)) => t,)
    op = LocalOperator(physical_spaces, terms...)

    @test op isa LocalOperator
    @test length(op.terms) == 1
    @test sectortype(op) === sectortype(vd)
    @test spacetype(op) === typeof(vd)
    @test physicalspace(op) == physical_spaces

    @test real(last(only(real(op).terms))) == real(t)
    @test real(last(only(imag(op).terms))) == imag(t)

    op2 = 2 * op
    @test op2 isa LocalOperator
    @test typeof(op2) === typeof(op)
    @test physicalspace(op2) == physical_spaces

    @test real(last(only(real(op2).terms))) ≈ 2 * real(t)
    @test real(last(only(imag(op2).terms))) ≈ 2 * imag(t)

    op3 = op / 3
    @test op3 isa LocalOperator
    @test typeof(op3) === typeof(op)
    @test physicalspace(op3) == physical_spaces

    @test real(last(only(real(op3).terms))) ≈ real(t) / 3
    @test real(last(only(imag(op3).terms))) ≈ imag(t) / 3

    t2 = randn(vd ⊗ vd ← vd ⊗ vd)
    terms2 = ((CartesianIndex(2, 1), CartesianIndex(1, 2)) => t2,)
    op4 = LocalOperator(physical_spaces, terms2...)
    op5 = op + op4
    @test op5 isa LocalOperator
    @test physicalspace(op5) == physical_spaces
    @test length(op5.terms) == 2
end
