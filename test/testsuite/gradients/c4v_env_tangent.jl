using Test
using Adapt
using Random
using PEPSKit
using PEPSKit: C4vEnvTangent
using TensorKit
using VectorInterface

Venv = ComplexSpace(4)
Vpeps = ComplexSpace(2)

function random_corner(AT, kind)
    return if kind === :diagonal
        adapt(AT, DiagonalTensorMap(randn(Float64, dim(Venv)), Venv))
    else
        adapt(AT, randn(Float64, Venv ← Venv))
    end
end
random_edge(AT, T) = adapt(AT, randn(T, Venv ⊗ Vpeps ⊗ Vpeps' ← Venv))
random_tangent(AT, kind, T) = C4vEnvTangent(random_corner(AT, kind), random_edge(AT, T))

parts(x::C4vEnvTangent) = (x.corner, x.edge)
issame(x::C4vEnvTangent, (C, E)) = x.corner ≈ C && x.edge ≈ E

# this file mostly tests the helper methods, because the
# main differentiation handling is already tested by the existing
# C4v gradient tests

function gradients_c4v_env_tangent(AT)
    return @testset "C4vEnvTangent vector interface ($kind corner, $T edge) ($AT)" for (kind, T) in
        ((:diagonal, ComplexF64), (:dense, Float64))
        Random.seed!(1234)
        x = random_tangent(AT, kind, T)
        y = random_tangent(AT, kind, T)
        α, β = 0.7, -1.3
        Cx, Ex = deepcopy(parts(x))
        Cy, Ey = deepcopy(parts(y))

        @testset "scalartype" begin
            @test scalartype(x) == promote_type(Float64, T)
        end

        @testset "zerovector" begin
            z = zerovector(x)
            @test z isa C4vEnvTangent
            @test scalartype(z.corner) == scalartype(z.edge) == scalartype(x)
            @test iszero(norm(z))
            zc = zerovector(x, ComplexF64)
            @test scalartype(zc) == ComplexF64
            @test iszero(norm(zc))
            z′ = deepcopy(x)
            @test zerovector!(z′) === z′
            @test iszero(norm(z′))
            z″ = deepcopy(x)
            @test zerovector!!(z″) === z″
            @test iszero(norm(z″))
        end

        # test the VI methods
        @testset "scale" begin
            @test issame(scale(x, α), (scale(Cx, α), scale(Ex, α)))
            @test issame(x, (Cx, Ex)) # not mutated
            x′ = deepcopy(x)
            @test scale!(x′, α) === x′
            @test issame(x′, (scale(Cx, α), scale(Ex, α)))
            x″ = deepcopy(x)
            @test scale!!(x″, α) === x″
            @test issame(x″, (scale(Cx, α), scale(Ex, α)))
            z = zerovector(x)
            @test scale!(z, x, α) === z
            @test issame(z, (scale(Cx, α), scale(Ex, α)))
            z′ = zerovector(x)
            @test scale!!(z′, x, α) === z′
            @test issame(z′, (scale(Cx, α), scale(Ex, α)))
            @test issame(x, (Cx, Ex)) # source not mutated
        end

        @testset "add" begin
            expected = (add(Cy, Cx, α, β), add(Ey, Ex, α, β))
            @test issame(add(y, x, α, β), expected)
            @test issame(y, (Cy, Ey)) && issame(x, (Cx, Ex)) # not mutated
            y′ = deepcopy(y)
            @test add!(y′, x, α, β) === y′
            @test issame(y′, expected)
            y″ = deepcopy(y)
            @test add!!(y″, x, α, β) === y″
            @test issame(y″, expected)
            @test issame(x, (Cx, Ex)) # source not mutated
        end

        @testset "inner and norm" begin
            @test inner(x, y) ≈ inner(Cx, Cy) + inner(Ex, Ey)
            @test inner(x, y) ≈ conj(inner(y, x))
            @test norm(x) ≈ sqrt(norm(Cx)^2 + norm(Ex)^2)
            @test norm(x)^2 ≈ real(inner(x, x))
        end
    end
end
