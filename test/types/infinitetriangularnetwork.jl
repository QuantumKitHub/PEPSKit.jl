using Test
using TensorKit
using PEPSKit
using Zygote
using LinearAlgebra
using PEPSKit: unitcell, virtualspace

@testset "Triangular constructors" begin
    for T in (Float64, ComplexF64), (V, W) in (
                (ComplexSpace(1), ComplexSpace(2)),
                (U1Space(0 => 1), U1Space(0 => 1, 1 => 1)),
            )
        NW = [V W V; W V W]
        NE = [W V V; V W W]
        E = [V V W; W V V]
        P = fill(W, 2, 3)
        state = InfinitePEPSTriangular(ones, T, P, NW, NE, E)
        A = unitcell(state)
        for (dir, δr, δc) in ((1, -1, 0), (2, -1, 1), (3, 0, 1)), r in 1:2, c in 1:3
            @test virtualspace(state, r, c, dir) == virtualspace(state, r + δr, c + δc, dir + 3)'
        end
        @test InfinitePEPSTriangular(copy(A)) == state
        @test unitcell(InfiniteTriangularNetwork(state)) == map(tuple, A, A)
        @test_throws ArgumentError InfinitePEPSTriangular(ones, T, P, NW[:, 1:2], NE, E)

        # Same six virtual legs in the classical tensor layout W SW SE ← NW NE E.
        pf(t) = ones(T, domain(t)[6]' ⊗ domain(t)[5]' ⊗ domain(t)[4]', domain(t)[1] ⊗ domain(t)[2] ⊗ domain(t)[3])
        tensors = map(pf, A)
        @test unitcell(InfiniteTriangularNetwork(tensors)) == tensors

        # Break each leg at a periodic boundary without changing its dimension.
        for dir in 1:6
            invalid = copy(A)
            spaces = collect(domain(A[1, 3]))
            spaces[dir] = spaces[dir]'
            invalid[1, 3] = ones(T, codomain(A[1, 3]), ProductSpace(spaces...))
            @test_throws SpaceMismatch InfinitePEPSTriangular(invalid)
            @test_throws SpaceMismatch InfiniteTriangularNetwork(map(tuple, invalid, A))
            @test_throws SpaceMismatch InfiniteTriangularNetwork(map(tuple, A, invalid))
            @test_throws SpaceMismatch InfiniteTriangularNetwork(map(pf, invalid))
        end

        uniform = InfinitePEPSTriangular(ones, T, W, V; unitcell = (1, 1))
        @test size(InfinitePEPSTriangular(uniform[1, 1]; unitcell = (2, 3))) == (2, 3)
    end

    P = Vect[FermionParity](0 => 1, 1 => 1)
    V = Vect[FermionParity](0 => 1)
    @test size(InfinitePEPSTriangular(ones, Float64, P, V)) == (1, 1)
    @test_throws ArgumentError InfinitePEPSTriangular(ones(Float64, P', V^3 ⊗ (V')^3))
end

@testset "Triangular constructor differentiation" begin
    V = ComplexSpace(1)
    A = unitcell(InfinitePEPSTriangular(ones, Float64, V, V; unitcell = (1, 2)))
    for construct in (InfinitePEPSTriangular, a -> InfiniteTriangularNetwork(map(tuple, a, a)))
        f(a) = sum(unitcell(construct(a))) do t
            x = t isa Tuple ? first(t) : t
            return real(dot(x, x))
        end
        g = only(Zygote.gradient(f, A))
        @test all(isapprox.(g, 2 .* A))
    end
end

@testset "Triangular environment types" begin
    V = ComplexSpace(1)
    for sandwich in (false, true)
        env = CTMRGEnvTriangular(ones, Float64, sandwich ? V ⊗ V' : V, V)
        complex_edges = complex.(env.Ea)
        for (Ea, Eb, E) in (
                (env.Ea, env.Eb, eltype(env.Ea)),
                (complex_edges, env.Eb, eltype(complex_edges)),
                (env.Ea, complex_edges, eltype(complex_edges)),
            )
            env′ = CTMRGEnvTriangular(env.C, Ea, Eb)
            @test PEPSKit.cornertype(env′) === eltype(env.C)
            @test PEPSKit.cornertype(typeof(env′)) === eltype(env.C)
            @test PEPSKit.edgetype(env′) === E
            @test PEPSKit.edgetype(typeof(env′)) === E
            @test spacetype(env′) === typeof(V)
            @test spacetype(typeof(env′)) === typeof(V)
        end
    end
end
