using Test
using Random
using TensorKit
using PEPSKit
using Adapt
using PEPSKit: random_dual!, eachcoordinate, NORTH
using PEPSKit: fix_relative_phases, _contract_site

ds = Dict(
    U1Irrep => U1Space(i => d for (i, d) in zip(-1:1, (1, 1, 2))),
    FermionParity => Vect[FermionParity](0 => 2, 1 => 1)
)
Ds = Dict(
    U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (1, 3, 2))),
    FermionParity => Vect[FermionParity](0 => 3, 1 => 2)
)
χs = Dict(
    U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (2, 4, 2))),
    FermionParity => Vect[FermionParity](0 => 4, 1 => 4)
)

"""Generate a non-unitary gauge near the identity to avoid ill-conditioned inverses."""
function _random_gauge_matrix(AT, V::ElementarySpace)
    return adapt(AT, id(ComplexF64, V) + 0.1 * randn(ComplexF64, V ← V))
end

"""Test virtual and boundary gauge invariance and the unitary gauge-fixing convention."""
function ctmrg_gaugetrans(AT)
    return @testset "CTMRGEnv of InfinitePEPS ($S) ($AT)" for S in keys(ds)
        d, D, χ, uc = ds[S], Ds[S], χs[S], (2, 3)
        N, E = random_dual!(fill(D, uc)), random_dual!(fill(D, uc))
        ψ = adapt(AT, InfinitePEPS(fill(d, uc), N, E))
        env = adapt(AT, CTMRGEnv(ψ, χ))
        gauge = VirtualGaugeTransform(
            map(eachcoordinate(ψ, 1:2)) do (dir, r, c)
                return _random_gauge_matrix(AT, dir == NORTH ? N[r, c] : E[r, c])
            end
        )
        boundary_gauge = CTMRGEnvGaugeTransform(
            map(env.edges) do edge
                return _random_gauge_matrix(AT, space(edge, 1))
            end
        )
        ψg, envg = gauge_transform(ψ, gauge), gauge_transform(env, gauge)
        envb = gauge_transform(env, boundary_gauge)
        for R in (rotl90, rotr90, rot180), (state, g, transformed) in
                ((ψ, gauge, ψg), (env, gauge, envg), (env, boundary_gauge, envb))
            @test R(transformed) ≈ gauge_transform(R(state), R(g))
        end
        for (r, c) in eachcoordinate(ψ)
            inds = [CartesianIndex(r, c)]
            ρ = reduced_densitymatrix(inds, ψ, env)
            @test reduced_densitymatrix(inds, ψg, envg) ≈ ρ
            @test reduced_densitymatrix(inds, ψ, envb) ≈ ρ
        end
        signs = map(boundary_gauge.matrices) do X
            Q, = left_orth(X)
            return Q
        end
        @test gauge_transform(env, CTMRGEnvGaugeTransform(signs, adjoint.(signs))) ≈
            fix_relative_phases(env, signs)
    end
end

"""Test independent PEPO layer gauges through single-layer density matrices."""
function ctmrg_gaugetrans_pepo(AT)
    return @testset "PEPO virtual gauges ($S) ($AT)" for S in keys(ds)
        d, D, χ, uc = ds[S], Ds[S], χs[S], (2, 3, 2)
        P, N, E = fill(d, uc), fill(D, uc), fill(D, uc)
        foreach(random_dual!, eachslice(N; dims = 3))
        foreach(random_dual!, eachslice(E; dims = 3))
        ρ = adapt(AT, InfinitePEPO(P, N, E))
        X = map(Iterators.product(1:2, axes(P)...)) do (dir, r, c, h)
            return _random_gauge_matrix(AT, dir == NORTH ? N[r, c, h] : E[r, c, h])
        end
        gauge = VirtualGaugeTransform(X)
        ρg = gauge_transform(ρ, gauge)
        for R in (rotl90, rotr90, rot180)
            @test R(ρg) ≈ gauge_transform(R(ρ), R(gauge))
        end
        for h in axes(P, 3)
            layer, layerg = InfinitePEPO(ρ.A[:, :, h:h]), InfinitePEPO(ρg.A[:, :, h:h])
            env = adapt(AT, CTMRGEnv(InfinitePartitionFunction(layer), χ))
            envg = gauge_transform(env, VirtualGaugeTransform(X[:, :, :, h:h]))
            for (r, c) in Iterators.product(axes(P, 1), axes(P, 2))
                inds = [CartesianIndex(r, c)]
                @test reduced_densitymatrix(inds, layerg, envg) ≈ reduced_densitymatrix(inds, layer, env)
            end
        end
    end
end

"""Test boundary gauge invariance with more than two network-facing edge legs."""
function ctmrg_gaugetrans_stacked_env(AT)
    return @testset "Stacked-network boundary gauges ($AT)" begin
        V = ℂ^2
        ψ = adapt(AT, InfinitePEPS(V, V))
        ρ = adapt(AT, InfinitePEPO(V, V; unitcell = (1, 1, 2)))
        network = InfiniteSquareNetwork(ψ, ρ, ψ)
        env = adapt(AT, CTMRGEnv(network, V))
        gauge = CTMRGEnvGaugeTransform(
            map(env.edges) do edge
                return _random_gauge_matrix(AT, space(edge, 1))
            end
        )
        @test _contract_site((1, 1), network, gauge_transform(env, gauge)) ≈
            _contract_site((1, 1), network, env)
    end
end
