"""
    gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}

Fix the gauge of `envfinal` based on the previous environment `envprev`.
This assumes that the `envfinal` is the result of one CTMRG iteration on `envprev`.
Given that the CTMRG run is converged, the returned environment will be
element-wise converged to `envprev`.
"""
function gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C′,T′}) where {C,C′,T,T′}
    # Check if spaces in envprev and envfinal are the same
    same_spaces = map(Iterators.product(axes(envfinal.edges)...)) do (dir, r, c)
        space(envfinal.edges[dir, r, c]) == space(envprev.edges[dir, r, c]) &&
            space(envfinal.corners[dir, r, c]) == space(envprev.corners[dir, r, c])
    end
    @assert all(same_spaces) "Spaces of envprev and envfinal are not the same"

    # "general" algorithm from https://arxiv.org/abs/2311.11894
    signs = map(Iterators.product(axes(envfinal.edges)...)) do (dir, r, c)
        # Gather edge tensors and pretend they're InfiniteMPSs
        if dir == NORTH
            Tsprev = circshift(envprev.edges[dir, r, :], 1 - c)
            Tsfinal = circshift(envfinal.edges[dir, r, :], 1 - c)
        elseif dir == EAST
            Tsprev = circshift(envprev.edges[dir, :, c], 1 - r)
            Tsfinal = circshift(envfinal.edges[dir, :, c], 1 - r)
        elseif dir == SOUTH
            Tsprev = circshift(reverse(envprev.edges[dir, r, :]), c)
            Tsfinal = circshift(reverse(envfinal.edges[dir, r, :]), c)
        elseif dir == WEST
            Tsprev = circshift(reverse(envprev.edges[dir, :, c]), r)
            Tsfinal = circshift(reverse(envfinal.edges[dir, :, c]), r)
        end

        # Random MPS of same bond dimension
        M = map(Tsfinal) do t
            TensorMap(randn, scalartype(t), codomain(t) ← domain(t))
        end

        # Find right fixed points of mixed transfer matrices
        ρinit = TensorMap(
            randn,
            scalartype(T),
            MPSKit._lastspace(Tsfinal[end])' ← MPSKit._lastspace(M[end])',
        )
        ρprev = transfermatrix_fixedpoint(Tsprev, M, ρinit)
        ρfinal = transfermatrix_fixedpoint(Tsfinal, M, ρinit)

        # Decompose and multiply
        Qprev, = leftorth(ρprev)
        Qfinal, = leftorth(ρfinal)

        return Qprev * Qfinal'
    end

    cornersfix, edgesfix = fix_relative_phases(envfinal, signs)
    return fix_global_phases(envprev, CTMRGEnv(cornersfix, edgesfix)), signs
end

# this is a bit of a hack to get the fixed point of the mixed transfer matrix
# because MPSKit is not compatible with AD
function transfermatrix_fixedpoint(tops, bottoms, ρinit)
    _, vecs, info = eigsolve(ρinit, 1, :LM, Arnoldi()) do ρ
        return foldr(zip(tops, bottoms); init=ρ) do (top, bottom), ρ
            return @tensor ρ′[-1; -2] := top[-1 4 3; 1] * conj(bottom[-2 4 3; 2]) * ρ[1; 2]
        end
    end
    info.converged > 0 || @warn "eigsolve did not converge"
    return first(vecs)
end

# Explicit fixing of relative phases (doing this compactly in a loop is annoying)
function _contract_gauge_corner(corner, σ_in, σ_out)
    @autoopt @tensor corner_fix[χ_in; χ_out] :=
        σ_in[χ_in; χ1] * corner[χ1; χ2] * conj(σ_out[χ_out; χ2])
end
function _contract_gauge_edge(edge, σ_in, σ_out)
    @autoopt @tensor edge_fix[χ_in D_above D_below; χ_out] :=
        σ_in[χ_in; χ1] * edge[χ1 D_above D_below; χ2] * conj(σ_out[χ_out; χ2])
end
function fix_relative_phases(envfinal::CTMRGEnv, signs)
    C1 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[NORTHWEST, r, c],
            signs[WEST, r, c],
            signs[NORTH, r, _next(c, end)],
        )
    end
    T1 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[NORTH, r, c],
            signs[NORTH, r, c],
            signs[NORTH, r, _next(c, end)],
        )
    end
    C2 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[NORTHEAST, r, c],
            signs[NORTH, r, c],
            signs[EAST, _next(r, end), c],
        )
    end
    T2 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[EAST, r, c], signs[EAST, r, c], signs[EAST, _next(r, end), c]
        )
    end
    C3 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[SOUTHEAST, r, c],
            signs[EAST, r, c],
            signs[SOUTH, r, _prev(c, end)],
        )
    end
    T3 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[SOUTH, r, c],
            signs[SOUTH, r, c],
            signs[SOUTH, r, _prev(c, end)],
        )
    end
    C4 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[SOUTHWEST, r, c],
            signs[SOUTH, r, c],
            signs[WEST, _prev(r, end), c],
        )
    end
    T4 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[WEST, r, c], signs[WEST, r, c], signs[WEST, _prev(r, end), c]
        )
    end

    return stack([C1, C2, C3, C4]; dims=1), stack([T1, T2, T3, T4]; dims=1)
end
function fix_relative_phases(
    U::Array{Ut,3}, V::Array{Vt,3}, signs
) where {Ut<:AbstractTensorMap,Vt<:AbstractTensorMap}
    U1 = map(Iterators.product(axes(U)[2:3]...)) do (r, c)
        return U[NORTH, r, c] * signs[NORTH, r, _next(c, end)]
    end
    V1 = map(Iterators.product(axes(V)[2:3]...)) do (r, c)
        return signs[NORTH, r, _next(c, end)]' * V[NORTH, r, c]
    end
    U2 = map(Iterators.product(axes(U)[2:3]...)) do (r, c)
        return U[EAST, r, c] * signs[EAST, _next(r, end), c]
    end
    V2 = map(Iterators.product(axes(V)[2:3]...)) do (r, c)
        return signs[EAST, _next(r, end), c]' * V[EAST, r, c]
    end
    U3 = map(Iterators.product(axes(U)[2:3]...)) do (r, c)
        return U[SOUTH, r, c] * signs[SOUTH, r, _prev(c, end)]
    end
    V3 = map(Iterators.product(axes(V)[2:3]...)) do (r, c)
        return signs[SOUTH, r, _prev(c, end)]' * V[SOUTH, r, c]
    end
    U4 = map(Iterators.product(axes(U)[2:3]...)) do (r, c)
        return U[WEST, r, c] * signs[WEST, _prev(r, end), c]
    end
    V4 = map(Iterators.product(axes(V)[2:3]...)) do (r, c)
        return signs[WEST, _prev(r, end), c]' * V[WEST, r, c]
    end

    return stack([U1, U2, U3, U4]; dims=1), stack([V1, V2, V3, V4]; dims=1)
end

# Fix global phases of corners and edges via dot product (to ensure compatibility with symm. tensors)
function fix_global_phases(envprev::CTMRGEnv, envfix::CTMRGEnv)
    cornersgfix = map(zip(envprev.corners, envfix.corners)) do (Cprev, Cfix)
        φ = dot(Cprev, Cfix)
        φ' * Cfix
    end
    edgesgfix = map(zip(envprev.edges, envfix.edges)) do (Tprev, Tfix)
        φ = dot(Tprev, Tfix)
        φ' * Tfix
    end
    return CTMRGEnv(cornersgfix, edgesgfix)
end

"""
    check_elementwise_convergence(envfinal, envfix; atol=1e-6)

Check if the element-wise difference of the corner and edge tensors of the final and fixed
CTMRG environments are below some tolerance.
"""
function check_elementwise_convergence(
    envfinal::CTMRGEnv, envfix::CTMRGEnv; atol::Real=1e-6
)
    ΔC = envfinal.corners .- envfix.corners
    ΔCmax = norm(ΔC, Inf)
    ΔCmean = norm(ΔC)
    @debug "maxᵢⱼ|Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmax   mean |Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmean"

    ΔT = envfinal.edges .- envfix.edges
    ΔTmax = norm(ΔT, Inf)
    ΔTmean = norm(ΔT)
    @debug "maxᵢⱼ|Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmax   mean |Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmean"

    # Check differences for all tensors in unit cell to debug properly
    for (dir, r, c) in Iterators.product(axes(envfinal.edges)...)
        @debug(
            "$((dir, r, c)): all |Cⁿ⁺¹ - Cⁿ|ᵢⱼ < ϵ: ",
            all(x -> abs(x) < atol, convert(Array, ΔC[dir, r, c])),
        )
        @debug(
            "$((dir, r, c)): all |Tⁿ⁺¹ - Tⁿ|ᵢⱼ < ϵ: ",
            all(x -> abs(x) < atol, convert(Array, ΔT[dir, r, c])),
        )
    end

    return isapprox(ΔCmax, 0; atol) && isapprox(ΔTmax, 0; atol)
end

@non_differentiable check_elementwise_convergence(args...)