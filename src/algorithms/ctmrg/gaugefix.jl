"""
    gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}

Fix the gauge of `envfinal` based on the previous environment `envprev`.
This assumes that the `envfinal` is the result of one CTMRG iteration on `envprev`.
Given that the CTMRG run is converged, the returned environment will be
element-wise converged to `envprev`.
"""
function gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}
    # Check if spaces in envprev and envfinal are the same
    same_spaces = map(Iterators.product(axes(envfinal.edges)...)) do (dir, r, c)
        space(envfinal.edges[dir, r, c]) == space(envprev.edges[dir, r, c]) &&
            space(envfinal.corners[dir, r, c]) == space(envprev.corners[dir, r, c])
    end
    @assert all(same_spaces) "Spaces of envprev and envfinal are not the same"

    # Try the "general" algorithm from https://arxiv.org/abs/2311.11894
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
        Qprev, = leftorth!(ρprev)
        Qfinal, = leftorth!(ρfinal)

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
function fix_relative_phases(envfinal::CTMRGEnv, signs)
    corners_fixed = map(Iterators.product(axes(envfinal.corners)...)) do (dir, r, c)
        if dir == NORTHWEST
            fix_gauge_northwest_corner((r, c), envfinal, signs)
        elseif dir == NORTHEAST
            fix_gauge_northeast_corner((r, c), envfinal, signs)
        elseif dir == SOUTHEAST
            fix_gauge_southeast_corner((r, c), envfinal, signs)
        elseif dir == SOUTHWEST
            fix_gauge_southwest_corner((r, c), envfinal, signs)
        end
    end

    edges_fixed = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (dir, r, c)
        if dir == NORTHWEST
            fix_gauge_north_edge((r, c), envfinal, signs)
        elseif dir == NORTHEAST
            fix_gauge_east_edge((r, c), envfinal, signs)
        elseif dir == SOUTHEAST
            fix_gauge_south_edge((r, c), envfinal, signs)
        elseif dir == SOUTHWEST
            fix_gauge_west_edge((r, c), envfinal, signs)
        end
    end

    return corners_fixed, edges_fixed
end
function fix_relative_phases(
    U::Array{Ut,3}, V::Array{Vt,3}, signs
) where {Ut<:AbstractTensorMap,Vt<:AbstractTensorMap}
    U_fixed = map(Iterators.product(axes(U)...)) do (dir, r, c)
        if dir == NORTHWEST
            fix_gauge_north_left_vecs((r, c), U, signs)
        elseif dir == NORTHEAST
            fix_gauge_east_left_vecs((r, c), U, signs)
        elseif dir == SOUTHEAST
            fix_gauge_south_left_vecs((r, c), U, signs)
        elseif dir == SOUTHWEST
            fix_gauge_west_left_vecs((r, c), U, signs)
        end
    end

    V_fixed = map(Iterators.product(axes(V)...)) do (dir, r, c)
        if dir == NORTHWEST
            fix_gauge_north_right_vecs((r, c), V, signs)
        elseif dir == NORTHEAST
            fix_gauge_east_right_vecs((r, c), V, signs)
        elseif dir == SOUTHEAST
            fix_gauge_south_right_vecs((r, c), V, signs)
        elseif dir == SOUTHWEST
            fix_gauge_west_right_vecs((r, c), V, signs)
        end
    end

    return U_fixed, V_fixed
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

function calc_convergence(envs, CSold, TSold)
    CSnew = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envs.corners)
    ΔCS = maximum(zip(CSold, CSnew)) do (c_old, c_new)
        # only compute the difference on the smallest part of the spaces
        smallest = infimum(MPSKit._firstspace(c_old), MPSKit._firstspace(c_new))
        e_old = isometry(MPSKit._firstspace(c_old), smallest)
        e_new = isometry(MPSKit._firstspace(c_new), smallest)
        return norm(e_new' * c_new * e_new - e_old' * c_old * e_old)
    end

    TSnew = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envs.edges)
    ΔTS = maximum(zip(TSold, TSnew)) do (t_old, t_new)
        MPSKit._firstspace(t_old) == MPSKit._firstspace(t_new) ||
            return scalartype(t_old)(Inf)
        return norm(t_new - t_old)
    end

    @debug "maxᵢ|Cⁿ⁺¹ - Cⁿ|ᵢ = $ΔCS   maxᵢ|Tⁿ⁺¹ - Tⁿ|ᵢ = $ΔTS"

    return max(ΔCS, ΔTS), CSnew, TSnew
end

@non_differentiable calc_convergence(args...)

"""
    calc_elementwise_convergence(envfinal, envfix; atol=1e-6)

Check if the element-wise difference of the corner and edge tensors of the final and fixed
CTMRG environments are below some tolerance.
"""
function calc_elementwise_convergence(envfinal::CTMRGEnv, envfix::CTMRGEnv; atol::Real=1e-6)
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

    return max(ΔCmax, ΔTmax)
end

@non_differentiable calc_elementwise_convergence(args...)
