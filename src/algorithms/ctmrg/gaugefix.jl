# TODO: add eigensolver algorithm, verbosity to gauge algorithm structs
"""
$(TYPEDEF)

CTMRG environment gauge fixing algorithm implementing the "general" technique from
https://arxiv.org/abs/2311.11894. This works by constructing a transfer matrix consisting
of an edge tensor and a random MPS, thus scrambling potential degeneracies, and then
performing a QR decomposition to extract the gauge signs. This is adapted accordingly for
asymmetric CTMRG algorithms using multi-site unit cell transfer matrices.
"""
struct ScramblingEnvGauge end

"""
$(SIGNATURES)

Fix the gauge of `envfinal` based on the previous environment `envprev`.
This assumes that the `envfinal` is the result of one CTMRG iteration on `envprev`.
Given that the CTMRG run is converged, the returned environment will be
element-wise converged to `envprev`.
The optional allocator supplies scratch space for the transfer and boundary contractions.
"""
function gauge_fix(
        envfinal::AbstractCTMRGEnv{C, T}, envprev::AbstractCTMRGEnv{C, T}, alg::ScramblingEnvGauge,
        allocator = DefaultAllocator(),
    ) where {C, T}
    signs = compute_relative_phases(envfinal, envprev, alg, allocator)
    return fix_global_phases(fix_relative_phases(envfinal, signs, allocator), envprev)
end

function _check_gauge_fix_spaces(envfinal::AbstractCTMRGEnv, envprev::AbstractCTMRGEnv)
    (axes(envfinal.corners) == axes(envprev.corners) && axes(envfinal.edges) == axes(envprev.edges)) ||
        throw(DimensionMismatch("Unit cells of envfinal and envprev do not match"))
    for (final, prev) in ((envfinal.corners, envprev.corners), (envfinal.edges, envprev.edges))
        same_spaces = all(zip(final, prev)) do (a, b)
            return space(a) == space(b)
        end
        same_spaces || throw(ArgumentError("Spaces of envfinal and envprev are not the same"))
    end
    return nothing
end

function compute_gauge_fix_gauge(
        envfinal::AbstractCTMRGEnv{C, T}, envprev::AbstractCTMRGEnv{C, T}, alg::ScramblingEnvGauge,
        allocator = DefaultAllocator(),
    ) where {C, T}
    signs = compute_relative_phases(envfinal, envprev, alg, allocator)
    corner_phases, edge_phases = compute_global_phases(
        fix_relative_phases(envfinal, signs, allocator), envprev
    )
    return signs, corner_phases, edge_phases
end

function fix_phases(env::AbstractCTMRGEnv, signs::CTMRGEnvGaugeTransform, corner_phases, edge_phases, allocator = DefaultAllocator())
    env_fixed_relative = fix_relative_phases(env, signs, allocator)
    env_fixed_global = fix_global_phases(env_fixed_relative, corner_phases, edge_phases)
    return env_fixed_global
end

function compute_relative_phases(
        envfinal::AbstractCTMRGEnv{C, T}, envprev::AbstractCTMRGEnv{C, T}, ::ScramblingEnvGauge,
        allocator = DefaultAllocator(),
    ) where {C, T}
    _check_gauge_fix_spaces(envfinal, envprev)
    # Julia 1.10 infers the gauge array with a comprehension.
    signs = [
        begin
            Tsprev = _gauge_transfer_edges(envprev.edges, coordinate)
            Tsfinal = _gauge_transfer_edges(envfinal.edges, coordinate)
            M = map(t -> randn(storagetype(t), space(t)), Tsfinal)
            _scrambling_gauge_matrix(Tsprev, Tsfinal, M, Arnoldi(), allocator)
        end for coordinate in eachcoordinate(envfinal, 1:4)
    ]

    return CTMRGEnvGaugeTransform(signs, adjoint.(signs))
end

function _gauge_transfer_edges(edges, (dir, r, c))
    dir == NORTH && return circshift(edges[dir, r, :], 1 - c)
    dir == EAST && return circshift(edges[dir, :, c], 1 - r)
    dir == SOUTH && return circshift(reverse(edges[dir, r, :]), c)
    dir == WEST && return circshift(reverse(edges[dir, :, c]), r)
    throw(ArgumentError(lazy"invalid direction $dir"))
end

function compute_relative_phases(
        envfinal::C4vCTMRGEnv{C, T}, envprev::C4vCTMRGEnv{C, T}, ::ScramblingEnvGauge,
        allocator = DefaultAllocator(),
    ) where {C, T}
    _check_gauge_fix_spaces(envfinal, envprev)
    # "general" algorithm from https://arxiv.org/abs/2311.11894
    Tprev = envprev.edges[1, 1, 1]
    Tfinal = envfinal.edges[1, 1, 1]

    # Random Hermitian MPS of same bond dimension
    # (make Hermitian such that T-M transfer matrix has real eigenvalues)
    M = _project_hermitian(randn(storagetype(Tfinal), space(Tfinal)))

    σ = _scrambling_gauge_matrix([Tprev], [Tfinal], [M], Lanczos(), allocator)
    return CTMRGEnvGaugeTransform(fill(σ, (4, 1, 1)), fill(σ', (4, 1, 1)))
end

"""Extract the relative unitary gauge from mixed-transfer fixed points with a shared random MPS."""
function _scrambling_gauge_matrix(Tsprev, Tsfinal, M, eigsolve_alg, allocator)
    ρprev = right_transfermatrix_fixedpoint(Tsprev, M, eigsolve_alg, initialize_right_fixedpoint(Tsprev, M), allocator)
    ρfinal = right_transfermatrix_fixedpoint(Tsfinal, M, eigsolve_alg, initialize_right_fixedpoint(Tsfinal, M), allocator)
    Qprev, = left_orth!(ρprev; positive = true)
    Qfinal, = left_orth!(ρfinal; positive = true)
    return Qprev * Qfinal'
end

# this is a bit of a hack to get the fixed point of the mixed transfer matrix
# because MPSKit is not compatible with AD
# NOTE: the action of the transfer operator here is NOT the same as that of
# MPSKit.transfer_right; to match the MPSKit implementation, it would require a twist on any
# index where spaces(Abar, 2:end) is dual (which we can just ignore as long as we use a random Abar)
@generated function mps_transfer_right(
        v::AbstractTensorMap{<:Any, S, 1, N₁},
        A::GenericMPSTensor{S, N₂}, Abar::GenericMPSTensor{S, N₂}, allocator,
    ) where {S, N₁, N₂}
    t_out = tensorexpr(:v, -1, -(2:(N₁ + 1)))
    t_top = tensorexpr(:A, (-1, reverse(3:(N₂ + 1))...), 1)
    t_bot = tensorexpr(:Abar, (-(N₁ + 1), reverse(3:(N₂ + 1))...), 2)
    t_in = tensorexpr(:v, 1, (-(2:N₁)..., 2))
    return macroexpand(
        @__MODULE__, :(return @tensor allocator = allocator $t_out := $t_top * conj($t_bot) * $t_in)
    )
end

function initialize_right_fixedpoint(tops, bottoms)
    ρ0 = randn(
        mapreduce(storagetype, TensorKit.promote_storagetype, tops),
        space(tops[end], numind(tops[end]))' ← space(bottoms[end], numind(bottoms[end]))'
    )
    return ρ0
end
function right_transfermatrix_fixedpoint(
        tops, bottoms, alg = Arnoldi(),
        ρ0 = initialize_right_fixedpoint(tops, bottoms),
        allocator = DefaultAllocator(),
    )
    _, vecs, info = eigsolve(ρ0, 1, :LM, alg) do ρ
        return foldr(zip(tops, bottoms); init = ρ) do (top, bottom), ρ
            return mps_transfer_right(ρ, top, bottom, allocator)
        end
    end
    ignore_derivatives() do
        info.converged > 0 || @warn "eigsolve did not converge"
    end
    return first(vecs)
end

fix_relative_phases(env::CTMRGEnv, signs::CTMRGEnvGaugeTransform, allocator = DefaultAllocator()) =
    gauge_transform(env, signs, allocator)

function fix_relative_phases(env::C4vCTMRGEnv, signs::CTMRGEnvGaugeTransform, allocator = DefaultAllocator())
    X, Xinv = signs.matrices, signs.inverses
    corners = map(env.corners) do C
        return absorb_left_right(C, X[WEST, 1, 1], Xinv[NORTH, 1, 1], allocator)
    end
    edges = map(eachcoordinate(env, 1:4)) do (dir, r, c)
        return absorb_left_right(env.edges[dir, r, c], X[dir, r, c], Xinv[dir, r, c], allocator)
    end
    return C4vCTMRGEnv(corners, edges)
end

function fix_relative_phases(
        U::Array{Ut, 3}, V::Array{Vt, 3}, signs::CTMRGEnvGaugeTransform
    ) where {Ut <: AbstractTensorMap, Vt <: AbstractTensorMap}
    nr, nc = size(U)[2:3]
    U_fixed = map(eachindex(IndexCartesian(), U)) do I
        d, r, c = Tuple(I)
        _, ri, ci = _next_coordinate((d, r, c), nr, nc)
        return U[I] * signs.inverses[d, ri, ci]
    end
    V_fixed = map(eachindex(IndexCartesian(), V)) do I
        d, r, c = Tuple(I)
        _, ri, ci = _next_coordinate((d, r, c), nr, nc)
        return signs.matrices[d, ri, ci] * V[I]
    end
    return U_fixed, V_fixed
end

"""
$(SIGNATURES)

Fix global multiplicative phase of the environment tensors. To that end, the dot products
between all corners and all edges are computed to obtain the global phase which is then
divided out.
"""
function fix_global_phases(envfix::AbstractCTMRGEnv, envprev::AbstractCTMRGEnv)
    corner_phases, edge_phases = compute_global_phases(envfix, envprev)
    return fix_global_phases(envfix, corner_phases, edge_phases)
end
function fix_global_phases(env::CTMRGEnv, corner_phases, edge_phases)
    cornersgfix = env.corners .* inv.(corner_phases)
    edgesgfix = env.edges .* inv.(edge_phases)
    return CTMRGEnv(cornersgfix, edgesgfix)
end
function fix_global_phases(env::C4vCTMRGEnv, corner_phases, edge_phases)
    cornersgfix = env.corners .* inv.(corner_phases)
    edgesgfix = env.edges .* inv.(edge_phases)
    return C4vCTMRGEnv(cornersgfix, edgesgfix)
end

function compute_global_phases(envfix::AbstractCTMRGEnv, envprev::AbstractCTMRGEnv)
    corner_phases = dot.(envprev.corners, envfix.corners)
    edge_phases = dot.(envprev.edges, envfix.edges)
    return corner_phases, edge_phases
end

"""
    calc_elementwise_convergence(envfinal, envfix; atol=1e-6)

Check if the element-wise difference of the corner and edge tensors of the final and fixed
CTMRG environments are below `atol` and return the maximal difference.
"""
function calc_elementwise_convergence(
        envfinal::AbstractCTMRGEnv{C}, envfix::AbstractCTMRGEnv{C′}; atol::Real = 1.0e-6
    ) where {C, C′}
    Cfinal = (C <: DiagonalTensorMap) ? convert.(TensorMap, envfinal.corners) : envfinal.corners
    Cfix = (C′ <: DiagonalTensorMap) ? convert.(TensorMap, envfix.corners) : envfix.corners
    ΔC = Cfinal .- Cfix
    ΔCmax = norm(ΔC, Inf)
    ΔCmean = norm(ΔC)
    @debug "maxᵢⱼ|Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmax   mean |Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmean"

    ΔT = envfinal.edges .- envfix.edges
    ΔTmax = norm(ΔT, Inf)
    ΔTmean = norm(ΔT)
    @debug "maxᵢⱼ|Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmax   mean |Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmean"

    # Check differences for all tensors in unit cell to debug properly
    for I in eachindex(IndexCartesian(), ΔT)
        dir, r, c = Tuple(I)
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
