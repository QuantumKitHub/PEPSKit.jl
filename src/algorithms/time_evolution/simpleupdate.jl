"""
$(TYPEDEF)

Algorithm struct for simple update (SU) of infinite PEPS with bond weights.
Each SU run is converged when the singular value difference becomes smaller than `tol`.

## Fields

$(TYPEDFIELDS)
"""
struct SimpleUpdate
    dt::Number
    tol::Float64
    maxiter::Int
    trscheme::TensorKit.TruncationScheme
end
# TODO: add kwarg constructor and SU Defaults

"""
$(SIGNATURES)

Simple update of the x-bond `peps.weights[1,r,c]`.

```
                [2,r,c]             [2,r,c+1]
                ↓                   ↓
    [1,r,c-1] ← T[r,c] ← [1,r,c] ←- T[r,c+1] ← [1,r,c+1]
                ↓                   ↓
                [2,r+1,c]           [2,r+1,c+1]
```
"""
function _su_xbond!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfiniteWeightPEPS,
    trscheme::TensorKit.TruncationScheme,
) where {T<:Number,S<:ElementarySpace}
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    cp1 = _next(col, Nc)
    # absorb environment weights
    A, B = peps.vertices[row, col], peps.vertices[row, cp1]
    sqrtsA = ntuple(dir -> (dir == EAST), 4)
    sqrtsB = ntuple(dir -> (dir == WEST), 4)
    A = _absorb_weights(A, peps.weights, row, col, Tuple(1:4), sqrtsA, false)
    B = _absorb_weights(B, peps.weights, row, cp1, Tuple(1:4), sqrtsB, false)
    # apply gate
    X, a, b, Y = _qr_bond(A, B)
    a, s, b, ϵ = _apply_gate(a, b, gate, trscheme)
    A, B = _qr_bond_undo(X, a, b, Y)
    # remove environment weights
    _allfalse = ntuple(Returns(false), 3)
    A = _absorb_weights(A, peps.weights, row, col, (NORTH, SOUTH, WEST), _allfalse, true)
    B = _absorb_weights(B, peps.weights, row, cp1, (NORTH, SOUTH, EAST), _allfalse, true)
    # update tensor dict and weight on current bond 
    # (max element of weight is normalized to 1)
    peps.vertices[row, col], peps.vertices[row, cp1] = A, B
    peps.weights[1, row, col] = s / norm(s, Inf)
    return ϵ
end

"""
    su_iter(gate::LocalOperator, peps::InfiniteWeightPEPS, alg::SimpleUpdate; bipartite::Bool=false)

One round of simple update on `peps` applying the nearest neighbor `gate`.
When the input `peps` has a unit cell size of (2, 2), one can set `bipartite = true` to enforce the bipartite structure. 
"""
function su_iter(
    gate::LocalOperator, peps::InfiniteWeightPEPS, alg::SimpleUpdate; bipartite::Bool=false
)
    @assert size(gate.lattice) == size(peps)
    Nr, Nc = size(peps)
    if bipartite
        @assert Nr == Nc == 2
    end
    (Nr >= 2 && Nc >= 2) || throw(
        ArgumentError(
            "iPEPS unit cell size for simple update should be no smaller than (2, 2)."
        ),
    )
    peps2 = deepcopy(peps)
    gate_mirrored = mirror_antidiag(gate)
    for direction in 1:2
        # mirror the y-weights to x-direction 
        # to update them using code for x-weights
        if direction == 2
            peps2 = mirror_antidiag(peps2)
            trscheme = mirror_antidiag(alg.trscheme)
        else
            trscheme = alg.trscheme
        end
        if bipartite
            for r in 1:2
                rp1 = _next(r, 2)
                term = get_gateterm(
                    direction == 1 ? gate : gate_mirrored,
                    (CartesianIndex(r, 1), CartesianIndex(r, 2)),
                )
                ϵ = _su_xbond!(r, 1, term, peps2, truncation_scheme(trscheme, 1, r, 1))
                peps2.vertices[rp1, 2] = deepcopy(peps2.vertices[r, 1])
                peps2.vertices[rp1, 1] = deepcopy(peps2.vertices[r, 2])
                peps2.weights[1, rp1, 2] = deepcopy(peps2.weights[1, r, 1])
            end
        else
            for site in CartesianIndices(peps2.vertices)
                r, c = Tuple(site)
                term = get_gateterm(
                    direction == 1 ? gate : gate_mirrored,
                    (CartesianIndex(r, c), CartesianIndex(r, c + 1)),
                )
                ϵ = _su_xbond!(r, c, term, peps2, truncation_scheme(trscheme, 1, r, c))
            end
        end
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
    end
    return peps2
end

"""
Perform simple update with Hamiltonian `ham` containing up to nearest neighbor interaction terms. 
"""
function _simpleupdate2site(
    peps::InfiniteWeightPEPS,
    ham::LocalOperator,
    alg::SimpleUpdate;
    bipartite::Bool=false,
    check_interval::Int=500,
)
    time_start = time()
    # exponentiating the 2-site Hamiltonian gate
    gate = get_expham(ham, alg.dt)
    wtdiff = 1.0
    wts0 = deepcopy(peps.weights)
    for count in 1:(alg.maxiter)
        time0 = time()
        peps = su_iter(gate, peps, alg; bipartite)
        wtdiff = compare_weights(peps.weights, wts0)
        converge = (wtdiff < alg.tol)
        cancel = (count == alg.maxiter)
        wts0 = deepcopy(peps.weights)
        time1 = time()
        if ((count == 1) || (count % check_interval == 0) || converge || cancel)
            @info "Space of x-weight at [1, 1] = $(space(peps.weights[1, 1, 1], 1))"
            label = (converge ? "conv" : (cancel ? "cancel" : "iter"))
            message = @sprintf(
                "SU %s %-7d:  dt = %.0e,  weight diff = %.3e,  time = %.3f sec\n",
                label,
                count,
                alg.dt,
                wtdiff,
                time1 - ((converge || cancel) ? time_start : time0)
            )
            cancel ? (@warn message) : (@info message)
        end
        converge && break
    end
    return peps, wtdiff
end

"""
    simpleupdate(peps::InfiniteWeightPEPS, ham::LocalOperator, alg::SimpleUpdate;
                 bipartite::Bool=false, force_3site::Bool=false, check_interval::Int=500)

Perform a simple update on the infinite PEPS (`peps`) using the Hamiltonian `ham`, which can contain up to next-nearest-neighbor interaction terms.

## Keyword Arguments

- `bipartite::Bool=false`: If `true`, enforces the bipartite structure of the PEPS. This assumes the input `peps` has a unit cell size of (2, 2). 
- `force_3site::Bool=false`: Forces the use of the 3-site simple update algorithm, even if the Hamiltonian contains only nearest-neighbor terms.
- `check_interval::Int=500`: Specifies the number of evolution steps between printing progress information.

## Notes

- The 3-site simple update algorithm is incompatible with a bipartite PEPS. Using `bipartite = true` with either `force_3site = true` or a `ham` with next-nearest neighbor terms is not allowed. 
"""
function simpleupdate(
    peps::InfiniteWeightPEPS,
    ham::LocalOperator,
    alg::SimpleUpdate;
    bipartite::Bool=false,
    force_3site::Bool=false,
    check_interval::Int=500,
)
    # determine if Hamiltonian contains nearest neighbor terms only
    nnonly = is_nearest_neighbour(ham)
    use_3site = force_3site || !nnonly
    @assert !(bipartite && use_3site) "3-site simple update is incompatible with bipartite lattice."
    bipartite &&
        @assert size(peps) == (2, 2) "Bipartite structure is only compatible with square unit cells."
    if use_3site
        return _simpleupdate3site(peps, ham, alg; check_interval)
    else
        return _simpleupdate2site(peps, ham, alg; bipartite, check_interval)
    end
end
