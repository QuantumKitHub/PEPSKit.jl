"""
    struct SimpleUpdate

Algorithm struct for simple update (SU) of infinite PEPS with bond weights.
Each SU run is converged when the singular value difference becomes smaller than `tol`.
"""
struct SimpleUpdate
    dt::Float64
    tol::Float64
    maxiter::Int
    trscheme::TensorKit.TruncationScheme
end
# TODO: add kwarg constructor and SU Defaults

"""
_su_bondx!(row::Int, col::Int, gate::AbstractTensorMap{T,S,2,2},
           peps::InfiniteWeightPEPS, alg::SimpleUpdate) where {S<:ElementarySpace}

Simple update of the x-bond `peps.weights[1,r,c]`.

```
                [2,r,c]             [2,r,c+1]
                ↓                   ↓
    [1,r,c-1] ← T[r,c] ← [1,r,c] ←- T[r,c+1] ← [1,r,c+1]
                ↓                   ↓
                [2,r+1,c]           [2,r+1,c+1]
```
"""
function _su_bondx!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfiniteWeightPEPS,
    alg::SimpleUpdate,
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
    a, s, b, ϵ = _apply_gate(a, b, gate, alg.trscheme)
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
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.vertices[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
        @assert [isdual(space(peps.weights[1, r, c], ax)) for ax in 1:2] == [0, 1]
        @assert [isdual(space(peps.weights[2, r, c], ax)) for ax in 1:2] == [0, 1]
    end
    peps2 = deepcopy(peps)
    gate_mirrored = mirror_antidiag(gate)
    for direction in 1:2
        # mirror the y-weights to x-direction 
        # to update them using code for x-weights
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
        if bipartite
            for r in 1:2
                rp1 = _next(r, 2)
                term = get_gateterm(
                    direction == 1 ? gate : gate_mirrored,
                    (CartesianIndex(r, 1), CartesianIndex(r, 2)),
                )
                ϵ = _su_bondx!(r, 1, term, peps2, alg)
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
                ϵ = _su_bondx!(r, c, term, peps2, alg)
            end
        end
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
    end
    return peps2
end

"""
    simpleupdate(peps::InfiniteWeightPEPS, ham::LocalOperator, alg::SimpleUpdate;
                 bipartite::Bool=false, check_interval::Int=500)

Perform simple update with nearest neighbor Hamiltonian `ham`, where the evolution
information is printed every `check_interval` steps. 

If `bipartite == true` (for square lattice), a unit cell size of `(2, 2)` is assumed, 
as well as tensors and x/y weights which are the same across the diagonals, i.e. at
`(row, col)` and `(row+1, col+1)`.
"""
function simpleupdate(
    peps::InfiniteWeightPEPS,
    ham::LocalOperator,
    alg::SimpleUpdate;
    bipartite::Bool=false,
    check_interval::Int=500,
)
    time_start = time()
    # exponentiating the 2-site Hamiltonian gate
    gate = get_expham(alg.dt, ham)
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
