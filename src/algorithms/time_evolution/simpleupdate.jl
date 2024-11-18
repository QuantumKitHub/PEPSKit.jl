"""
Absorb environment weight on axis `ax` into tensor `t` at position `(row,col)`

Weights around the tensor at `(row, col)` are
```
                ↓
                y[r,c]
                ↓
    ←x[r,c-1] ← T[r,c] ← x[r,c] ←
                ↓
                y[r+1,c]
                ↓
```
"""
function absorb_wt(
    t::AbstractTensorMap,
    row::Int,
    col::Int,
    ax::Int,
    weights::SUWeight;
    sqrtwt::Bool=false,
    invwt::Bool=false,
)
    Nr, Nc = size(weights)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert 2 <= ax <= 5
    pow = (sqrtwt ? 1 / 2 : 1) * (invwt ? -1 : 1)
    if ax == 2 # north
        wt = weights.y[row, col]
    elseif ax == 3 # east
        wt = weights.x[row, col]
    elseif ax == 4 # south
        wt = weights.y[_next(row, Nr), col]
    else # west
        wt = weights.x[row, _prev(col, Nc)]
    end
    wt2 = sdiag_pow(wt, pow)
    indices_t = collect(-1:-1:-5)
    indices_t[ax] = 1
    indices_wt = (ax in (2, 3) ? [1, -ax] : [-ax, 1])
    t2 = ncon((t, wt2), (indices_t, indices_wt))
    t2 = permute(t2, (1,), Tuple(2:5))
    return t2
end

"""
Simple update of bond `peps.weights.x[r,c]`
```
                y[r,c]              y[r,c+1]
                ↓                   ↓
    x[r,c-1] ←- T[r,c] ←- x[r,c] ←- T[r,c+1] ← x[r,c+1]
                ↓                   ↓
                y[r+1,c]            y[r+1,c+1]
```
"""
function _su_bondx!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap,
    peps::InfiniteWeightPEPS,
    Dcut::Int,
    svderr::Float64=1e-10,
)
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    row2, col2 = row, _next(col, Nc)
    T1, T2 = peps.vertices[row, col], peps.vertices[row2, col2]
    # absorb environment weights
    for ax in (2, 4, 5)
        T1 = absorb_wt(T1, row, col, ax, peps.weights)
    end
    for ax in (2, 3, 4)
        T2 = absorb_wt(T2, row2, col2, ax, peps.weights)
    end
    # absorb bond weight
    T1 = absorb_wt(T1, row, col, 3, peps.weights; sqrtwt=true)
    T2 = absorb_wt(T2, row2, col2, 5, peps.weights; sqrtwt=true)
    #= QR and LQ decomposition

        2   1               1             2
        ↓ ↗                 ↓            ↗
    5 ← T ← 3   ====>   3 ← X ← 4 ← 1 ← aR ← 3
        ↓                   ↓
        4                   2

        2   1                 2         2
        ↓ ↗                 ↗           ↓
    5 ← T ← 3   ====>  1 ← bL ← 3 ← 1 ← Y ← 3
        ↓                               ↓
        4                               4
    =#
    X, aR = leftorth(T1, ((2, 4, 5), (1, 3)); alg=QRpos())
    bL, Y = rightorth(T2, ((5, 1), (2, 3, 4)); alg=LQpos())
    #= apply gate

            -2          -3
            ↑           ↑
            |----gate---|
            ↑           ↑
            1           2
            ↑           ↑
        -1← aR -← 3 -← bL ← -4
    =#
    tmp = ncon((gate, aR, bL), ([-2, -3, 1, 2], [-1, 1, 3], [3, 2, -4]))
    # SVD
    truncscheme = truncerr(svderr) & truncdim(Dcut)
    aR, s, bL, ϵ = tsvd(tmp, ((1, 2), (3, 4)); trunc=truncscheme)
    #=
            -2         -1              -1    -2
            |         ↗               ↗       |
        -5- X ← 1 ← aR - -3     -5 - bL ← 1 ← Y - -3
            |                                 |
            -4                               -4
    =#
    T1 = ncon((X, aR), ([-2, -4, -5, 1], [1, -1, -3]))
    T2 = ncon((bL, Y), ([-5, -1, 1], [1, -2, -3, -4]))
    # remove environment weights
    for ax in (2, 4, 5)
        T1 = absorb_wt(T1, row, col, ax, peps.weights; invwt=true)
    end
    for ax in (2, 3, 4)
        T2 = absorb_wt(T2, row2, col2, ax, peps.weights; invwt=true)
    end
    # update tensor dict and weight on current bond 
    # (max element of weight is normalized to 1)
    peps.vertices[row, col], peps.vertices[row2, col2] = T1, T2
    peps.weights.x[row, col] = s / norm(s, Inf)
    return ϵ
end

"""
One round of simple update on the input 
InfiniteWeightPEPS `peps` with the nearest neighbor gate `gate`

When `bipartite === true` (for square lattice), the unit cell size should be 2 x 2, 
and the tensor and x/y weight at `(row, col)` is the same as `(row+1, col+1)`
"""
function su_iter!(
    gate::AbstractTensorMap,
    peps::InfiniteWeightPEPS,
    Dcut::Int,
    svderr::Float64=1e-10;
    bipartite::Bool=false,
)
    Nr, Nc = size(peps)
    if bipartite
        @assert Nr == Nc == 2
    end
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.vertices[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
        @assert [isdual(space(peps.weights.x[r, c], ax)) for ax in 1:2] == [0, 1]
        @assert [isdual(space(peps.weights.y[r, c], ax)) for ax in 1:2] == [0, 1]
    end
    for direction in 1:2
        # mirror the y-weights to x-direction 
        # to update them using code for x-weights
        if direction == 2
            mirror_antidiag!(peps)
        end
        if bipartite
            ϵ = _su_bondx!(1, 1, gate, peps, Dcut, svderr)
            (peps.vertices[2, 2], peps.vertices[2, 1], peps.weights.x[2, 2]) =
                deepcopy.((peps.vertices[1, 1], peps.vertices[1, 2], peps.weights.x[1, 1]))
            ϵ = _su_bondx!(2, 1, gate, peps, Dcut, svderr)
            (peps.vertices[1, 2], peps.vertices[1, 1], peps.weights.x[1, 2]) =
                deepcopy.((peps.vertices[2, 1], peps.vertices[2, 2], peps.weights.x[2, 1]))
        else
            for site in CartesianIndices(peps.vertices)
                row, col = Tuple(site)
                ϵ = _su_bondx!(row, col, gate, peps, Dcut)
            end
        end
        if direction == 2
            mirror_antidiag!(peps)
        end
    end
    return nothing
end

function compare_weights(wts1::SUWeight, wts2::SUWeight)
    wtdiff = sum(_singular_value_distance((wt1, wt2)) for (wt1, wt2) in zip(wts1, wts2))
    return wtdiff / (2 * prod(size(wts1)))
end

"""
Perform simple update (maximum `evolstep` iterations)
with nearest neighbor Hamiltonian `ham` and time step `dt` 
until the change of bond weights is smaller than `wtdiff_tol` 
"""
function simpleupdate!(
    peps::InfiniteWeightPEPS,
    ham::AbstractTensorMap,
    dt::Float64,
    Dcut::Int;
    evolstep::Int=400000,
    svderr::Float64=1e-10,
    wtdiff_tol::Float64=1e-10,
    bipartite::Bool=false,
    check_int::Int=500,
)
    time_start = time()
    N1, N2 = size(peps)
    if bipartite
        @assert N1 == N2 == 2
    end
    @printf("%-9s%6s%12s  %s\n", "Step", "dt", "wt_diff", "speed/s")
    # exponentiating the 2-site Hamiltonian gate
    gate = exp(-dt * ham)
    wtdiff = 1e+3
    wts0 = deepcopy(peps.weights)
    for count in 1:evolstep
        time0 = time()
        su_iter!(gate, peps, Dcut, svderr; bipartite=bipartite)
        wtdiff = compare_weights(peps.weights, wts0)
        stop = (wtdiff < wtdiff_tol) || (count == evolstep)
        wts0 = deepcopy(peps.weights)
        time1 = time()
        if ((count == 1) || (count % check_int == 0) || stop)
            @printf("%-9d%6.0e%12.3e  %.3f\n", count, dt, wtdiff, time1 - time0)
        end
        if stop
            break
        end
    end
    time_end = time()
    @printf("Evolution time: %.2f s\n\n", time_end - time_start)
    return wtdiff
end
