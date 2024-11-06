"""
Mirror the unit cell of an iPEPS by its anti-diagonal line
"""
function mirror_antidiag!(peps::InfinitePEPS)
    peps.A[:] = mirror_antidiag(peps.A)
    for (i, t) in enumerate(peps.A)
        peps.A[i] = permute(t, (1,), (3,2,5,4))
    end
end

"""
Mirror the unit cell of an iPEPS with weights by its anti-diagonal line
"""
function mirror_antidiag!(wts::SUWeight)
    wts.x[:], wts.y[:] = mirror_antidiag(wts.y), mirror_antidiag(wts.x)
end

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
    t::AbstractTensorMap, row::Int, col::Int, 
    ax::Int, wts::SUWeight; 
    sqrtwt::Bool=false, invwt::Bool=false
)
    Nr, Nc = size(wts)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert 2 <= ax <= 5
    pow = (sqrtwt ? 1/2 : 1) * (invwt ? -1 : 1)
    if ax == 2 # north
        wt = wts.y[row, col]
    elseif ax == 3 # east
        wt = wts.x[row, col]
    elseif ax == 4 # south
        wt = wts.y[_next(row,Nr), col]
    else # west
        wt = wts.x[row, _prev(col,Nc)]
    end
    wt2 = sdiag_pow(wt, pow)
    indices_t = collect(-1:-1:-5)
    indices_t[ax] = 1
    indices_wt = (ax in (2,3) ? [1,-ax] : [-ax,1])
    t2 = ncon((t, wt2), (indices_t, indices_wt))
    t2 = permute(t2, (1,), Tuple(2:5))
    return t2
end


"""
Simple update of bond `wts.x[r,c]`
```
                y[r,c]              y[r,c+1]
                ↓                   ↓
    x[r,c-1] ←- T[r,c] ←- x[r,c] ←- T[r,c+1] ← x[r,c+1]
                ↓                   ↓
                y[r+1,c]            y[r+1,c+1]
```
"""
function _su_bondx!(
    row::Int, col::Int, gate::AbstractTensorMap, 
    peps::InfinitePEPS, wts::SUWeight,
    Dcut::Int, svderr::Float64=1e-10
)
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    row2, col2 = row, _next(col,Nc)
    T1, T2 = peps[row,col], peps[row2,col2]
    # absorb environment weights
    for ax in (2,4,5)
        T1 = absorb_wt(T1, row, col, ax, wts)
    end
    for ax in (2,3,4)
        T2 = absorb_wt(T2, row2, col2, ax, wts)
    end
    # absorb bond weight
    T1 = absorb_wt(T1, row, col, 3, wts; sqrtwt=true)
    T2 = absorb_wt(T2, row2, col2, 5, wts; sqrtwt=true)
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
    X, aR = leftorth(T1, ((2,4,5), (1,3)), alg=QRpos())
    bL, Y = rightorth(T2, ((5,1), (2,3,4)), alg=LQpos())
    #= apply gate
    
            -2          -3
            ↑           ↑
            |----gate---|
            ↑           ↑
            1           2
            ↑           ↑
        -1← aR -← 3 -← bL ← -4
    =#
    tmp = ncon((gate, aR, bL), ([-2,-3,1,2], [-1,1,3], [3,2,-4]))
    # SVD
    truncscheme = truncerr(svderr) & truncdim(Dcut)
    aR, s, bL, ϵ = tsvd(tmp, ((1,2), (3,4)); trunc=truncscheme)
    #=
            -2         -1              -1    -2
            |         ↗               ↗       |
        -5- X ← 1 ← aR - -3     -5 - bL ← 1 ← Y - -3
            |                                 |
            -4                               -4
    =#
    T1 = ncon((X, aR), ([-2,-4,-5,1], [1,-1,-3]))
    T2 = ncon((bL, Y), ([-5,-1,1], [1,-2,-3,-4]))
    # remove environment weights
    for ax in (2,4,5)
        T1 = absorb_wt(T1, row, col, ax, wts; invwt=true)
    end
    for ax in (2,3,4)
        T2 = absorb_wt(T2, row2, col2, ax, wts; invwt=true)
    end
    # update tensor dict and weight on current bond 
    # (max element of weight is normalized to 1)
    peps.A[row,col], peps.A[row2,col2] = T1, T2
    wts.x[row,col] = s / maxabs(s)
    return ϵ
end


"""
One round of simple update on the input InfinitePEPS `peps` 
and SUWeight `wts` with the nearest neighbor gate `gate`

When `bipartite === true` (for square lattice), the unit cell size should be 2 x 2, 
and the tensor and x/y weight at `(row, col)` is the same as `(row+1, col+1)`
"""
function simpleupdate!(
    gate::AbstractTensorMap, peps::InfinitePEPS, wts::SUWeight, 
    Dcut::Int, svderr::Float64=1e-10; bipartite::Bool=false,
)
    Nr, Nc = size(peps)
    if bipartite
        @assert Nr == Nc == 2
    end
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.A[r, c], ax)) for ax in 1:5] == [0,1,1,0,0]
        @assert [isdual(space(wts.x[r, c], ax)) for ax in 1:2] == [0,1]
        @assert [isdual(space(wts.y[r, c], ax)) for ax in 1:2] == [0,1]
    end
    for direction in 1:2
        # mirror the y-weights to x-direction 
        # to update them using code for x-weights
        if direction == 2
            mirror_antidiag!(peps); mirror_antidiag!(wts)
        end
        if bipartite
            ϵ = _su_bondx!(1, 1, gate, peps, wts, Dcut, svderr)
            (peps.A[2,2], peps.A[2,1], wts.x[2,2]) = deepcopy.((peps.A[1,1], peps.A[1,2], wts.x[1,1]))
            ϵ = _su_bondx!(2, 1, gate, peps, wts, Dcut, svderr)
            (peps.A[1,2], peps.A[1,1], wts.x[1,2]) = deepcopy.((peps.A[2,1], peps.A[2,2], wts.x[2,1]))
        else
            for site in CartesianIndices(peps.A)
                row, col = Tuple(site)
                ϵ = _su_bondx!(row, col, gate, peps, wts, Dcut)
            end
        end
        if direction == 2
            mirror_antidiag!(peps); mirror_antidiag!(wts)
        end
    end
    return nothing
end

