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
    gate::AbstractTensorMap{S,2,2},
    peps::InfiniteWeightPEPS,
    Dcut::Int,
    svderr::Float64=1e-10,
) where {S}
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
    @tensor tmp[:] := gate[-2, -3, 1, 2] * aR[-1, 1, 3] * bL[3, 2, -4]
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
    @tensor T1[-1; -2 -3 -4 -5] := X[-2, -4, -5, 1] * aR[1, -1, -3]
    @tensor T2[-1; -2 -3 -4 -5] := bL[-5, -1, 1] * Y[1, -2, -3, -4]
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
function su_iter(
    gate::LocalOperator,
    peps::InfiniteWeightPEPS,
    Dcut::Int,
    svderr::Float64=1e-10;
    bipartite::Bool=false,
)
    @assert size(gate.lattice) == size(peps)
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
    peps2 = deepcopy(peps)
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
                    gate,
                    if direction == 1
                        (CartesianIndex(r, 1), CartesianIndex(r, 2))
                    else
                        #= the bond currently at [r, 1] [r, 2]
                        was originally at [2, 3-r] [1, 3-r] before mirroring =#
                        (CartesianIndex(2, 3 - r), CartesianIndex(1, 3 - r))
                    end,
                )
                ϵ = _su_bondx!(r, 1, term, peps2, Dcut, svderr)
                peps2.vertices[rp1, 2] = deepcopy(peps2.vertices[r, 1])
                peps2.vertices[rp1, 1] = deepcopy(peps2.vertices[r, 2])
                peps2.weights.x[rp1, 2] = deepcopy(peps2.weights.x[r, 1])
            end
        else
            for site in CartesianIndices(peps2.vertices)
                r, c = Tuple(site)
                term = get_gateterm(
                    gate,
                    if direction == 1
                        (CartesianIndex(r, c), CartesianIndex(r, c + 1))
                    else
                        #= the bond currently at [r, c] [r, c+1]
                        was originally at [Nr-c+1, Nc-r+1] [Nr-c, Nc-r+1]
                        before mirroring =#
                        (
                            CartesianIndex((c == Nr ? Nr + 1 : Nr - c + 1), Nc - r + 1),
                            CartesianIndex((c == Nr ? Nr : Nr - c), Nc - r + 1),
                        )
                    end,
                )
                ϵ = _su_bondx!(r, c, term, peps2, Dcut)
            end
        end
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
    end
    return peps2
end

"""
Perform simple update (maximum `evolstep` iterations)
with nearest neighbor Hamiltonian `ham` and time step `dt` 
until the change of bond weights is smaller than `wtdiff_tol` 
"""
function simpleupdate(
    peps::InfiniteWeightPEPS,
    ham::LocalOperator,
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
    # exponentiating the 2-site Hamiltonian gate
    gate = get_gate(dt, ham)
    wtdiff = 1e+3
    wts0 = deepcopy(peps.weights)
    for count in 1:evolstep
        time0 = time()
        peps = su_iter(gate, peps, Dcut, svderr; bipartite=bipartite)
        wtdiff = compare_weights(peps.weights, wts0)
        converge = wtdiff < wtdiff_tol
        cancel = count == evolstep
        wts0 = deepcopy(peps.weights)
        time1 = time()
        if ((count == 1) || (count % check_int == 0) || converge || cancel)
            label = (converge ? "conv" : (cancel ? "cancel" : "iter"))
            message = @sprintf(
                "SU %s %-7d:  dt = %.0e,  weight diff = %.3e,  time = %.3f sec\n",
                label,
                count,
                dt,
                wtdiff,
                time1 - ((converge || cancel) ? time_start : time0)
            )
            cancel ? (@warn message) : (@info message)
        end
        if converge || cancel
            break
        end
    end
    return peps, wtdiff
end
