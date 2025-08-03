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
    trscheme::TruncationScheme
end
# TODO: add kwarg constructor and SU Defaults

"""
$(SIGNATURES)

Simple update of the x-bond between `[r,c]` and `[r,c+1]`.

```
        |           |
    -- T[r,c] -- T[r,c+1] --
        |           |
```
"""
function _su_xbond!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfinitePEPS,
    env::SUWeight,
    trscheme::TruncationScheme,
) where {T<:Number,S<:ElementarySpace}
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    cp1 = _next(col, Nc)
    # absorb environment weights
    A, B = peps.A[row, col], peps.A[row, cp1]
    A = _absorb_weights(A, row, col, (NORTH, SOUTH, WEST), env; invwt=false)
    B = _absorb_weights(B, row, cp1, (NORTH, SOUTH, EAST), env; invwt=false)
    normalize!(A, Inf)
    normalize!(B, Inf)
    # apply gate
    X, a, b, Y = _qr_bond(A, B)
    a, s, b, ϵ = _apply_gate(a, b, gate, trscheme)
    A, B = _qr_bond_undo(X, a, b, Y)
    # remove environment weights
    A = _absorb_weights(A, row, col, (NORTH, SOUTH, WEST), env; invwt=true)
    B = _absorb_weights(B, row, cp1, (NORTH, SOUTH, EAST), env; invwt=true)
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    # update tensor dict and weight on current bond 
    peps.A[row, col], peps.A[row, cp1] = A, B
    env.data[1, row, col] = s
    return ϵ
end

"""
Simple update of the y-bond between `[r,c]` and `[r-1,c]`.
```
        |
    --T[r-1,c] --
        |
    -- T[r,c] ---
        |
```
"""
function _su_ybond!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfinitePEPS,
    env::SUWeight,
    trscheme::TruncationScheme,
) where {T<:Number,S<:ElementarySpace}
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    rm1 = _prev(row, Nr)
    # absorb environment weights
    A, B = peps.A[row, col], peps.A[rm1, col]
    A = _absorb_weights(A, row, col, (EAST, SOUTH, WEST), env; invwt=false)
    B = _absorb_weights(B, rm1, col, (NORTH, EAST, WEST), env; invwt=false)
    normalize!(A, Inf)
    normalize!(B, Inf)
    # apply gate
    X, a, b, Y = _qr_bond(rotr90(A), rotr90(B))
    a, s, b, ϵ = _apply_gate(a, b, gate, trscheme)
    A, B = rotl90.(_qr_bond_undo(X, a, b, Y))
    # remove environment weights
    A = _absorb_weights(A, row, col, (EAST, SOUTH, WEST), env; invwt = true)
    B = _absorb_weights(B, rm1, col, (NORTH, EAST, WEST), env; invwt = true)
    # update tensor dict and weight on current bond 
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    peps.A[row, col], peps.A[rm1, col] = A, B
    env.data[2, row, col] = s
    return ϵ
end

"""
    su_iter(gate::LocalOperator, peps::InfinitePEPS, env::SUWeight, alg::SimpleUpdate; bipartite::Bool=false)

One round of simple update on `peps` applying the nearest neighbor `gate`.
When the input `peps` has a unit cell size of (2, 2), one can set `bipartite = true` to enforce the bipartite structure. 
"""
function su_iter(
    gate::LocalOperator,
    peps::InfinitePEPS,
    env::SUWeight,
    alg::SimpleUpdate;
    bipartite::Bool=false,
)
    @assert size(gate.lattice) == size(peps)
    Nr, Nc = size(peps)
    bipartite && (@assert Nr == Nc == 2)
    (Nr >= 2 && Nc >= 2) || throw(
        ArgumentError(
            "iPEPS unit cell size for simple update should be no smaller than (2, 2)."
        ),
    )
    peps2, env2 = deepcopy(peps), deepcopy(env)
    for r in 1:Nr, c in 1:Nc
        term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r, c + 1)))
        trscheme = truncation_scheme(alg.trscheme, 1, r, c)
        _su_xbond!(r, c, term, peps2, env2, trscheme)
        if bipartite
            rp1, cp1 = _next(r, Nr), _next(c, Nc)
            peps2.A[rp1, cp1] = deepcopy(peps2.A[r, c])
            peps2.A[rp1, c] = deepcopy(peps2.A[r, cp1])
            env2.data[1, rp1, cp1] = deepcopy(env2.data[1, r, c])
        end
        term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r - 1, c)))
        trscheme = truncation_scheme(alg.trscheme, 2, r, c)
        _su_ybond!(r, c, term, peps2, env2, trscheme)
        if bipartite
            rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
            peps2.A[rm1, cm1] = deepcopy(peps2.A[r, c])
            peps2.A[r, cm1] = deepcopy(peps2.A[rm1, c])
            env2.data[2, rm1, cm1] = deepcopy(env2.data[2, r, c])
        end
    end
    return peps2, env2
end

"""
Perform simple update with Hamiltonian `ham` containing up to nearest neighbor interaction terms. 
"""
function _simpleupdate2site(
    peps::InfinitePEPS,
    env::SUWeight,
    ham::LocalOperator,
    alg::SimpleUpdate;
    bipartite::Bool=false,
    check_interval::Int=500,
)
    time_start = time()
    # exponentiating the 2-site Hamiltonian gate
    gate = get_expham(ham, alg.dt)
    wtdiff = 1.0
    env0 = deepcopy(env)
    for count in 1:(alg.maxiter)
        time0 = time()
        peps, env = su_iter(gate, peps, env, alg; bipartite)
        wtdiff = compare_weights(env, env0)
        converge = (wtdiff < alg.tol)
        cancel = (count == alg.maxiter)
        env0 = deepcopy(env)
        time1 = time()
        if ((count == 1) || (count % check_interval == 0) || converge || cancel)
            @info "Space of x-weight at [1, 1] = $(space(env[1, 1, 1], 1))"
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
    return peps, env, wtdiff
end

"""
    simpleupdate(peps::InfinitePEPS, env::SUWeight, ham::LocalOperator, alg::SimpleUpdate;
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
    peps::InfinitePEPS,
    env::SUWeight,
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
    # TODO: check SiteDependentTruncation is compatible with bipartite structure
    if use_3site
        return _simpleupdate3site(peps, env, ham, alg; check_interval)
    else
        return _simpleupdate2site(peps, env, ham, alg; bipartite, check_interval)
    end
end
