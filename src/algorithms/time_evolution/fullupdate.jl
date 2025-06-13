"""
$(TYPEDEF)

Algorithm struct for full update (FU) of infinite PEPS.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct FullUpdate
    "Time evolution step, such that the Trotter gate is exp(-dt * Hᵢⱼ).
    Use imaginary `dt` for real time evolution."
    dt::Number
    "Number of evolution steps without fully reconverging the environment."
    niter::Int
    "Fix gauge of bond environment."
    fixgauge::Bool = true
    "Bond truncation algorithm after applying time evolution gate."
    opt_alg::Union{ALSTruncation,FullEnvTruncation} = ALSTruncation(;
        trscheme=truncerr(1e-10)
    )
    "CTMRG algorithm to reconverge environment.
    Its `projector_alg` is also used for the fast update
    of the environment after each FU iteration."
    ctm_alg::CTMRGAlgorithm = SequentialCTMRG(;
        tol=1e-9,
        maxiter=20,
        verbosity=1,
        trscheme=truncerr(1e-10),
        projector_alg=:fullinfinite,
    )
end

"""
Full update for the bond between `[row, col]` and `[row, col+1]`.
"""
function _fu_xbond!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{T,S,2,2},
    peps::InfinitePEPS,
    env::CTMRGEnv,
    alg::FullUpdate,
) where {T<:Number,S<:ElementarySpace}
    cp1 = _next(col, size(peps, 2))
    A, B = peps[row, col], peps[row, cp1]
    X, a, b, Y = _qr_bond(A, B)
    # positive/negative-definite approximant: benv = ± Z Z†
    benv = bondenv_fu(row, col, X, Y, env)
    Z = positive_approx(benv)
    @debug "cond(benv) before gauge fix: $(LinearAlgebra.cond(Z' * Z))"
    # fix gauge
    if alg.fixgauge
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X, Y = _fixgauge_benvXY(X, Y, Linv, Rinv)
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(Z' * Z))"
    end
    benv = Z' * Z
    # apply gate
    need_flip = isdual(space(b, 1))
    a, s, b, = _apply_gate(a, b, gate, truncerr(1e-15))
    a, b = absorb_s(a, s, b)
    # optimize a, b
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)
    a, b = absorb_s(a, s, b)
    # bond truncation is done with arrow `a ← b`.
    # now revert back to `a → b` when needed.
    if need_flip
        a, b = flip(a, 3), flip(b, 1)
    end
    a /= norm(a, Inf)
    b /= norm(b, Inf)
    A, B = _qr_bond_undo(X, a, b, Y)
    peps.A[row, col] = A / norm(A, Inf)
    peps.A[row, cp1] = B / norm(B, Inf)
    return s, info
end

"""
Update all horizontal bonds in the c-th column
(i.e. `(r,c) (r,c+1)` for all `r = 1, ..., Nr`).
To update rows, rotate the network clockwise by 90 degrees.
The iPEPS `peps` is modified in place.
"""
function _fu_column!(
    col::Int, gate::LocalOperator, peps::InfinitePEPS, env::CTMRGEnv, alg::FullUpdate
)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    fid = 1.0
    wts_col = Vector{PEPSWeight}(undef, Nr)
    for row in 1:Nr
        term = get_gateterm(gate, (CartesianIndex(row, col), CartesianIndex(row, col + 1)))
        wts_col[row], info = _fu_xbond!(row, col, term, peps, env, alg)
        fid = min(fid, info.fid)
    end
    # update CTMRGEnv
    network = InfiniteSquareNetwork(peps)
    env2, info = ctmrg_leftmove(col, network, env, alg.ctm_alg.projector_alg)
    env2, info = ctmrg_rightmove(_next(col, Nc), network, env2, alg.ctm_alg.projector_alg)
    for c in [col, _next(col, Nc)]
        env.corners[:, :, c] = env2.corners[:, :, c]
        env.edges[:, :, c] = env2.edges[:, :, c]
    end
    return wts_col, fid
end

"""
One round of full update on the input InfinitePEPS `peps` and its CTMRGEnv `env`.

Reference: Physical Review B 92, 035142 (2015)
"""
function fu_iter(gate::LocalOperator, peps::InfinitePEPS, env::CTMRGEnv, alg::FullUpdate)
    Nr, Nc = size(peps)
    fidmin = 1.0
    peps2, env2 = deepcopy(peps), deepcopy(env)
    wts = Array{PEPSWeight}(undef, 2, Nr, Nc)
    for i in 1:4
        N = size(peps2, 2)
        for col in 1:N
            wts_col, fid_col = _fu_column!(col, gate, peps2, env2, alg)
            fid = min(fidmin, fid_col)
            # assign the weights to the un-rotated `wts`
            if i == 1
                wts[1, :, col] = wts_col
            elseif i == 2
                wts[2, _next(col, N), :] = reverse(wts_col)
            elseif i == 3
                wts[1, :, mod1(N - col, N)] = reverse(wts_col)
            else
                wts[2, N + 1 - col, :] = wts_col
            end
        end
        gate, peps2, env2 = rotl90(gate), rotl90(peps2), rotl90(env2)
    end
    return peps2, env2, SUWeight(collect(wt for wt in wts)), fidmin
end

"""
Full update an infinite PEPS with nearest neighbor Hamiltonian.
"""
function fu_iter2(ham::LocalOperator, peps::InfinitePEPS, env::CTMRGEnv, alg::FullUpdate)
    # Each NN bond is updated twice in _fu_iter, 
    # thus `dt` is divided by 2 when exponentiating `ham`.
    gate = get_expham(alg.dt / 2, ham)
    wts, fidmin = nothing, 1.0
    for it in 1:(alg.niter)
        peps, env, wts, fid = fu_iter(gate, peps, env, alg)
        fidmin = min(fidmin, fid)
    end
    # reconverge environment
    env, = leading_boundary(env, peps, alg.ctm_alg)
    return peps, env, wts, fidmin
end
