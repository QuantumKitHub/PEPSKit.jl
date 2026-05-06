# Apply a gate - belief-propagation style
# ---------------------------------------

const Gate{N, T, S} = AbstractTensorMap{T, S, N, N}

@doc """
    apply(state, circuit::LocalCircuit, alg::SimpleUpdate, messages::BPEnv) -> state, messages, ϵ
    apply!(state, circuit::LocalCircuit, alg::SimpleUpdate, messages::BPEnv) -> state, messages, ϵ
    apply(state, (sites, gate)::Pair, alg::SimpleUpdate, messages::BPEnv) -> state, messages, ϵ
    apply!(state, (sites, gate)::Pair, alg::SimpleUpdate, messages::BPEnv) -> state, messages, ϵ

Apply a gate (or a `LocalCircuit` of gates) to `state` using belief-propagation simple update.
This means that the square roots of the surrounding messages are absorbed into the acted-on PEPS tensors,
the gate is applied on the reduced bond, the result is truncated by SVD according to `alg.trunc`,
and the new singular value spectrum replaces the bond message between the acted sites.

## Returns

- `state`: updated `InfinitePEPS`.
- `messages`: updated `BPEnv`
- `ϵ`: (maximal) local truncation error.
""" apply(state, site_gate, alg, messages::BPEnv), apply!(state, site_gate, alg, messages::BPEnv)

apply(state, site_gate, alg, messages::BPEnv) =
    apply!(copy(state), site_gate, alg, copy(messages))

function apply!(state, circuit::LocalCircuit, alg, messages::BPEnv)
    ϵ = zero(real(scalartype(state)))
    for (site, gate) in circuit.gates
        state, messages, ϵ_local = apply!(state, (site => gate), alg, messages)
        ϵ = max(ϵ, ϵ_local)
    end
    return state, messages, ϵ
end
function apply!(
        state, (sites, gate)::Pair{Vector{CartesianIndex{2}}, <:Gate}, alg, messages::BPEnv
    )
    if length(sites) == 1
        return apply_gate_1x1!(state, sites, gate, alg, messages)
    elseif length(sites) == 2
        diff = sites[2] - sites[1]
        if diff == CartesianIndex(-1, 0) || diff == CartesianIndex(1, 0)
            return apply_gate_2x1!(state, sites, gate, alg, messages)
        elseif diff == CartesianIndex(0, 1) || diff == CartesianIndex(0, -1)
            return apply_gate_1x2!(state, sites, gate, alg, messages)
        end
    end

    error("Not implemented: $sites")
end

function apply_gate_1x1!(
        state, sites::Vector{CartesianIndex{2}}, gate::Gate{1},
        alg::SimpleUpdate, messages::BPEnv
    )
    length(sites) == 1 || throw(ArgumentError("invalid sites: $sites"))
    site = only(sites)
    state[site] = gate * state[site]
    return state, messages, zero(real(scalartype(state)))
end

function apply_gate_1x2!(
        state, sites::Vector{CartesianIndex{2}}, gate::Gate{2}, alg::SimpleUpdate, messages::BPEnv
    )
    length(sites) == 2 || throw(ArgumentError("invalid sites: $sites"))

    # Note that we must truncate along the canonical arrow direction to not alter the result,
    # so we use west - east here and map east - west to the canonical direction.
    diff = sites[2] - sites[1]
    if diff == CartesianIndex(0, -1)
        sites = reverse(sites)
        gate = permute(gate, ((2, 1), (4, 3)))
    elseif diff != CartesianIndex(0, 1)
        throw(ArgumentError("invalid sites: $sites"))
    end

    # extract tensors and messages
    siteL = CartesianIndex(mod1.(Tuple(sites[1]), (size(state, 1), size(state, 2)))...)
    siteR = CartesianIndex(mod1.(Tuple(sites[2]), (size(state, 1), size(state, 2)))...)
    A_L = state[siteL]
    A_R = state[siteR]
    MN_L = messages[NORTH, siteL - CartesianIndex(1, 0)]
    MN_R = messages[NORTH, siteR - CartesianIndex(1, 0)]
    ME = messages[EAST, siteR + CartesianIndex(0, 1)]
    MS_L = messages[SOUTH, siteL + CartesianIndex(1, 0)]
    MS_R = messages[SOUTH, siteR + CartesianIndex(1, 0)]
    MW = messages[WEST, siteL - CartesianIndex(0, 1)]

    # settings
    trunc = only(_get_cluster_trunc(alg.trunc, sites, size(state)))

    state[siteL], state[siteR], messages[EAST, siteR], ϵ =
        _apply_gate_1x2((A_L, A_R), gate, (MN_L, MN_R, ME, MS_L, MS_R, MW); trunc)

    return state, messages, ϵ
end

function _apply_gate_1x2((A_L, A_R)::NTuple{2, PEPSTensor}, gate, (MN_L, MN_R, ME, MS_L, MS_R, MW); trunc)
    # compute square roots
    (sqrtMN_L, isqrtMN_L), (sqrtMN_R, isqrtMN_R), (sqrtME, isqrtME),
        (sqrtMS_L, isqrtMS_L), (sqrtMS_R, isqrtMS_R), (sqrtMW, isqrtMW) =
        sqrt_invsqrt.((MN_L, MN_R, ME, MS_L, MS_R, MW))

    # absorb message tensors
    @tensor T_L[N W S; E P] := A_L[P; N' E S' W'] * sqrtMN_L[N'; N] *
        sqrtMS_L[S; S'] * sqrtMW[W; W']
    @tensor T_R[W P; N E S] := A_R[P; N' E' S' W] * sqrtMN_R[N'; N] *
        sqrtMS_R[S; S'] * sqrtME[E'; E]

    # separate off the indices that are acted on for efficiency
    Q_L, RQ_L = left_orth!(T_L; positive = true)
    LQ_R, Q_R = right_orth!(T_R; positive = true)

    # apply gate
    @tensor gated[-1 -2; -3 -4] := RQ_L[-1; 1 2] * gate[-2 -4; 2 3] * LQ_R[1 3; -3]
    U, S, Vᴴ, ϵ = svd_trunc!(gated; trunc)

    sqrtS = sqrt(S)
    U′ = rmul!(U, sqrtS)
    Vᴴ′ = lmul!(sqrtS, Vᴴ)

    # extract new PEPS tensors
    @tensor A_L′[P; N E S W] := Q_L[N' W' S'; D] * U′[D P; E] *
        isqrtMN_L[N'; N] * isqrtMS_L[S; S'] * isqrtMW[W; W']
    @tensor A_R′[P; N E S W] := Q_R[D; N' E' S'] * Vᴴ′[W; D P] *
        isqrtMN_R[N'; N] * isqrtMS_R[S; S'] * isqrtME[E'; E]

    return A_L′, A_R′, S, ϵ
end
function _apply_gate_1x2((A_L, A_R)::NTuple{2, PEPOTensor}, gate, (MN_L, MN_R, ME, MS_L, MS_R, MW); trunc, purified::Bool = false)
    # compute square roots
    (sqrtMN_L, isqrtMN_L), (sqrtMN_R, isqrtMN_R), (sqrtME, isqrtME),
        (sqrtMS_L, isqrtMS_L), (sqrtMS_R, isqrtMS_R), (sqrtMW, isqrtMW) =
        sqrt_invsqrt.((MN_L, MN_R, ME, MS_L, MS_R, MW))

    # Stage 1: act on the ket leg (P1).
    # Absorb outer messages and place P_bra (P2) on the Q-side, P_ket (P1) on the gate-side.
    # separate off the ket indices for efficiency
    @tensor T_L[N W S P2; E P1] := A_L[P1 P2; N' E S' W'] * sqrtMN_L[N'; N] *
        sqrtMS_L[S; S'] * sqrtMW[W; W']
    Q_L, RQ_L = left_orth!(T_L; positive = true)
    @tensor T_R[W P1; N E S P2] := A_R[P1 P2; N' E' S' W] * sqrtMN_R[N'; N] *
        sqrtMS_R[S; S'] * sqrtME[E'; E]
    LQ_R, Q_R = right_orth!(T_R; positive = true)

    # apply gate
    @tensor gated[-1 -2; -3 -4] := RQ_L[-1; 1 2] * gate[-2 -4; 2 3] * LQ_R[1 3; -3]
    U, S, Vᴴ, ϵ = svd_trunc!(gated; trunc)

    sqrtS = sqrt(S)
    U′ = rmul!(U, sqrtS)
    Vᴴ′ = lmul!(sqrtS, Vᴴ)

    if alg.purified
        # extract new PEPO tensors
        @tensor A_L′[P1 P2; N E S W] := Q_L[N' W' S' P2; D] * U′[D P1; E] *
            isqrtMN_L[N'; N] * isqrtMS_L[S; S'] * isqrtMW[W; W']
        @tensor A_R′[P1 P2; N E S W] := Q_R[D; N' E' S' P2] * Vᴴ′[W; D P1] *
            isqrtMN_R[N'; N] * isqrtMS_R[S; S'] * isqrtME[E'; E]
        return A_L′, A_R′, S, ϵ
    end

    # Stage 2: act on the bra leg (P2)
    # No need to reabsorb messages
    # separate off the bra indices for efficiency
    @tensor T_L[N W S P1; E P2] := Q_L[N W S P2; D] * U′[D P1; E]
    Q_L, RQ_L = left_orth!(T_L; positive = true)
    @tensor T_R[W P2; N E S P1] := Vᴴ′[W; D P1] * Q_R[D; N E S P2]
    LQ_R, Q_R = right_orth!(T_R; positive = true)

    # apply gate
    @tensor gated[-1 -2; -3 -4] := RQ_L[-1; 1 2] * gate'[2 3; -2 -4] * LQ_R[1 3; -3]
    U, S, Vᴴ, ϵ₂ = svd_trunc!(gated; trunc)
    ϵ = max(ϵ, ϵ₂)

    sqrtS = sqrt(S)
    U′ = rmul!(U, sqrtS)
    Vᴴ′ = lmul!(sqrtS, Vᴴ)

    # extract new PEPS tensors
    @tensor A_L′[P1 P2; N E S W] := Q_L[N' W' S' P1; D] * U′[D P2; E] *
        isqrtMN_L[N'; N] * isqrtMS_L[S; S'] * isqrtMW[W; W']
    @tensor A_R′[P1 P2; N E S W] := Q_R[D; N' E' S' P1] * Vᴴ′[W; D P2] *
        isqrtMN_R[N'; N] * isqrtMS_R[S; S'] * isqrtME[E'; E]

    return A_L′, A_R′, S, ϵ
end

function apply_gate_2x1!(
        state, sites::Vector{CartesianIndex{2}}, gate::Gate{2}, alg::SimpleUpdate, messages::BPEnv
    )
    length(sites) == 2 || throw(ArgumentError("invalid sites: $sites"))

    # Note that we must truncate along the canonical arrow direction to not alter the result,
    # so we use south - north here and map north - south to the canonical direction.
    diff = sites[2] - sites[1]
    if diff == CartesianIndex(1, 0)
        sites = reverse(sites)
        gate = permute(gate, ((2, 1), (4, 3)))
    elseif diff != CartesianIndex(-1, 0)
        throw(ArgumentError("invalid sites: $sites"))
    end

    # extract tensors and square roots of the messages
    siteB, siteT = sites
    A_B = state[mod1.(Tuple(siteB), size(state))...]
    A_T = state[mod1.(Tuple(siteT), size(state))...]
    sqrtMN, isqrtMN = sqrt_invsqrt(messages[NORTH, siteT - CartesianIndex(1, 0)])
    sqrtME_T, isqrtME_T = sqrt_invsqrt(messages[EAST, siteT + CartesianIndex(0, 1)])
    sqrtME_B, isqrtME_B = sqrt_invsqrt(messages[EAST, siteB + CartesianIndex(0, 1)])
    sqrtMS, isqrtMS = sqrt_invsqrt(messages[SOUTH, siteB + CartesianIndex(1, 0)])
    sqrtMW_T, isqrtMW_T = sqrt_invsqrt(messages[WEST, siteT - CartesianIndex(0, 1)])
    sqrtMW_B, isqrtMW_B = sqrt_invsqrt(messages[WEST, siteB - CartesianIndex(0, 1)])

    # settings
    trunc = only(_get_cluster_trunc(alg.trunc, sites, size(state)))

    # absorb message tensors (leave the inner S ← N bond free)
    @tensor T_B[E S W; N P] := A_B[P; N E' S' W'] * sqrtME_B[E'; E] *
        sqrtMS[S; S'] * sqrtMW_B[W; W']
    @tensor T_T[S P; N E W] := A_T[P; N' E' S W'] * sqrtMN[N'; N] *
        sqrtME_T[E'; E] * sqrtMW_T[W; W']

    # separate off the indices that are acted on for efficiency
    Q_B, RQ_B = left_orth!(T_B; positive = true)
    LQ_T, Q_T = right_orth!(T_T; positive = true)

    # apply gate
    @tensor gated[-1 -2; -3 -4] := RQ_B[-1; 1 2] * gate[-2 -4; 2 3] * LQ_T[1 3; -3]
    U, S, Vᴴ, ϵ = svd_trunc!(gated; trunc)

    sqrtS = sqrt(S)
    U′ = rmul!(U, sqrtS)
    Vᴴ′ = lmul!(sqrtS, Vᴴ)

    # extract new PEPS tensors
    @tensor A_B′[P; N E S W] := Q_B[E' S' W'; D] * U′[D P; N] *
        isqrtME_B[E'; E] * isqrtMS[S; S'] * isqrtMW_B[W; W']
    @tensor A_T′[P; N E S W] := Q_T[D; N' E' W'] * Vᴴ′[S; D P] *
        isqrtMN[N'; N] * isqrtME_T[E'; E] * isqrtMW_T[W; W']

    # insert tensors
    state[mod1.(Tuple(siteB), size(state))...] = A_B′
    state[mod1.(Tuple(siteT), size(state))...] = A_T′
    messages[NORTH, siteT] = messages[SOUTH, siteB] = S

    return state, messages, ϵ
end
