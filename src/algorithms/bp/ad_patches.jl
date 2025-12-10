#
# Temporary patches to make BP gauge fixing differentiable
#

# environment updating during optimization
function update!(env::BPEnv{T}, env´::BPEnv{T}) where {T}
    env.messages .= env´.messages
    return env
end


## InfinitePEPS AD patch

function ChainRulesCore.rrule(::Type{InfinitePEPS}, A::Matrix{<:PEPSTensor})
    network = InfinitePEPS(A)
    function InfinitePEPS_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        return NoTangent(), unitcell(Δnetwork)
    end
    return network, InfinitePEPS_pullback
end


## BPEnv AD patches

# initial constructor
@non_differentiable BPEnv(state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)

# adjoint accumulation
function Base.:+(e₁::BPEnv, e₂::BPEnv)
    return BPEnv(e₁.messages + e₂.messages)
end
function Base.:-(e₁::BPEnv, e₂::BPEnv)
    return BPEnv(e₁.messages - e₂.messages)
end
Base.:*(α::Number, e::BPEnv) = BPEnv(α * e.messages)

# constructor from message array
function ChainRulesCore.rrule(::Type{BPEnv}, messages)
    bpenv_pullback(ē) = NoTangent(), ē.messages
    return BPEnv(messages), bpenv_pullback
end

# field access
function ChainRulesCore.rrule(::typeof(getproperty), e::BPEnv, name::Symbol)
    result = getproperty(e, name)
    if name === :messages
        function messages_pullback(Δmessages_)
            Δmessages = unthunk(Δmessages_)
            return NoTangent(), BPEnv(Δmessages), NoTangent()
        end
        return result, messages_pullback
    else
        # this should never happen because already errored in forwards pass
        throw(ArgumentError("No rrule for getproperty of $name"))
    end
end

# indexing
VI.zerovector(e::BPEnv) = BPEnv(zerovector(e.messages))
function ChainRulesCore.rrule(::typeof(Base.getindex), e::BPEnv, ax::Int, r::Int, c::Int)
    tensor = e.messages[ax, r, c]

    function getindex_pullback(Δmessage_)
        Δmessage = unthunk(Δmessage_)
        Δweights = zerovector(e)
        Δweights.messages[ax, r, c] = Δmessage
        return NoTangent(), Δweights, NoTangent(), NoTangent(), NoTangent()
    end
    return tensor, getindex_pullback
end


## SUWeight AD patches

# accumulation
function Base.:+(w₁::SUWeight, w₂::SUWeight)
    return SUWeight(w₁.data + w₂.data)
end
function Base.:-(w₁::SUWeight, w₂::SUWeight)
    return SUWeight(w₁.data - w₂.data)
end
Base.:*(α::Number, w::SUWeight) = SUWeight(α * w.data)
function ChainRulesCore.rrule(::Type{SUWeight}, data)
    suweight_pullback(ē) = NoTangent(), ē.data
    return SUWeight(data), suweight_pullback
end

# indexing
VI.zerovector(w::SUWeight) = SUWeight(zerovector(w.data))
function ChainRulesCore.rrule(::typeof(Base.getindex), w::SUWeight, ax::Int, r::Int, c::Int)
    tensor = w.data[ax, r, c]

    function getindex_pullback(Δmessage_)
        Δmessage = unthunk(Δmessage_)
        Δweights = zerovector(w)
        Δweights[ax, r, c] = Δmessage
        return NoTangent(), Δweights, NoTangent(), NoTangent(), NoTangent()
    end
    return tensor, getindex_pullback
end

## Bond weight absorption patch

# circumvent ncon issues
function _absorb_weights(t::PEPSTensor, weights::SUWeight, row::Int, col::Int)
    Nr, Nc = size(weights)[2:end]
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    wt_N = sdiag_pow(weights[2, row, col], 0.5)
    wt_E = sdiag_pow(weights[1, row, col], 0.5)
    wt_S = sdiag_pow(weights[2, _next(row, Nr), col], 0.5)
    wt_W = sdiag_pow(weights[1, row, _prev(col, Nc)], 0.5)
    return _multiply_weights(t, wt_N, wt_E, wt_S, wt_W)
end
function _multiply_weights(t, wt_N, wt_E, wt_S, wt_W)
    # dual space: t ← wt
    # non-dual space: t → wt
    if isdual(north_virtualspace(t))
        @tensor tn[d; DN DE DS DW] := t[d; DNc DE DS DW] * wt_N[DNc; DN]
    else
        @tensor tn[d; DN DE DS DW] := t[d; DNc DE DS DW] * wt_N[DN; DNc]
    end
    if isdual(east_virtualspace(t))
        @tensor te[d; DN DE DS DW] := tn[d; DN DEc DS DW] * wt_E[DEc; DE]
    else
        @tensor te[d; DN DE DS DW] := tn[d; DN DEc DS DW] * wt_E[DE; DEc]
    end
    if isdual(south_virtualspace(t))
        @tensor ts[d; DN DE DS DW] := te[d; DN DE DSc DW] * wt_S[DSc; DS]
    else
        @tensor ts[d; DN DE DS DW] := te[d; DN DE DSc DW] * wt_S[DS; DSc]
    end
    if isdual(west_virtualspace(t))
        @tensor tw[d; DN DE DS DW] := ts[d; DN DE DS DWc] * wt_W[DWc; DW]
    else
        @tensor tw[d; DN DE DS DW] := ts[d; DN DE DS DWc] * wt_W[DW; DWc]
    end
    return tw
end

## Pullback of TensorKit.transpose patch

_transpose(t::AbstractTensorMap{T, S, 1, 1}) where {T, S} = permute(t, ((2,), (1,)))

## Pullback of one-norm of a TensorMap

_one_norm(t::AbstractTensorMap) = TensorKit._norm(blocks(t), 1, zero(real(scalartype(t))))
function ChainRulesCore.rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(_one_norm), t::AbstractTensorMap
    )
    P_t = ProjectTo(t)
    n = float(zero(scalartype(t)))
    abssum(x) = sum(abs, x)
    abssum_pullbacks = map(blocks(t)) do (c, b)
        temp, abssum_pullback = rrule_via_ad(cfg, abssum, b)
        n += oftype(n, dim(c) * temp)
        return c => abssum_pullback
    end
    function _one_norm_pullback(Δn_)
        Δn = unthunk(Δn_)
        Δt = similar(t)
        for (c, pb) in abssum_pullbacks
            copy!(block(Δt, c), last(pb(Δn * dim(c))))
        end
        return NoTangent(), P_t(Δt)
    end
    return n, _one_norm_pullback
end

## Degeneracy checking

function check_degenerate_spectrum(S::AbstractTensorMap)
    if PEPSKit.is_degenerate_spectrum(S)
        svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
        @warn("Degenerate singular values in root of BP message: ", svals)
    end
    return nothing
end
