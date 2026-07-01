# Correlator adapters for InfinitePEPS / InfinitePEPO

struct _PEPSCorrelator{B <: InfinitePEPS, K <: InfinitePEPS, E <: CTMRGEnv}
    bra::B
    ket::K
    env::E

    function _PEPSCorrelator(bra::B, ket::K, env::E) where {B, K, E}
        size(ket) == size(bra) ||
            throw(DimensionMismatch("The ket and bra must have the same unit cell."))
        return new{B, K, E}(bra, ket, env)
    end
end

struct _PEPOPurifiedCorrelator{B <: InfinitePEPO, K <: InfinitePEPO, E <: CTMRGEnv}
    bra::B
    ket::K
    env::E

    function _PEPOPurifiedCorrelator(bra::B, ket::K, env::E) where {B, K, E}
        size(ket) == size(bra) ||
            throw(DimensionMismatch("The ket and bra must have the same unit cell."))
        size(ket, 3) == size(bra, 3) == 1 ||
            throw(ArgumentError("Purified PEPO correlators require one-layer PEPOs."))
        return new{B, K, E}(bra, ket, env)
    end
end

const _BraketCorrelator = Union{_PEPSCorrelator, _PEPOPurifiedCorrelator}

_braket_correlator(bra::InfinitePEPS, ket::InfinitePEPS, env::CTMRGEnv) =
    _PEPSCorrelator(bra, ket, env)
_braket_correlator(bra::InfinitePEPO, ket::InfinitePEPO, env::CTMRGEnv) =
    _PEPOPurifiedCorrelator(bra, ket, env)

struct _PEPOTraceCorrelator{P <: InfinitePEPO, E <: CTMRGEnv}
    ρ::P
    env::E

    function _PEPOTraceCorrelator(ρ::P, env::E) where {P, E}
        (size(ρ, 3) == 1) ||
            throw(ArgumentError("The input PEPO ρ must have only one layer."))
        return new{P, E}(ρ, env)
    end
end

function _correlator_scalartype(context::_BraketCorrelator, O::FiniteMPO)
    return TensorOperations.promote_contract(
        scalartype(context.bra), scalartype(context.ket),
        scalartype(context.env), scalartype.(O)...
    )
end

function _correlator_scalartype(context::_PEPOTraceCorrelator, O::FiniteMPO)
    return TensorOperations.promote_contract(
        scalartype(context.ρ), scalartype(context.env), scalartype.(O)...
    )
end

_correlator_unitcell(context::_BraketCorrelator) = size(context.bra)[1:2]
_correlator_unitcell(context::_PEPOTraceCorrelator) = size(context.ρ)[1:2]

function Base.rotl90(context::_PEPSCorrelator)
    rotated_bra = rotl90(context.bra)
    rotated_ket = context.bra === context.ket ? rotated_bra : rotl90(context.ket)
    return _PEPSCorrelator(rotated_bra, rotated_ket, rotl90(context.env))
end

function Base.rotl90(context::_PEPOPurifiedCorrelator)
    rotated_bra = rotl90(context.bra)
    rotated_ket = context.bra === context.ket ? rotated_bra : rotl90(context.ket)
    return _PEPOPurifiedCorrelator(rotated_bra, rotated_ket, rotl90(context.env))
end

Base.rotl90(context::_PEPOTraceCorrelator) =
    _PEPOTraceCorrelator(rotl90(context.ρ), rotl90(context.env))

function _start_correlator(
        i::CartesianIndex{2}, context::_BraketCorrelator, O::MPOTensor
    )
    return start_correlator(i, context.bra, O, context.ket, context.env)
end

function _start_correlator(
        i::CartesianIndex{2}, context::_PEPOTraceCorrelator, O::PFTensor
    )
    return start_correlator(i, context.ρ, O, context.env)
end

function _end_correlator_numerator(
        j::CartesianIndex{2},
        V::AbstractTensorMap{T, S, 4, 1},
        context::_BraketCorrelator,
        O::MPOTensor,
    ) where {T, S}
    return end_correlator_numerator(j, V, context.bra, O, context.ket, context.env)
end

function _end_correlator_numerator(
        j::CartesianIndex{2},
        V::CTMRGEdgeTensor{T, S, 3},
        context::_PEPOTraceCorrelator,
        O::PFTensor,
    ) where {T, S}
    return end_correlator_numerator(j, V, context.ρ, O, context.env)
end

function _edge_transfermatrix(row::Int, col::Int, context::_BraketCorrelator)
    return _edge_transfermatrix(row, col, context.bra, context.ket, context.env)
end

function _edge_transfermatrix(row::Int, col::Int, context::_PEPOTraceCorrelator)
    return _edge_transfermatrix(row, col, context.ρ, context.env)
end

function _correlator_horizontal(
        context, operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
    )
    _issorted_correlator_sites(i, js)
    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))
    # preallocate with correct scalartype
    G = similar(js, _correlator_scalartype(context, O))
    # left start for operator and norm contractions
    c = i # current column being handled
    Vn, Vo = _start_correlator(c, context, O[1])
    j_last = last(js)
    for (k, j) in enumerate(js)
        local numerator
        while j > c
            c += CartesianIndex(0, 1)
            if c == j
                numerator = _end_correlator_numerator(j, Vo, context, O[2])
            end
            T = _edge_transfermatrix(c[1], c[2], context)
            c != j_last && (Vo = Vo * T)
            Vn = Vn * T
        end
        # compute overlap without operator
        denominator = end_correlator_denominator(j, Vn, context.env)
        G[k] = numerator / denominator
    end
    return G
end

function _correlator_vertical(context, operator, i::CartesianIndex{2}, js)
    unitcell = _correlator_unitcell(context)
    rotated_i = siterotl90(i, unitcell)
    rotated_js = map(j -> siterotl90(j, unitcell), js)
    return _correlator_horizontal(rotl90(context), operator, rotated_i, rotated_js)
end

function _correlator(
        context, O, i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}}
    )
    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return _correlator_horizontal(context, O, i, js)
    elseif all(==(i[2]) ∘ last ∘ Tuple, js)
        return _correlator_vertical(context, O, i, js)
    else
        error("Only horizontal or vertical correlators are implemented")
    end
end

function _issorted_correlator_sites(i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}})
    return issorted(vcat(i, js); by = last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))
end
