# Direct local patch contractions
# -------------------------------

"""
    contract_local_operator(inds, O, ket::InfinitePEPS, bra::InfinitePEPS, env)
    contract_local_operator(inds, O, ket::InfinitePEPO, bra::InfinitePEPO, env)
    contract_local_operator(inds, O, state::InfinitePEPO, env)

Contract a local operator `O` between `ket` and `bra` states, computing `⟨bra|O|ket⟩`, where
`ket` and `bra` correspond to either a PEPS or a PEPO representing a PEPS with ancillary
legs. Alternatively, contract a local operator `O` with a density matrix PEPO `state`,
computing `tr(O * state)`. `O` is applied to the open physical indices at sites `inds`, and
the result is contracted over the surrounding virtual indices using the environment `env`.
"""
function contract_local_operator(
        inds::Vector{CartesianIndex{2}}, O,
        ket::InfinitePEPS, bra::InfinitePEPS, env,
    )
    static_inds = Tuple(Val.(inds))
    return _contract_local_operator(static_inds, O, (ket, bra), env)
end
function contract_local_operator(
        inds::Vector{CartesianIndex{2}}, O, state::InfinitePEPO, env
    )
    size(state, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_operator(static_inds, O, state, env)
end
function contract_local_operator(
        inds::Vector{CartesianIndex{2}}, O, ket::InfinitePEPO, bra::InfinitePEPO, env
    )
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_operator(static_inds, O, (ket, bra), env)
end
function contract_local_operator(inds::Vector{Tuple{Int, Int}}, O, args...)
    return contract_local_operator(CartesianIndex.(inds), O, args...)
end

Base.@deprecate(
    contract_local_operator(
        inds::NTuple, args...
    ),
    contract_local_operator(collect(inds), args...)
)

"""
    contract_local_norm(inds, ket::InfinitePEPS, bra::InfinitePEPS, env)
    contract_local_norm(inds, ket::InfinitePEPO, bra::InfinitePEPO, env)
    contract_local_norm(inds, state::InfinitePEPO, env)

Contract a local norm corresponding to the overlap `ket` and `bra` states, computing a patch
of `⟨bra|ket⟩`, where `ket` and `bra` correspond to either a PEPS or a PEPO representing a
PEPS with ancillary legs.
Alternatively, contract a local norm patch of a density matrix PEPO `state`, computing a patch of `tr(state)`.

The contracted rectangular norm patch is determined by the open physical indices `inds` and
is contracted over the surrounding virtual indices using the environment `env`. In
particular, the patch location is precisely the same as that of the patch used in
[`contract_local_operator`](@ref).
"""
function contract_local_norm(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env
    )
    static_inds = Tuple(Val.(inds))
    return _contract_local_norm(static_inds, (ket, bra), env)
end
function contract_local_norm(inds::Vector{CartesianIndex{2}}, state::InfinitePEPO, env)
    size(state, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_norm(static_inds, state, env)
end
function contract_local_norm(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPO, bra::InfinitePEPO, env
    )
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_norm(static_inds, (ket, bra), env)
end
function contract_local_norm(inds::Vector{Tuple{Int, Int}}, args...)
    return contract_local_norm(CartesianIndex.(inds), args...)
end

Base.@deprecate(
    contract_local_norm(inds::NTuple, ket::InfinitePEPS, bra::InfinitePEPS, env),
    contract_local_norm(collect(inds), ket, bra, env)
)
