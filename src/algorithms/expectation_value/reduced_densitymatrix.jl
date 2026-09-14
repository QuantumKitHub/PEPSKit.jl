# Reduced density matrices
# ------------------------

@doc """
    reduced_densitymatrix(inds, ket::InfinitePEPS, bra::InfinitePEPS = ket, env)
    reduced_densitymatrix(inds, ket::InfinitePEPO, bra::InfinitePEPO, env)
    reduced_densitymatrix(inds, state::InfinitePEPO, env)

Construct the reduced density matrix `ρ` of `|ket⟩⟨bra|`, where both `ket` and `bra`
correspond to either a PEPS or a PEPO representing a PEPS with ancillary legs.
Alternatively, construct the reduced density matrix `ρ` of a mixed state specified by the 
density matrix PEPO `state`. The reduced density matrix is contracted over the virtual
indices surrounding the open physical indices at sites `inds` using the environment `env`,
and is normalized such that `str(ρ) = 1`.

See also [`str`](@ref).
""" reduced_densitymatrix

# PEPS case has fast-path specializations
function reduced_densitymatrix(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env
    )
    length(inds) == 1 && return reduced_densitymatrix1x1(only(inds), ket, bra, env)

    if length(inds) == 2
        if inds[2] - inds[1] == CartesianIndex(1, 0)
            return reduced_densitymatrix2x1(inds[1], ket, bra, env)
        elseif inds[2] - inds[1] == CartesianIndex(0, 1)
            return reduced_densitymatrix1x2(inds[1], ket, bra, env)
        end
    end

    static_inds = Tuple(Val.(inds))
    return _contract_densitymatrix(static_inds, (ket, bra), env)
end
function reduced_densitymatrix(
        inds::Vector{CartesianIndex{2}}, state::InfinitePEPO, env
    )
    size(state, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_densitymatrix(static_inds, state, env)
end
function reduced_densitymatrix(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPO, bra::InfinitePEPO, env
    )
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_densitymatrix(static_inds, (ket, bra), env)
end
reduced_densitymatrix(inds, ket::InfinitePEPS, env) =
    reduced_densitymatrix(inds, ket, ket, env)

# handle deprecations of Tuple inds specifications
Base.@deprecate(
    reduced_densitymatrix(
        inds::NTuple{N, CartesianIndex{2}}, args...
    ) where {N},
    reduced_densitymatrix(collect(inds), args...)
)
Base.@deprecate(
    reduced_densitymatrix(
        inds::NTuple{N, Tuple{Int, Int}}, args...
    ) where {N},
    reduced_densitymatrix(collect(CartesianIndex.(inds)), args...)
)


# Fixed-size fast paths
# ---------------------
#
# These carry the docstrings and reject unsupported environments; the implementations live in
# `algorithms/contractions/local_patch/densitymatrix/`.

"""
    reduced_densitymatrix1x1(ind, ket, bra, env)

Construct the reduced density matrix of `|ket⟩⟨bra|` on the single site `ind`, using an
optimized contraction for the environment `env`.
"""
reduced_densitymatrix1x1(ind, ket, bra, env) = throw(
    ArgumentError(
        "No 1x1 reduced density matrix contraction defined for environments of type $(typeof(env))."
    )
)

"""
    reduced_densitymatrix2x1(ind, ket, bra, env)

Construct the reduced density matrix of `|ket⟩⟨bra|` on the vertical pair of sites starting
at `ind`, using an optimized contraction for the environment `env`.
"""
reduced_densitymatrix2x1(ind, ket, bra, env) = throw(
    ArgumentError(
        "No 2x1 reduced density matrix contraction defined for environments of type $(typeof(env))."
    )
)

"""
    reduced_densitymatrix1x2(ind, ket, bra, env)

Construct the reduced density matrix of `|ket⟩⟨bra|` on the horizontal pair of sites starting
at `ind`, using an optimized contraction for the environment `env`.
"""
reduced_densitymatrix1x2(ind, ket, bra, env) = throw(
    ArgumentError(
        "No 1x2 reduced density matrix contraction defined for environments of type $(typeof(env))."
    )
)
