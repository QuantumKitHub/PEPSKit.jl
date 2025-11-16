"""
$(TYPEDEF)

Abstract super type for time evolution algorithms of InfinitePEPS or InfinitePEPO.
"""
abstract type TimeEvolution end

"""
    mutable struct TimeEvolver{TE <: TimeEvolution, G, S, N <: Number}

Iterator for Trotter-based time evolution of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
mutable struct TimeEvolver{TE <: TimeEvolution, G, S, N <: Number}
    "Time evolution algorithm (currently supported: `SimpleUpdate`)"
    alg::TE
    "Trotter time step"
    dt::N
    "The number of iteration steps"
    nstep::Int
    "Trotter gates"
    gate::G
    "Internal state of the iterator, including the number of
    already performed iterations, evolved time, PEPS/PEPO and its environment"
    state::S
end

Base.iterate(it::TimeEvolver) = iterate(it, it.state)

function _timeevol_sanity_check(
        ψ₀::InfiniteState, Pspaces::M, alg::A
    ) where {A <: TimeEvolution, M <: AbstractMatrix{<:ElementarySpace}}
    Nr, Nc, = size(ψ₀)
    @assert (Nr >= 2 && Nc >= 2) "Unit cell size for simple update should be no smaller than (2, 2)."
    @assert Pspaces == physicalspace(ψ₀) "Physical spaces of `ψ₀` do not match `Pspaces`."
    if hasfield(typeof(alg), :purified) && !alg.purified
        @assert ψ₀ isa InfinitePEPO "alg.purified = false is only applicable to PEPO."
    end
    if hasfield(typeof(alg), :bipartite) && alg.bipartite
        @assert Nr == Nc == 2 "`bipartite = true` requires 2 x 2 unit cell size."
    end
    return nothing
end
