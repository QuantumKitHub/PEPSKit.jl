"""
$(TYPEDEF)

Abstract super type for time evolution algorithms of InfinitePEPS or InfinitePEPO.
"""
abstract type TimeEvolution end

"""
    mutable struct TimeEvolver{TE <: TimeEvolution, N <: Number, G, S}

Iterator for Trotter-based time evolution of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
mutable struct TimeEvolver{TE <: TimeEvolution, N <: Number, G, S}
    # Time evolution algorithm
    alg::TE
    # Trotter time step
    dt::N
    # Maximal iteration steps
    nstep::Int
    # Trotter gates
    gate::G
    # PEPS/PEPO (and environment)
    state::S
end

Base.iterate(it::TimeEvolver) = iterate(it, it.state)

function _timeevol_sanity_check(
        ψ₀::InfiniteState, Pspaces::M, alg::A
    ) where {A <: TimeEvolution, M <: AbstractMatrix{<:ElementarySpace}}
    Nr, Nc, = size(ψ₀)
    @assert (Nr >= 2 && Nc >= 2) "Unit cell size for simple update should be no smaller than (2, 2)."
    @assert Pspaces == physicalspace(ψ₀) "Physical spaces of `ψ₀` do not match `Pspaces`."
    if hasfield(typeof(alg), :gate_bothsides) && alg.gate_bothsides
        @assert ψ₀ isa InfinitePEPO "alg.gate_bothsides = true is only compatible with PEPO."
    end
    if hasfield(typeof(alg), :bipartite) && alg.bipartite
        @assert Nr == Nc == 2 "`bipartite = true` requires 2 x 2 unit cell size."
        @assert ψ₀ isa InfinitePEPS "Evolution of PEPO with bipartite structure is not implemented."
    end
    return nothing
end
