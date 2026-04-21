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

"""
Process the Trotter time step `dt` according to the intended usage.
"""
function _get_dt(
        state::InfiniteState, dt::Number, imaginary_time::Bool
    )
    # PEPS update: exp(-H dt)|ψ⟩
    # PEPO update (purified): exp(-H dt/2)|ρ⟩
    # PEPO update (not purified): exp(-H dt/2) ρ exp(-H dt/2)
    dt′ = (state isa InfinitePEPS) ? dt : (dt / 2)
    if (state isa InfinitePEPO)
        @assert size(state)[3] == 1
    end
    if !imaginary_time
        @assert (state isa InfinitePEPS) "Real time evolution of InfinitePEPO (Heisenberg picture) is not implemented."
        dt′ = complex(zero(dt′), dt′)
    end
    return dt′
end

function _timeevol_sanity_check(
        ψ₀::InfiniteState, Pspaces::M, alg::A
    ) where {A <: TimeEvolution, M <: AbstractMatrix{<:ElementarySpace}}
    Nr, Nc, = size(ψ₀)
    @assert (Nr >= 2 && Nc >= 2) "Unit cell size for time evolution should be no smaller than (2, 2)."
    if ψ₀ isa InfinitePEPO
        @assert size(ψ₀, 3) == 1 "PEPO to be time evolved should have only one layer."
    end
    @assert Pspaces == physicalspace(ψ₀) "Physical spaces of `ψ₀` do not match `Pspaces`."
    if hasfield(typeof(alg), :purified) && !alg.purified
        @assert ψ₀ isa InfinitePEPO "alg.purified = false is only applicable to PEPO."
    end
    if hasfield(typeof(alg), :bipartite) && alg.bipartite
        @assert _is_bipartite(ψ₀) "Input state is not bipartite with 2 x 2 unit cell."
    end
    return nothing
end

function MPSKit.infinite_temperature_density_matrix(H::LocalOperator)
    T = scalartype(H)
    A = map(physicalspace(H)) do Vp
        ψ = permute(TensorKit.id(T, Vp), (1, 2))
        Vv = oneunit(Vp) # trivial (1D) virtual space
        virt = ones(T, domain(ψ) ← Vv ⊗ Vv ⊗ Vv' ⊗ Vv')
        return ψ * virt
    end
    return InfinitePEPO(cat(A; dims = 3))
end

"""
Get the `SiteDependentTruncation` used by time evolution
that preserves virtual spaces of `state`.
"""
function _get_fixedspacetrunc(state::InfiniteState)
    if state isa InfinitePEPO
        size(state, 3) != 1 && error("Input InfinitePEPO is expect to have only one layer.")
    end
    Nr, Nc = size(state)
    return SiteDependentTruncation(
        map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
            V = domain(state[r, c], (d == 1) ? EAST : NORTH)
            isdual(V) && (V = flip(V))
            return truncspace(V)
        end
    )
end
