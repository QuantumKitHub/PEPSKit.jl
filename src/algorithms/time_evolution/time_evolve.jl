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

function _state_bipartite_check(psi::InfiniteState)
    if isa(psi, InfinitePEPO)
        @assert size(psi, 3) == 1 "Input InfinitePEPO is expected to have only one layer."
    end
    if !(size(psi, 1) == size(psi, 2) == 2)
        return false
    end
    for (r, c) in Iterators.product(1:2, 1:2)
        r′, c′ = _next(r, 2), _next(c, 2)
        if psi[r, c] != psi[r′, c′]
            return false
        end
    end
    return true
end

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
        dt′ = 1.0im * dt′
    end
    return dt′
end

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
        @assert _state_bipartite_check(ψ₀) "Input state is not bipartite with 2 x 2 unit cell."
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
