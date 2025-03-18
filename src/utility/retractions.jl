#=
Utilities for preserving the norm of (VectorInterface-compliant) vectors during optimization.
=#

"""
    norm_preserving_retract(A, η, α)

Performs a norm-preserving retraction of vector `A` along the direction `η` with step size
`α`, giving a new vector `A´`,
```math
A' ← \\cos ( α ‖η‖ / ‖A‖ ) A + \\sin ( α ‖η‖ / ‖A‖ ) ‖A‖ η / ‖η‖,
```
and corresponding directional derivative `ξ`,
```math
ξ = \\cos ( α ‖η‖ / ‖A‖ ) η - \\sin ( α ‖η‖ / ‖A‖ ) ‖η‖ A / ‖A‖,
```
such that ``⟨ A', ξ ⟩ = 0`` and ``‖A'‖ = ‖A‖``.

!!! note
    The vectors `A` and `η` should satisfy the interface specified by
    [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)

"""
function norm_preserving_retract(A, η, α)
    n_A = norm(A)
    n_η = norm(η)
    sn, cs = sincos(α * n_η / n_A)

    A´ = add(A, η, sn * n_A / n_η, cs)
    ξ = add(A, η, cs, -sn * n_η / n_A)

    return A´, ξ
end

"""
    norm_preserving_transport!(ξ, A, η, α, A′)

Transports a direction `ξ` at `A` to a valid direction at `A´` corresponding to
the norm-preserving retraction of `A` along `η` with step size `α`. In particular, starting
from a direction `η` of the form
```math
ξ = ⟨ η / ‖η‖, ξ ⟩ η / ‖η‖ + Δξ
```
where ``⟨ Δξ, A ⟩ = ⟨ Δξ, η ⟩ = 0``, it returns
```math
ξ(α) = ⟨ η / ‖η‖, ξ ⟩ ( \\cos ( α ‖η‖ / ‖A‖ ) η / ‖η‖ - \\sin ( α ‖η‖ / ‖A‖ ) A / ‖A‖ ) + Δξ
```
such that ``‖ξ(α)‖ = ‖ξ‖, ⟨ A', ξ(α) ⟩ = 0``.

!!! note
    The vectors `A` and `η` should satisfy the interface specified by
    [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl)

"""
function norm_preserving_transport!(ξ, A, η, α, A´)
    n_A = norm(A)
    n_η = norm(η)
    sn, cs = sincos(α * n_η / n_A)

    overlaps_η_ξ = inner(η, ξ) / n_η
    add!(ξ, η, (cs - 1) * overlaps_η_ξ / n_η)
    add!(ξ, A, -sn * overlaps_η_ξ / n_A)

    return ξ
end
