#=
Utilities for preserving the norm of (VectorInterface-compliant) vectors during optimization.
=#

"""
$(SIGNATURES)

Performs a norm-preserving retraction of vector `A` along the direction `η` with step size `α`, giving a new vector ``A′``,
```math
A′ ← \\cos ( α ‖η‖ / ‖A‖ ) A + \\sin ( α ‖η‖ / ‖A‖ ) ‖A‖ η / ‖η‖,
```
and corresponding directional derivative `ξ`,
```math
ξ = \\cos ( α ‖η‖ / ‖A‖ ) η - \\sin ( α ‖η‖ / ‖A‖ ) ‖η‖ A / ‖A‖,
```
such that ``\\operatorname{Re} ⟨ A′, ξ ⟩ = 0`` and ``‖A′‖ = ‖A‖``.
The inputs must satisfy ``\\operatorname{Re} ⟨ A, η ⟩ = 0``.
For `η = 0`, return copies of `A` and `η`.

!!! note
    The vectors `A` and `η` should satisfy the interface specified by [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl).

"""
function norm_preserving_retract(A, η, α)
    n_A = norm(A)
    n_η = norm(η)
    iszero(n_η) && return scale(A, true), zerovector(η)
    sn, cs = sincos(α * n_η / n_A)

    A′ = add(A, η, sn * n_A / n_η, cs)
    ξ = add(A, η, cs, -sn * n_η / n_A)

    return A′, ξ
end

"""
$(SIGNATURES)

Transports a direction `ξ` at `A` to a valid direction at ``A′`` corresponding to the norm-preserving retraction of `A` along `η` with step size `α`.
In particular, starting from a direction `ξ` of the form
```math
ξ = \\operatorname{Re} ⟨ η / ‖η‖, ξ ⟩ η / ‖η‖ + Δξ
```
where ``\\operatorname{Re} ⟨ Δξ, A ⟩ = \\operatorname{Re} ⟨ Δξ, η ⟩ = 0``, it returns
```math
ξ(α) = \\operatorname{Re} ⟨ η / ‖η‖, ξ ⟩ ( \\cos ( α ‖η‖ / ‖A‖ ) η / ‖η‖ - \\sin ( α ‖η‖ / ‖A‖ ) A / ‖A‖ ) + Δξ
```
such that ``‖ξ(α)‖ = ‖ξ‖, \\operatorname{Re} ⟨ A′, ξ(α) ⟩ = 0``.
Both `η` and `ξ` must be real-orthogonal to `A`.
For `η = 0`, return `ξ` unchanged.

!!! note
    The vectors `A` and `η` should satisfy the interface specified by [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl).

"""
function norm_preserving_transport!(ξ, A, η, α, A′)
    n_A = norm(A)
    n_η = norm(η)
    iszero(n_η) && return ξ
    sn, cs = sincos(α * n_η / n_A)

    overlaps_η_ξ = real(inner(η, ξ)) / n_η
    add!(ξ, η, (cs - 1) * overlaps_η_ξ / n_η)
    add!(ξ, A, -sn * overlaps_η_ξ / n_A)

    return ξ
end
