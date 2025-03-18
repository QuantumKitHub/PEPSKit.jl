#=
Utilities for preserving the norm of (VectorInterface-compliant) vectors during optimization.
=#

"""
    vector_retract(A, η, α)

Performs a norm-preserving retraction of vector `A` along the direction `η` with step size
`α`, giving a new vector `A´`,
```math
A' \\leftarrow \\cos \\left( α \\frac{||η||}{||A||} \\right) A + \\sin \\left( α \\frac{||η||}{||A||} \\right) ||A|| \\frac{η}{||η||},
```
and corresponding directional derivative `ξ`,
```math
ξ = \\cos \\left( α \\frac{||η||}{||A||} \\right) η - \\sin \\left( α \\frac{||η||}{||A||} \\right) ||η|| \\frac{A}{||A||},
```
such that ``\\langle A', ξ \\rangle = 0`` and ``||A'|| = ||A||``.
"""
function vector_retract(A, η, α)
    n_A = norm(A)
    n_η = norm(η)
    sn, cs = sincos(α * n_η / n_A)

    A´ = add(A, η, sn * n_A / n_η, cs)
    ξ = add(A, η, cs, -sn * n_η / n_A)

    return A´, ξ
end

"""
    vector_transport!(ξ, A, η, α, A′)

Transports a direction `ξ` at `A` to a valid direction at `A´` corresponding to
the norm-preserving retraction of `A` along `η` with step size `α`. In particular, starting
from a direction `η` of the form
```math
ξ = ⟨ η / ‖η‖, ξ ⟩ η / ‖η‖ + Δξ
```
where ``\\langle Δξ, A \\rangle = \\langle Δξ, η \\rangle = 0``, it returns
```math
ξ(α) = \\left\\langle \\frac{η}{||η||}, ξ \\right \\rangle \\left( \\cos \\left( α \\frac{||η||}{||A||} \\right) \\frac{η}{||η||} - \\sin( \\left( α \\frac{||η||}{||A||} \\right) \\frac{A}{||A||} \\right) + Δξ
```
such that ``||ξ(α)|| = ||ξ||, \\langle A', ξ(α) \\rangle = 0``.
"""
function vector_transport!(ξ, A, η, α, A´)
    n_A = norm(A)
    n_η = norm(η)
    sn, cs = sincos(α * n_η / n_A)

    overlaps_η_ξ = inner(η, ξ) / n_η
    add!(ξ, η, (cs - 1) * overlaps_η_ξ / n_η)
    add!(ξ, A, -sn * overlaps_η_ξ / n_A)

    return ξ
end
