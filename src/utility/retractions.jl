#=
Utilities for preserving the norm of (VectorInterface-compliant) vectors during optimization.
=#

"""Return the real Euclidean inner product of vectors, optionally accepting an unused base point."""
real_inner(η₁, η₂) = real(inner(η₁, η₂))
real_inner(_, η₁, η₂) = real(dot(η₁, η₂))

"""
$(SIGNATURES)

Project `η` onto the tangent space of the sphere through the nonzero vector `A` without mutating either input.
Here ``⟨⋅, ⋅⟩`` denotes the real part of the Hermitian inner product:
```math
Π_A(η) = η - \\frac{⟨ A, η ⟩}{⟨ A, A ⟩} A.
```
Complex vectors are treated as a real vector space, retaining the tangent phase direction ``i A``.
"""
function project_tangent(η, A)
    return add(η, A, -real_inner(A, η) / real_inner(A, A))
end

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
such that ``⟨ A′, ξ ⟩ = 0`` and ``‖A′‖ = ‖A‖``.
The inputs must satisfy ``⟨ A, η ⟩ = 0``.
For `η = 0`, return copies of `A` and `η`.

Here ``⟨⋅, ⋅⟩`` denotes the real part of the Hermitian inner product, and ``‖⋅‖`` is the Euclidean norm.
The base point `A` must be nonzero.

!!! note
    The vectors `A` and `η` should satisfy the interface specified by [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl).

"""
function norm_preserving_retract(A, η, α)
    n_A = sqrt(real_inner(A, A))
    n_η = sqrt(real_inner(η, η))
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
ξ = ⟨ η / ‖η‖, ξ ⟩ η / ‖η‖ + Δξ
```
where ``⟨ Δξ, A ⟩ = ⟨ Δξ, η ⟩ = 0``, it returns
```math
ξ(α) = ⟨ η / ‖η‖, ξ ⟩ ( \\cos ( α ‖η‖ / ‖A‖ ) η / ‖η‖ - \\sin ( α ‖η‖ / ‖A‖ ) A / ‖A‖ ) + Δξ
```
such that ``‖ξ(α)‖ = ‖ξ‖, ⟨ A′, ξ(α) ⟩ = 0``.
Both `η` and `ξ` must be orthogonal to `A` in the real Euclidean inner product.
For `η = 0`, return `ξ` unchanged.

Here ``⟨⋅, ⋅⟩`` denotes the real part of the Hermitian inner product, and ``‖⋅‖`` is the Euclidean norm.
The base point `A` must be nonzero.

!!! note
    The vectors `A` and `η` should satisfy the interface specified by [VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl).

"""
function norm_preserving_transport!(ξ, A, η, α, A′)
    n_A = sqrt(real_inner(A, A))
    n_η = sqrt(real_inner(η, η))
    iszero(n_η) && return ξ
    sn, cs = sincos(α * n_η / n_A)

    overlaps_η_ξ = real_inner(η, ξ) / n_η
    add!(ξ, η, (cs - 1) * overlaps_η_ξ / n_η)
    add!(ξ, A, -sn * overlaps_η_ξ / n_A)

    return ξ
end
