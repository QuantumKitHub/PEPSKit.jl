using LinearAlgebra
using TensorKit
using ChainRulesCore, ChainRulesTestUtils, Zygote
using PEPSKit

# Truncated SVD with outdated adjoint
oldsvd(t::AbstractTensorMap, χ::Int; kwargs...) = itersvd(t, χ; kwargs...)

# Outdated adjoint not taking truncated part into account
function ChainRulesCore.rrule(
    ::typeof(oldsvd), t::AbstractTensorMap, χ::Int; εbroad=0, kwargs...
)
    U, S, V = oldsvd(t, χ; kwargs...)

    function oldsvd_pullback((ΔU, ΔS, ΔV))
        ∂t = similar(t)
        for (c, b) in blocks(∂t)
            copyto!(
                b,
                oldsvd_rev(
                    block(U, c),
                    block(S, c),
                    block(V, c),
                    block(ΔU, c),
                    block(ΔS, c),
                    block(ΔV, c);
                    εbroad,
                ),
            )
        end
        return NoTangent(), ∂t, NoTangent()
    end

    return (U, S, V), oldsvd_pullback
end

function oldsvd_rev(
    U::AbstractMatrix,
    S::AbstractMatrix,
    V::AbstractMatrix,
    ΔU,
    ΔS,
    ΔV;
    εbroad=0,
    atol::Real=0,
    rtol::Real=atol > 0 ? 0 : eps(scalartype(S))^(3 / 4),
)
    tol = atol > 0 ? atol : rtol * S[1, 1]
    F = PEPSKit.invert_S²(S, tol; εbroad)  # Includes Lorentzian broadening
    S⁻¹ = pinv(S; atol=tol)

    # dS contribution
    term = ΔS isa ZeroTangent ? ΔS : Diagonal(diag(ΔS))

    # dU₁ and dV₁ off-diagonal contribution
    J = F .* (U' * ΔU)
    term += (J + J') * S
    VΔV = (V * ΔV')
    K = F .* VΔV
    term += S * (K + K')

    # dV₁ diagonal contribution (diagonal of dU₁ is gauged away)
    if scalartype(U) <: Complex && !(ΔV isa ZeroTangent) && !(ΔU isa ZeroTangent)
        L = Diagonal(diag(VΔV))
        term += 0.5 * S⁻¹ * (L' - L)
    end
    ΔA = U * term * V

    # Projector contribution for non-square A
    UUd = U * U'
    VdV = V' * V
    Uproj = one(UUd) - UUd
    Vproj = one(VdV) - VdV
    ΔA += Uproj * ΔU * S⁻¹ * V + U * S⁻¹ * ΔV * Vproj  # Wrong truncation contribution

    return ΔA
end

# Gauge-invariant loss function
function lossfun(A, R=TensorMap(randn, space(A)), svdfunc=tsvd)
    U, _, V = svdfunc(A)
    return real(dot(R, U * V))  # Overlap with random tensor R is gauge-invariant and differentiable, also for m≠n
end

m, n = 20, 30
dtype = ComplexF64
χ = 15
r = TensorMap(randn, dtype, ℂ^m ← ℂ^n)
R = TensorMap(randn, space(r))

println("Non-truncated SVD:")
loldsvd, goldsvd = withgradient(A -> lossfun(A, R, x -> oldsvd(x, min(m, n))), r)
ltensorkit, gtensorkit = withgradient(
    A -> lossfun(A, R, x -> tsvd(x; trunc=truncdim(min(m, n)))), r
)
litersvd, gitersvd = withgradient(A -> lossfun(A, R, x -> itersvd(x, min(m, n))), r)
@show loldsvd ≈ ltensorkit ≈ litersvd
@show norm(gtensorkit[1] - goldsvd[1])
@show norm(gtensorkit[1] - gitersvd[1])

println("\nTruncated SVD with χ=$χ:")
loldsvd, goldsvd = withgradient(A -> lossfun(A, R, x -> oldsvd(x, χ)), r)
ltensorkit, gtensorkit = withgradient(
    A -> lossfun(A, R, x -> tsvd(x; trunc=truncdim(χ))), r
)
litersvd, gitersvd = withgradient(A -> lossfun(A, R, x -> itersvd(x, χ)), r)
@show loldsvd ≈ ltensorkit ≈ litersvd
@show norm(gtensorkit[1] - goldsvd[1])
@show norm(gtensorkit[1] - gitersvd[1])
