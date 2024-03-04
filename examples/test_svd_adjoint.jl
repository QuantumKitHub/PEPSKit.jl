using LinearAlgebra
using TensorKit
using ChainRulesCore, Zygote
using PEPSKit

# Non-proper truncated SVD with outdated adjoint
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
    S = diagm(S)
    V = copy(V')

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
    ΔA += Uproj * ΔU * S⁻¹ * V + U * S⁻¹ * ΔV * Vproj  # Old wrong stuff

    return ΔA
end

# Loss function taking the nfirst first singular vectors into account
function nfirst_loss(A, svdfunc; nfirst=1)
    U, _, V = svdfunc(A)
    U = convert(Array, U)
    V = convert(Array, V)
    return real(sum([U[i, i] * V[i, i] for i in 1:nfirst]))
end

m, n = 30, 20
dtype = ComplexF64
χ = 15
r = TensorMap(randn, dtype, ℂ^m ← ℂ^n)

ltensorkit, gtensorkit = withgradient(A -> nfirst_loss(A, x -> oldsvd(x, χ); nfirst=3), r)
litersvd, gitersvd = withgradient(A -> nfirst_loss(A, x -> itersvd(x, χ); nfirst=3), r)

@show ltensorkit ≈ litersvd
@show gtensorkit ≈ gitersvd