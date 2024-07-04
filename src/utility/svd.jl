using TensorKit: SectorDict

# Wrapper around Krylov Kit's GKL iterative SVD solver
@kwdef struct IterSVD
    alg::KrylovKit.GKL = KrylovKit.GKL(; tol=1e-14, krylovdim=25)
    lorentz_broad::Float64 = 0.0
    alg_rrule::Union{GMRES,BiCGStab,Arnoldi} = GMRES(; tol=1e-14)
end

# Compute SVD data block-wise using KrylovKit algorithm
function TensorKit._tsvd!(t, alg::IterSVD, trunc::TruncationScheme, p::Real=2)
    # TODO
end

# function TensorKit._compute_svddata!(t::TensorMap, alg::IterSVD)
#     InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:tsvd!)
#     I = sectortype(t)
#     A = storagetype(t)
#     Udata = SectorDict{I,A}()
#     Vdata = SectorDict{I,A}()
#     dims = SectorDict{I,Int}()
#     local Σdata
#     for (c, b) in blocks(t)
#         x₀ = randn(eltype(b), size(b, 1))
#         Σ, lvecs, rvecs, info = KrylovKit.svdsolve(b, x₀, alg.howmany, :LR, alg.alg)
#         if info.converged < alg.howmany  # Fall back to dense SVD if not properly converged
#             U, Σ, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SVD())
#             Udata[c] = U
#             Vdata[c] = V
#         else
#             Udata[c] = stack(lvecs)
#             Vdata[c] = stack(rvecs)'
#         end
#         if @isdefined Σdata # cannot easily infer the type of Σ, so use this construction
#             Σdata[c] = Σ
#         else
#             Σdata = SectorDict(c => Σ)
#         end
#         dims[c] = length(Σ)
#     end
#     return Udata, Σdata, Vdata, dims
# end

function ChainRulesCore.rrule(
    ::typeof(TensorKit.tsvd),
    t::AbstractTensorMap;
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
    alg::IterSVD=IterSVD(),
)
    # TODO: IterSVD adjoint utilizing KryloVKit svdsolve adjoint
end


# Full SVD with old adjoint that doesn't account for truncation properly
@kwdef struct OldSVD{A<:Union{FullSVD,IterSVD}}
    alg::A = FullSVD()
    lorentz_broad::Float64 = 0.0
end

# Perform TensorKit.SVD in forward pass 
function TensorKit._tsvd!(t, ::OldSVD, trunc::TruncationScheme, p::Real=2)
    return TensorKit._tsvd(t, TensorKit.SVD(), trunc, p)
end

# Use outdated adjoint in reverse pass (not taking truncated part into account for testing purposes)
function ChainRulesCore.rrule(
    ::typeof(TensorKit.tsvd),
    t::AbstractTensorMap;
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
    alg::OldSVD=OldSVD(),
)
    U, S, V, ϵ = tsvd(t; trunc, p, alg)

    function tsvd_oldsvd_pullback((ΔU, ΔS, ΔV, Δϵ))
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
                    lorentz_broad=alg.lorentz_broad,
                ),
            )
        end
        return NoTangent(), ∂t, NoTangent()
    end

    return (U, S, V, ϵ), tsvd_oldsvd_pullback
end

function oldsvd_rev(
    U::AbstractMatrix,
    S::AbstractMatrix,
    V::AbstractMatrix,
    ΔU,
    ΔS,
    ΔV;
    lorentz_broad=0,
    atol::Real=0,
    rtol::Real=atol > 0 ? 0 : eps(scalartype(S))^(3 / 4),
)
    tol = atol > 0 ? atol : rtol * S[1, 1]
    F = _invert_S²(S, tol, lorentz_broad)  # Includes Lorentzian broadening
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

# Computation of F in SVD adjoint, including Lorentzian broadening
function _invert_S²(S::AbstractMatrix{T}, tol::Real, ε=0) where {T<:Real}
    F = similar(S)
    @inbounds for i in axes(F, 1), j in axes(F, 2)
        F[i, j] = if i == j
            zero(T)
        else
            sᵢ, sⱼ = S[i, i], S[j, j]
            Δs = abs(sⱼ - sᵢ) < tol ? tol : sⱼ^2 - sᵢ^2
            ε > 0 && (Δs = _lorentz_broaden(Δs, ε))
            1 / Δs
        end
    end
    return F
end

# Lorentzian broadening for SVD adjoint F-singularities
function _lorentz_broaden(x::Real, ε=1e-12)
    x′ = 1 / x
    return x′ / (x′^2 + ε)
end
