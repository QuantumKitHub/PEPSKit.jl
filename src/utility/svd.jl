using TensorKit:
    SectorDict,
    _tsvd!,
    _empty_svdtensors,
    _compute_svddata!,
    _create_svdtensors,
    NoTruncation,
    TruncationSpace
CRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
    PEPSKit.tsvd(t::AbstractTensorMap, alg; trunc=notrunc(), p=2)

Wrapper around `TensorKit.tsvd` which dispatches on the `alg` argument.
This is needed since a custom adjoint for `PEPSKit.tsvd` may be defined,
depending on the algorithm. E.g., for `IterSVD` the adjoint for a truncated
SVD from `KrylovKit.svdsolve` is used.
"""
PEPSKit.tsvd(t::AbstractTensorMap, alg; kwargs...) = PEPSKit.tsvd!(copy(t), alg; kwargs...)
function PEPSKit.tsvd!(
    t::AbstractTensorMap, alg; trunc::TruncationScheme=notrunc(), p::Real=2
)
    return TensorKit.tsvd!(t; alg, trunc, p)
end

# Wrapper around Krylov Kit's GKL iterative SVD solver
@kwdef struct IterSVD
    alg::KrylovKit.GKL = KrylovKit.GKL(; tol=1e-14, krylovdim=25)
    fallback_threshold::Float64 = Inf
    lorentz_broad::Float64 = 0.0
    alg_rrule::Union{GMRES,BiCGStab,Arnoldi} = GMRES(; tol=1e-14)
end

# Compute SVD data block-wise using KrylovKit algorithm
function TensorKit._tsvd!(
    t, alg::Union{IterSVD}, trunc::Union{NoTruncation,TruncationSpace}, p::Real=2
)
    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return _empty_svdtensors(t)..., truncerr
    end

    Udata, Σdata, Vdata, dims = _compute_svddata!(t, alg, trunc)
    U, S, V = _create_svdtensors(t, Udata, Σdata, Vdata, spacetype(t)(dims))
    truncerr = trunc isa NoTruncation ? abs(zero(scalartype(t))) : norm(U * S * V - t, p)

    return U, S, V, truncerr
end
function TensorKit._compute_svddata!(
    t::TensorMap, alg::IterSVD, trunc::Union{NoTruncation,TruncationSpace}
)
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    A = storagetype(t)
    Udata = SectorDict{I,A}()
    Vdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    local Sdata
    for (c, b) in blocks(t)
        x₀ = randn(eltype(b), size(b, 1))
        howmany = trunc isa NoTruncation ? minimum(size(b)) : blockdim(trunc.space, c)

        if howmany / minimum(size(b)) > alg.fallback_threshold  # Use dense SVD for small blocks
            U, S, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SVD())
            Udata[c] = @view U[:, 1:howmany]
            Vdata[c] = @view V[1:howmany, :]
        else
            S, lvecs, rvecs, info = KrylovKit.svdsolve(b, x₀, howmany, :LR, alg.alg)
            if info.converged < howmany  # Fall back to dense SVD if not properly converged
                U, S, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SVD())
                Udata[c] = @view U[:, 1:howmany]
                Vdata[c] = @view V[1:howmany, :]
            else  # Slice in case more values were converged than requested
                Udata[c] = stack(view(lvecs, 1:howmany))
                Vdata[c] = stack(conj, view(rvecs, 1:howmany); dims=1)
            end
        end

        S = @view S[1:howmany]
        if @isdefined Sdata # cannot easily infer the type of Σ, so use this construction
            Sdata[c] = S
        else
            Sdata = SectorDict(c => S)
        end
        dims[c] = length(S)
    end
    return Udata, Sdata, Vdata, dims
end

# IterSVD adjoint for tsvd! using KrylovKit.svdsolve adjoint machinery for each block
function ChainRulesCore.rrule(
    ::typeof(PEPSKit.tsvd!),
    t::AbstractTensorMap,
    alg::IterSVD;
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
)
    U, S, V, ϵ = PEPSKit.tsvd(t, alg; trunc, p)

    function tsvd_itersvd_pullback((ΔU, ΔS, ΔV, Δϵ))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Sc, Vc = block(U, c), block(S, c), block(V, c)
            ΔUc, ΔSc, ΔVc = block(ΔU, c), block(ΔS, c), block(ΔV, c)
            Sdc = view(Sc, diagind(Sc))
            ΔSdc = ΔSc isa AbstractZero ? ΔSc : view(ΔSc, diagind(ΔSc))

            n_vals = length(Sdc)
            lvecs = Vector{Vector{scalartype(t)}}(eachcol(Uc))
            rvecs = Vector{Vector{scalartype(t)}}(eachcol(Vc'))
            minimal_info = KrylovKit.ConvergenceInfo(length(Sdc), nothing, nothing, -1, -1)  # Just supply converged to SVD pullback

            if ΔUc isa AbstractZero && ΔVc isa AbstractZero  # Handle ZeroTangent singular vectors
                Δlvecs = fill(ZeroTangent(), n_vals)
                Δrvecs = fill(ZeroTangent(), n_vals)
            else
                Δlvecs = Vector{Vector{scalartype(t)}}(eachcol(ΔUc))
                Δrvecs = Vector{Vector{scalartype(t)}}(eachcol(ΔVc'))
            end

            xs, ys = CRCExt.compute_svdsolve_pullback_data(
                ΔSc isa AbstractZero ? fill(zero(Sc[1]), n_vals) : ΔSdc,
                Δlvecs,
                Δrvecs,
                Sdc,
                lvecs,
                rvecs,
                minimal_info,
                block(t, c),
                :LR,
                alg.alg,
                alg.alg_rrule,
            )
            copyto!(
                b,
                CRCExt.construct∂f_svd(HasReverseMode(), block(t, c), lvecs, rvecs, xs, ys),
            )
        end
        return NoTangent(), Δt, NoTangent()
    end
    function tsvd_itersvd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V, ϵ), tsvd_itersvd_pullback
end

# Full SVD with old adjoint that doesn't account for truncation properly
@kwdef struct OldSVD{A<:Union{TensorKit.SDD,TensorKit.SVD,IterSVD}}
    alg::A = TensorKit.SVD()
    lorentz_broad::Float64 = 0.0
end

# Perform TensorKit.SVD in forward pass 
function TensorKit._tsvd!(t, ::OldSVD, trunc::TruncationScheme, p::Real=2)
    return _tsvd!(t, TensorKit.SVD(), trunc, p)
end

# Use outdated adjoint in reverse pass (not taking truncated part into account for testing purposes)
function ChainRulesCore.rrule(
    ::typeof(PEPSKit.tsvd),
    t::AbstractTensorMap,
    alg::OldSVD;
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
)
    U, S, V, ϵ = PEPSKit.tsvd(t, alg; trunc, p)

    function tsvd_oldsvd_pullback((ΔU, ΔS, ΔV, Δϵ))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Sc, Vc = block(U, c), block(S, c), block(V, c)
            ΔUc, ΔSc, ΔVc = block(ΔU, c), block(ΔS, c), block(ΔV, c)
            copyto!(
                b, oldsvd_rev(Uc, Sc, Vc, ΔUc, ΔSc, ΔVc; lorentz_broad=alg.lorentz_broad)
            )
        end
        return NoTangent(), Δt, NoTangent()
    end
    function tsvd_oldsvd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
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
