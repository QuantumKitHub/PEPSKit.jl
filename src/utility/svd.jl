import TensorKit:
    SectorDict,
    _empty_svdtensors,
    _compute_svddata!,
    _truncate!,
    _implement_svdtruncation!,
    _create_svdtensors

# Plain copy of tsvd!(...) from TensorKit to lift alg type restriction
function _tensorkit_svd!(
    t::TensorMap;
    trunc::TruncationScheme=TensorKit.NoTruncation(),
    p::Real=2,
    alg=TensorKit.SVD(),
)
    #early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return _empty_svdtensors(t)..., truncerr
    end

    S = spacetype(t)
    Udata, Σdata, Vdata, dims = _compute_svddata!(t, alg)
    if !isa(trunc, TensorKit.NoTruncation)
        Σdata, truncerr = _truncate!(Σdata, trunc, p)
        Udata, Σdata, Vdata, dims = _implement_svdtruncation!(t, Udata, Σdata, Vdata, dims)
        W = S(dims)
    else
        truncerr = abs(zero(scalartype(t)))
        W = S(dims)
        if length(domain(t)) == 1 && domain(t)[1] ≅ W
            W = domain(t)[1]
        elseif length(codomain(t)) == 1 && codomain(t)[1] ≅ W
            W = codomain(t)[1]
        end
    end
    return _create_svdtensors(t, Udata, Σdata, Vdata, W)..., truncerr
end

# Wrapper struct around TensorKit's SVD algorithms
@kwdef struct FullSVD
    alg::Union{<:TensorKit.SVD,<:TensorKit.SDD} = TensorKit.SVD()
    lorentz_broad::Float64 = 0.0
end

function svdwrap(t::AbstractTensorMap, alg::FullSVD; trunc=notrunc(), kwargs...)
    # TODO: Replace _tensorkit_svd! with just tsvd eventually to use the full TensorKit machinery
    return _tensorkit_svd!(copy(t); trunc, alg.alg)
end

function ChainRulesCore.rrule(
    ::typeof(svdwrap), t::AbstractTensorMap, alg::FullSVD; trunc=notrunc(), kwargs...
)
    tsvd_return, tsvd!_pullback = ChainRulesCore.rrule(tsvd!, t; alg=TensorKit.SVD(), trunc)
    function svdwrap_fullsvd_pullback(Δ)
        return tsvd!_pullback(Δ)..., NoTangent()
    end
    return tsvd_return, svdwrap_fullsvd_pullback
end

# Wrapper around Krylov Kit's GKL iterative SVD solver
@kwdef struct IterSVD
    alg::KrylovKit.GKL = KrylovKit.GKL(; tol=1e-14, krylovdim=25)
    howmany::Int = 20
    lorentz_broad::Float64 = 0.0
    alg_rrule::Union{GMRES,BiCGStab,Arnoldi} = GMRES(; tol=1e-14)
end

function svdwrap(t::AbstractTensorMap, alg::IterSVD; trunc=notrunc(), kwargs...)
    U, S, V, = _tensorkit_svd!(copy(t); trunc, alg)  # TODO: Also replace this with tsvd eventually
    ϵ = norm(t - U * S * V)  # Compute truncation error separately
    return U, S, V, ϵ
end

# Compute SVD data block-wise using KrylovKit algorithm
function TensorKit._compute_svddata!(t::TensorMap, alg::IterSVD)
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    A = storagetype(t)
    Udata = SectorDict{I,A}()
    Vdata = SectorDict{I,A}()
    dims = SectorDict{I,Int}()
    local Σdata
    for (c, b) in blocks(t)
        x₀ = randn(eltype(b), size(b, 1))
        Σ, lvecs, rvecs, info = svdsolve(b, x₀, alg.howmany, :LR, alg.alg)
        if info.converged < alg.howmany  # Fall back to dense SVD if not properly converged
            U, Σ, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SVD())
            Udata[c] = U
            Vdata[c] = V
        else
            Udata[c] = stack(lvecs)
            Vdata[c] = stack(rvecs)'
        end
        if @isdefined Σdata # cannot easily infer the type of Σ, so use this construction
            Σdata[c] = Σ
        else
            Σdata = SectorDict(c => Σ)
        end
        dims[c] = length(Σ)
    end
    return Udata, Σdata, Vdata, dims
end

# TODO: IterSVD adjoint utilizing KryloVKit svdsolve adjoint

# Full SVD with old adjoint that doesn't account for truncation properly
@kwdef struct OldSVD{A<:Union{FullSVD,IterSVD}}
    alg::A = FullSVD()
    lorentz_broad::Float64 = 0.0
end

function svdwrap(t::AbstractTensorMap, alg::OldSVD; kwargs...)
    return svdwrap(t, alg.alg; kwargs...)
end

# Outdated adjoint not taking truncated part into account for testing purposes
function ChainRulesCore.rrule(
    ::typeof(svdwrap), t::AbstractTensorMap, alg::OldSVD; kwargs...
)
    U, S, V, ϵ = svdwrap(t, alg; kwargs...)

    function svdwrap_oldsvd_pullback((ΔU, ΔS, ΔV, Δϵ))
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

    return (U, S, V, ϵ), svdwrap_oldsvd_pullback
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