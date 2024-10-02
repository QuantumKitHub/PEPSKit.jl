using TensorKit:
    SectorDict,
    _tsvd!,
    _empty_svdtensors,
    _compute_svddata!,
    _create_svdtensors,
    NoTruncation,
    TruncationSpace

const CRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
    struct SVDAdjoint(; fwd_alg=Defaults.fwd_alg, rrule_alg=Defaults.rrule_alg,
                      broadening=nothing)

Wrapper for a SVD algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
If `isnothing(rrule_alg)`, Zygote differentiates the forward call automatically.
In case of degenerate singular values, one might need a `broadening` scheme which
removes the divergences from the adjoint.
"""
@kwdef struct SVDAdjoint{F,R,B}
    fwd_alg::F = Defaults.fwd_alg
    rrule_alg::R = Defaults.rrule_alg
    broadening::B = nothing
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

"""
    PEPSKit.tsvd(t::AbstractTensorMap, alg; trunc=notrunc(), p=2)

Wrapper around `TensorKit.tsvd` which dispatches on the `alg` argument.
This is needed since a custom adjoint for `PEPSKit.tsvd` may be defined,
depending on the algorithm. E.g., for `IterSVD` the adjoint for a truncated
SVD from `KrylovKit.svdsolve` is used.
"""
PEPSKit.tsvd(t::AbstractTensorMap, alg; kwargs...) = PEPSKit.tsvd!(copy(t), alg; kwargs...)
function PEPSKit.tsvd!(
    t::AbstractTensorMap, alg::SVDAdjoint; trunc::TruncationScheme=notrunc(), p::Real=2
)
    return TensorKit.tsvd!(t; alg=alg.fwd_alg, trunc, p)
end

"""
    struct FixedSVD

SVD struct containing a pre-computed decomposition or even multiple ones.
The call to `tsvd` just returns the pre-computed U, S and V. In the reverse
pass, the SVD adjoint is computed with these exact U, S, and V.
"""
struct FixedSVD{Ut,St,Vt}
    U::Ut
    S::St
    V::Vt
end

# Return pre-computed SVD
function TensorKit._tsvd!(t, alg::FixedSVD, ::NoTruncation, ::Real=2)
    return alg.U, alg.S, alg.V, 0
end

"""
    struct IterSVD(; alg=KrylovKit.GKL(), fallback_threshold = Inf)

Iterative SVD solver based on KrylovKit's GKL algorithm, adapted to (symmetric) tensors.
The number of targeted singular values is set via the `TruncationSpace` in `ProjectorAlg`.
In particular, this make it possible to specify the targeted singular values block-wise.
In case the symmetry block is too small as compared to the number of singular values, or
the iterative SVD didn't converge, the algorithm falls back to a dense SVD.
"""
@kwdef struct IterSVD
    alg::KrylovKit.GKL = KrylovKit.GKL(; tol=1e-14, krylovdim=25)
    fallback_threshold::Float64 = Inf
end

# Compute SVD data block-wise using KrylovKit algorithm
function TensorKit._tsvd!(
    t, alg::IterSVD, trunc::Union{NoTruncation,TruncationSpace}, p::Real=2
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
        howmany = trunc isa NoTruncation ? minimum(size(b)) : blockdim(trunc.space, c)

        if howmany / minimum(size(b)) > alg.fallback_threshold  # Use dense SVD for small blocks
            U, S, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SVD())
            Udata[c] = U[:, 1:howmany]
            Vdata[c] = V[1:howmany, :]
        else
            # x₀ = randn(eltype(b), size(b, 1))  # Leads to erroneous gauge fixing of U, S, V and thus failing element-wise conv.
            # u, = TensorKit.MatrixAlgebra.svd!(deepcopy(b), TensorKit.SVD())
            # x₀ = sum(u[:, i] for i in 1:howmany)  # Element-wise convergence works fine
            # x₀ = dropdims(sum(b[:, 1:3]; dims=2); dims=2)  # Summing too many columns also makes gauge fixing fail
            x₀ = b[:, 1]  # Leads so slower convergence of SVD than randn, but correct element-wise convergence
            S, lvecs, rvecs, info = KrylovKit.svdsolve(b, x₀, howmany, :LR, alg.alg)
            if info.converged < howmany  # Fall back to dense SVD if not properly converged
                @warn "Iterative SVD did not converge for block $c, falling back to dense SVD"
                U, S, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SVD())
                Udata[c] = U[:, 1:howmany]
                Vdata[c] = V[1:howmany, :]
            else  # Slice in case more values were converged than requested
                Udata[c] = stack(view(lvecs, 1:howmany))
                Vdata[c] = stack(conj, view(rvecs, 1:howmany); dims=1)
            end
        end

        resize!(S, howmany)
        if @isdefined Sdata
            Sdata[c] = S
        else
            Sdata = SectorDict(c => S)
        end
        dims[c] = length(S)
    end
    return Udata, Sdata, Vdata, dims
end

function ChainRulesCore.rrule(
    ::typeof(PEPSKit.tsvd!),
    t::AbstractTensorMap,
    alg::SVDAdjoint{F,R,B};
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
) where {F<:Union{IterSVD,FixedSVD},R<:Union{GMRES,BiCGStab,Arnoldi},B}
    U, S, V, ϵ = PEPSKit.tsvd(t, alg; trunc, p)

    function tsvd!_itersvd_pullback((ΔU, ΔS, ΔV, Δϵ))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Sc, Vc = block(U, c), block(S, c), block(V, c)
            ΔUc, ΔSc, ΔVc = block(ΔU, c), block(ΔS, c), block(ΔV, c)
            Sdc = view(Sc, diagind(Sc))
            ΔSdc = ΔSc isa AbstractZero ? ΔSc : view(ΔSc, diagind(ΔSc))

            n_vals = length(Sdc)
            lvecs = Vector{Vector{scalartype(t)}}(eachcol(Uc))
            rvecs = Vector{Vector{scalartype(t)}}(eachcol(Vc'))

            # Dummy objects only used for warnings
            minimal_info = KrylovKit.ConvergenceInfo(n_vals, nothing, nothing, -1, -1)  # Only num. converged is used
            minimal_alg = GKL(; tol=1e-6)  # Only tolerance is used for gauge sensitivity (# TODO: How do we not hard-code this tolerance?)

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
                minimal_alg,
                alg.rrule_alg,
            )
            copyto!(
                b,
                CRCExt.construct∂f_svd(HasReverseMode(), block(t, c), lvecs, rvecs, xs, ys),
            )
        end
        return NoTangent(), Δt, NoTangent()
    end
    function tsvd!_itersvd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V, ϵ), tsvd!_itersvd_pullback
end

"""
    struct NonTruncAdjoint

Old SVD adjoint that does not account for the truncated part of truncated SVDs.
"""
struct NonTruncSVDAdjoint end

# Use outdated adjoint in reverse pass (not taking truncated part into account for testing purposes)
function ChainRulesCore.rrule(
    ::typeof(PEPSKit.tsvd!),
    t::AbstractTensorMap,
    alg::SVDAdjoint{F,NonTruncSVDAdjoint,B};
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
) where {F,B}
    U, S, V, ϵ = PEPSKit.tsvd(t, alg; trunc, p)

    function tsvd!_nontruncsvd_pullback((ΔU, ΔS, ΔV, Δϵ))
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Sc, Vc = block(U, c), block(S, c), block(V, c)
            ΔUc, ΔSc, ΔVc = block(ΔU, c), block(ΔS, c), block(ΔV, c)
            copyto!(
                b, oldsvd_rev(Uc, Sc, Vc, ΔUc, ΔSc, ΔVc; lorentz_broadening=alg.broadening)
            )
        end
        return NoTangent(), Δt, NoTangent()
    end
    function tsvd!_nontruncsvd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V, ϵ), tsvd!_nontruncsvd_pullback
end

function oldsvd_rev(
    U::AbstractMatrix,
    S::AbstractMatrix,
    V::AbstractMatrix,
    ΔU,
    ΔS,
    ΔV;
    lorentz_broadening=0,
    atol::Real=0,
    rtol::Real=atol > 0 ? 0 : eps(scalartype(S))^(3 / 4),
)
    tol = atol > 0 ? atol : rtol * S[1, 1]
    F = _invert_S²(S, tol, lorentz_broadening)  # Includes Lorentzian broadening
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
