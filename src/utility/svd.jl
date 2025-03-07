using TensorKit:
    SectorDict,
    RealOrComplexFloat,
    _tsvd!,
    _empty_svdtensors,
    _compute_svddata!,
    _create_svdtensors,
    _compute_truncdim,
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
    PEPSKit.tsvd(t, alg; trunc=notrunc(), p=2)

Wrapper around `TensorKit.tsvd` which dispatches on the `alg` argument.
This is needed since a custom adjoint for `PEPSKit.tsvd` may be defined,
depending on the algorithm. E.g., for `IterSVD` the adjoint for a truncated
SVD from `KrylovKit.svdsolve` is used.
"""
PEPSKit.tsvd(t, alg; kwargs...) = PEPSKit.tsvd!(copy(t), alg; kwargs...)
function PEPSKit.tsvd!(t, alg::SVDAdjoint; trunc::TruncationScheme=notrunc(), p::Real=2)
    return TensorKit.tsvd!(t; alg, trunc, p)
end

# TODO: add `LinearAlgebra.cond` to TensorKit
# Compute condition number smax / smin for diagonal singular value TensorMap
function _condition_number(S::AbstractTensorMap)
    smax = maximum(first ∘ last, blocks(S))
    smin = maximum(last ∘ last, blocks(S))
    return smax / smin
end
@non_differentiable _condition_number(S::AbstractTensorMap)

# Copy code from TensorKit but additionally return full U, S and V to make compatible with :fixed mode
function TensorKit._tsvd!(
    t::TensorMap{<:RealOrComplexFloat}, alg::SVDAdjoint, trunc::TruncationScheme, p::Real=2
)
    U, Σ, V⁺, truncerr = tsvd(t; trunc=NoTruncation(), p, alg)

    if !(trunc isa TensorKit.NoTruncation) && !isempty(blocksectors(t))
        Σdata = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(Σ))

        truncdim = TensorKit._compute_truncdim(Σdata, trunc, p)
        truncerr = TensorKit._compute_truncerr(Σdata, truncdim, p)

        SVDdata = TensorKit.SectorDict(
            c => (block(U, c), Σc, block(V⁺, c)) for (c, Σc) in Σdata
        )

        Ũ, Σ̃, Ṽ⁺ = TensorKit._create_svdtensors(t, SVDdata, truncdim)
    else
        Ũ, Σ̃, Ṽ⁺ = U, Σ, V⁺
    end

    # construct info NamedTuple
    truncerr /= norm(Σ)
    condnum = @ignore_derivatives(_condition_number(S))
    info = (; truncation_error=truncerr, condition_number=condnum, U=U, S=Σ, V=V⁺)
    return Ũ, Σ̃, Ṽ⁺, info
end

# adjust TensorKit.tsvd! rrule to info NamedTuple return value
function ChainRulesCore.rrule(
    ::typeof(TensorKit.tsvd!),
    t::AbstractTensorMap;
    trunc::TruncationScheme=TensorKit.NoTruncation(),
    p::Real=2,
    alg::SVDAdjoint=SVDAdjoint(),
)
    Ũ, Σ̃, Ṽ⁺, info = tsvd(t; trunc, p, alg)
    Ũ, Σ̃, Ṽ⁺ = info.U, info.S, info.V # untruncated SVD decomposition

    function tsvd!_pullback(ΔUSVϵ)
        ΔU, ΔΣ, ΔV⁺, = unthunk.(ΔUSVϵ)
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Σc, V⁺c = block(U, c), block(Σ, c), block(V⁺, c)
            ΔUc, ΔΣc, ΔV⁺c = block(ΔU, c), block(ΔΣ, c), block(ΔV⁺, c)
            Σdc = view(Σc, diagind(Σc))
            ΔΣdc = (ΔΣc isa AbstractZero) ? ΔΣc : view(ΔΣc, diagind(ΔΣc))
            svd_pullback!(b, Uc, Σdc, V⁺c, ΔUc, ΔΣdc, ΔV⁺c)
        end
        return NoTangent(), Δt
    end
    function tsvd!_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent()
    end

    return (Ũ, Σ̃, Ṽ⁺, info), tsvd!_pullback
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
    start_vector = random_start_vector
end

function random_start_vector(t::AbstractMatrix)
    return randn(scalartype(t), size(t, 1))
end

# Compute SVD data block-wise using KrylovKit algorithm
function TensorKit._tsvd!(
    f, alg::IterSVD, trunc::Union{NoTruncation,TruncationSpace}, p::Real=2
)
    # early return
    if isempty(blocksectors(f))
        truncerr = zero(real(scalartype(f)))
        return _empty_svdtensors(f)..., truncerr
    end

    SVDdata, dims = _compute_svddata!(f, alg, trunc)
    U, S, V = _create_svdtensors(f, SVDdata, dims)
    truncerr = trunc isa NoTruncation ? abs(zero(scalartype(f))) : norm(U * S * V - f, p)

    return U, S, V, truncerr
end
function TensorKit._compute_svddata!(
    f, alg::IterSVD, trunc::Union{NoTruncation,TruncationSpace}
)
    InnerProductStyle(f) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(f)
    dims = SectorDict{I,Int}()

    generator = Base.Iterators.map(blocks(f)) do (c, b)
        howmany = trunc isa NoTruncation ? minimum(size(b)) : blockdim(trunc.space, c)

        if howmany / minimum(size(b)) > alg.fallback_threshold  # Use dense SVD for small blocks
            U, S, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SDD())
            U = U[:, 1:howmany]
            V = V[1:howmany, :]
        else
            x₀ = alg.start_vector(b)
            S, lvecs, rvecs, info = KrylovKit.svdsolve(b, x₀, howmany, :LR, alg.alg)
            if info.converged < howmany  # Fall back to dense SVD if not properly converged
                @warn "Iterative SVD did not converge for block $c, falling back to dense SVD"
                U, S, V = TensorKit.MatrixAlgebra.svd!(b, TensorKit.SDD())
                U = U[:, 1:howmany]
                V = V[1:howmany, :]
            else  # Slice in case more values were converged than requested
                U = stack(view(lvecs, 1:howmany))
                V = stack(conj, view(rvecs, 1:howmany); dims=1)
            end
        end

        resize!(S, howmany)
        dims[c] = length(S)
        return c => (U, S, V)
    end

    SVDdata = SectorDict(generator)
    return SVDdata, dims
end

# Rrule with custom pullback to make KrylovKit rrule compatible with TensorMaps & function handles
function ChainRulesCore.rrule(
    ::typeof(PEPSKit.tsvd!),
    f,
    alg::SVDAdjoint{F,R,B};
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
) where {F<:Union{IterSVD,FixedSVD},R<:Union{GMRES,BiCGStab,Arnoldi},B}
    U, S, V, ϵ = PEPSKit.tsvd(f, alg; trunc, p)

    function tsvd!_itersvd_pullback(ΔUSVϵ)
        Δf = similar(f)
        ΔU, ΔS, ΔV, = unthunk.(ΔUSVϵ)

        for (c, b) in blocks(Δf)
            Uc, Sc, Vc = block(U, c), block(S, c), block(V, c)
            ΔUc, ΔSc, ΔVc = block(ΔU, c), block(ΔS, c), block(ΔV, c)
            Sdc = view(Sc, diagind(Sc))
            ΔSdc = ΔSc isa AbstractZero ? ΔSc : view(ΔSc, diagind(ΔSc))

            n_vals = length(Sdc)
            lvecs = Vector{Vector{scalartype(f)}}(eachcol(Uc))
            rvecs = Vector{Vector{scalartype(f)}}(eachcol(Vc'))

            # Dummy objects only used for warnings
            minimal_info = KrylovKit.ConvergenceInfo(n_vals, nothing, nothing, -1, -1)  # Only num. converged is used
            minimal_alg = GKL(; tol=1e-6)  # Only tolerance is used for gauge sensitivity (# TODO: How do we not hard-code this tolerance?)

            if ΔUc isa AbstractZero && ΔVc isa AbstractZero  # Handle ZeroTangent singular vectors
                Δlvecs = fill(ZeroTangent(), n_vals)
                Δrvecs = fill(ZeroTangent(), n_vals)
            else
                Δlvecs = Vector{Vector{scalartype(f)}}(eachcol(ΔUc))
                Δrvecs = Vector{Vector{scalartype(f)}}(eachcol(ΔVc'))
            end

            xs, ys = CRCExt.compute_svdsolve_pullback_data(
                ΔSc isa AbstractZero ? fill(zero(Sc[1]), n_vals) : ΔSdc,
                Δlvecs,
                Δrvecs,
                Sdc,
                lvecs,
                rvecs,
                minimal_info,
                block(f, c),
                :LR,
                minimal_alg,
                alg.rrule_alg,
            )
            copyto!(
                b,
                CRCExt.construct∂f_svd(HasReverseMode(), block(f, c), lvecs, rvecs, xs, ys),
            )
        end
        return NoTangent(), Δf, NoTangent()
    end
    function tsvd!_itersvd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V, ϵ), tsvd!_itersvd_pullback
end
