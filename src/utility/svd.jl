using TensorKit:
    SectorDict,
    RealOrComplexFloat,
    NoTruncation,
    TruncationSpace,
    _tsvd!,
    _empty_svdtensors,
    _compute_svddata!,
    _create_svdtensors,
    _compute_truncdim
const TensorKitCRCExt = Base.get_extension(TensorKit, :TensorKitChainRulesCoreExt)
const KrylovKitCRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
    struct SVDAdjoint
    SVDAdjoint(; kwargs...)

Wrapper for a SVD algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
If `isnothing(rrule_alg)`, Zygote differentiates the forward call automatically.
In case of degenerate singular values, one might need a `broadening` scheme which
removes the divergences from the adjoint.

## Keyword arguments

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.svd_fwd_alg))`: SVD algorithm of the forward pass which can either be passed as an `Algorithm` instance or a `NamedTuple` where `alg` is one of the following:
    - `:sdd`: TensorKit's wrapper for LAPACK's `_gesdd`
    - `:svd`: TensorKit's wrapper for LAPACK's `_gesvd`
    - `:iterative`: Iterative SVD only computing the specifed number of singular values and vectors, see ['IterSVD'](@ref)
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.svd_rrule_alg))`: Reverse-rule algorithm for differentiating the SVD. Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
    - `:tsvd`: Uses TensorKit's reverse-rule for `tsvd` which doesn't solve any linear problem and instead requires access to the full SVD, see [TensorKit](https://github.com/Jutho/TensorKit.jl/blob/f9cddcf97f8d001888a26f4dce7408d5c6e2228f/ext/TensorKitChainRulesCoreExt/factorizations.jl#L3)
    - `:gmres`: GMRES iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.GMRES) for details
    - `:bicgstab`: BiCGStab iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BiCGStab) for details
    - `:arnoldi`: Arnoldi Krylov algorithm, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Arnoldi) for details
* `broadening=nothing`: Broadening of singular value differences to stabilize the SVD gradient. Currently not implemented.
"""
struct SVDAdjoint{F,R,B}
    fwd_alg::F
    rrule_alg::R
    broadening::B
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

const SVD_FWD_SYMBOLS = IdDict{Symbol,Any}(
    :sdd => TensorKit.SDD,
    :svd => TensorKit.SVD,
    :iterative =>
        (; tol=1e-14, krylovdim=25, kwargs...) ->
            IterSVD(; alg=GKL(; tol, krylovdim), kwargs...),
)
const SVD_RRULE_SYMBOLS = IdDict{Symbol,Type{<:Any}}(
    :tsvd => Nothing, :gmres => GMRES, :bicgstab => BiCGStab, :arnoldi => Arnoldi
)

function SVDAdjoint(; fwd_alg=(;), rrule_alg=(;), broadening=nothing)
    # parse forward SVD algorithm
    fwd_algorithm = if fwd_alg isa NamedTuple
        fwd_kwargs = (; alg=Defaults.svd_fwd_alg, fwd_alg...) # overwrite with specified kwargs
        haskey(SVD_FWD_SYMBOLS, fwd_kwargs.alg) ||
            throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
        fwd_type = SVD_FWD_SYMBOLS[fwd_kwargs.alg]
        fwd_kwargs = Base.structdiff(fwd_kwargs, (; alg=nothing)) # remove `alg` keyword argument
        fwd_type(; fwd_kwargs...)
    else
        fwd_alg
    end

    # parse reverse-rule SVD algorithm
    rrule_algorithm = if rrule_alg isa NamedTuple
        rrule_kwargs = (;
            alg=Defaults.svd_rrule_alg,
            tol=Defaults.svd_rrule_tol,
            krylovdim=Defaults.svd_rrule_min_krylovdim,
            verbosity=Defaults.svd_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(SVD_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = SVD_RRULE_SYMBOLS[rrule_kwargs.alg]

        # IterSVD is incompatible with tsvd rrule -> default to Arnoldi
        if rrule_type <: Nothing && fwd_algorithm isa IterSVD
            rrule_type = Arnoldi
        end

        if rrule_type <: Nothing
            nothing
        else
            rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg=nothing)) # remove `alg` keyword argument
            rrule_type <: BiCGStab &&
                (rrule_kwargs = Base.structdiff(rrule_kwargs, (; krylovdim=nothing))) # BiCGStab doens't take `krylovdim`
            rrule_type(; rrule_kwargs...)
        end
    else
        rrule_alg
    end

    return SVDAdjoint(fwd_algorithm, rrule_algorithm, broadening)
end

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

## Forward algorithms

# TODO: add `LinearAlgebra.cond` to TensorKit
# Compute condition number smax / smin for diagonal singular value TensorMap
function _condition_number(S::AbstractTensorMap)
    smax = maximum(first ∘ last, blocks(S))
    smin = maximum(last ∘ last, blocks(S))
    return smax / smin
end

# Copy code from TensorKit but additionally return full U, S and V to make compatible with :fixed mode
function TensorKit._tsvd!(
    t::TensorMap{<:RealOrComplexFloat}, alg::SVDAdjoint, trunc::TruncationScheme, p::Real=2
)
    U, S, V⁺, truncerr = tsvd(t; trunc=NoTruncation(), p, alg=alg.fwd_alg)

    if !(trunc isa TensorKit.NoTruncation) && !isempty(blocksectors(t))
        Sdata = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))

        truncdim = TensorKit._compute_truncdim(Sdata, trunc, p)
        truncerr = TensorKit._compute_truncerr(Sdata, truncdim, p)

        SVDdata = TensorKit.SectorDict(
            c => (block(U, c), Sc, block(V⁺, c)) for (c, Sc) in Sdata
        )

        Ũ, S̃, Ṽ⁺ = TensorKit._create_svdtensors(t, SVDdata, truncdim)
    else
        Ũ, S̃, Ṽ⁺ = U, S, V⁺
    end

    # construct info NamedTuple
    condnum = _condition_number(S)
    info = (;
        truncation_error=truncerr, condition_number=condnum, U_full=U, S_full=S, V_full=V⁺
    )
    return Ũ, S̃, Ṽ⁺, info
end

"""
    struct FixedSVD

SVD struct containing a pre-computed decomposition or even multiple ones. Additionally, it
can contain the untruncated full decomposition as well. The call to `tsvd` just returns the
pre-computed U, S and V. In the reverse pass, the SVD adjoint is computed with these exact
U, S, and V and, potentially, the full decompositions if the adjoints needs access to them.
"""
struct FixedSVD{Ut,St,Vt,Utf,Stf,Vtf}
    U::Ut
    S::St
    V::Vt
    U_full::Utf
    S_full::Stf
    V_full::Vtf
end

# check whether the full U, S and V are supplied
function isfullsvd(alg::FixedSVD)
    if isnothing(alg.U_full) || isnothing(alg.S_full) || isnothing(alg.V_full)
        return false
    else
        return true
    end
end

# Return pre-computed SVD
function TensorKit.tsvd!(
    t, alg::SVDAdjoint{F}; trunc::NoTruncation=notrunc(), p::Real=2
) where {F<:FixedSVD}
    svd = alg.fwd_alg
    info = (;
        truncation_error=0,
        condition_number=_condition_number(svd.S),
        U_full=svd.U_full,
        S_full=svd.S_full,
        V_full=svd.V_full,
    )
    return svd.U, svd.S, svd.V, info
end

"""
    struct IterSVD(; alg=KrylovKit.GKL(), fallback_threshold = Inf, start_vector=random_start_vector)
    IterSVD(; kwargs...)

Iterative SVD solver based on KrylovKit's GKL algorithm, adapted to (symmetric) tensors.
The number of targeted singular values is set via the `TruncationSpace` in `ProjectorAlg`.
In particular, this make it possible to specify the targeted singular values block-wise.
In case the symmetry block is too small as compared to the number of singular values, or
the iterative SVD didn't converge, the algorithm falls back to a dense SVD.

## Keyword arguments

* `alg::KrlovKit.GKL=KrylovKit.GKL(; tol=1e-14, krylovdim=25)`: GKL algorithm struct for block-wise iterative SVD.
* `fallback_threshold::Float64=Inf`: Threshold for `howmany / minimum(size(block))` above which (if the block is too small) the algorithm falls back to TensorKit's dense SVD.
* `start_vector=random_start_vector`: Function providing the initial vector for the iterative SVD algorithm.
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
function TensorKit.tsvd!(
    f, alg::SVDAdjoint{F}; trunc::Union{NoTruncation,TruncationSpace}=notrunc(), p::Real=2
) where {F<:IterSVD}
    # early return
    if isempty(blocksectors(f))
        truncation_error = zero(real(scalartype(f)))
        return _empty_svdtensors(f)..., truncation_error
    end

    SVDdata, dims = _compute_svddata!(f, alg.fwd_alg, trunc)
    U, S, V = _create_svdtensors(f, SVDdata, dims)
    truncation_error =
        trunc isa NoTruncation ? abs(zero(scalartype(f))) : norm(U * S * V - f, p)

    # construct info NamedTuple
    condition_number = _condition_number(S)
    info = (;
        truncation_error, condition_number, U_full=nothing, S_full=nothing, V_full=nothing
    )

    return U, S, V, info
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

## Reverse-rule algorithms

# TensorKit.tsvd! rrule with info NamedTuple return value
function ChainRulesCore.rrule(
    ::typeof(TensorKit.tsvd!),
    t::AbstractTensorMap,
    alg::SVDAdjoint{F,R,B};
    trunc::TruncationScheme=TensorKit.NoTruncation(),
    p::Real=2,
) where {F,R<:Nothing,B}
    @assert !(alg.fwd_alg isa IterSVD) "IterSVD is not compatible with tsvd reverse-rule"
    Ũ, S̃, Ṽ⁺, info = tsvd(t, alg; trunc, p)
    U, S, V⁺ = info.U_full, info.S_full, info.V_full # untruncated SVD decomposition

    function tsvd!_nothing_pullback(ΔUSVi)
        ΔU, ΔS, ΔV⁺, = unthunk.(ΔUSVi)
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Sc, V⁺c = block(U, c), block(S, c), block(V⁺, c)
            ΔUc, ΔSc, ΔV⁺c = block(ΔU, c), block(ΔS, c), block(ΔV⁺, c)
            Sdc = view(Sc, diagind(Sc))
            ΔSdc = (ΔSc isa AbstractZero) ? ΔSc : view(ΔSc, diagind(ΔSc))
            TensorKitCRCExt.svd_pullback!(b, Uc, Sdc, V⁺c, ΔUc, ΔSdc, ΔV⁺c)
        end
        return NoTangent(), Δt, NoTangent()
    end
    function tsvd!_nothing_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (Ũ, S̃, Ṽ⁺, info), tsvd!_nothing_pullback
end

# KrylovKit rrule compatible with TensorMaps & function handles
function ChainRulesCore.rrule(
    ::typeof(PEPSKit.tsvd!),
    f,
    alg::SVDAdjoint{F,R,B};
    trunc::TruncationScheme=notrunc(),
    p::Real=2,
) where {F<:Union{IterSVD,FixedSVD},R<:Union{GMRES,BiCGStab,Arnoldi},B}
    U, S, V, info = PEPSKit.tsvd(f, alg; trunc, p)

    function tsvd!_itersvd_pullback(ΔUSVi)
        Δf = similar(f)
        ΔU, ΔS, ΔV, = unthunk.(ΔUSVi)

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

            xs, ys = KrylovKitCRCExt.compute_svdsolve_pullback_data(
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
                KrylovKitCRCExt.construct∂f_svd(
                    HasReverseMode(), block(f, c), lvecs, rvecs, xs, ys
                ),
            )
        end
        return NoTangent(), Δf, NoTangent()
    end
    function tsvd!_itersvd_pullback(::Tuple{ZeroTangent,ZeroTangent,ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V, info), tsvd!_itersvd_pullback
end
