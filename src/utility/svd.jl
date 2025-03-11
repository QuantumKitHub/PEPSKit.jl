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
    - `:gmres`: GMRES iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.GMRES) for details
    - `:bicgstab`: BiCGStab iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BiCGStab) for details
    - `:arnoldi`: Arnoldi Krylov algorithm, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Arnoldi) for details
* `broadening=nothing`: Broadening of singular value differences to stabilize the SVD gradient. Currently not implemented.
"""
struct SVDAdjoint{F,R,B}
    fwd_alg::F
    rrule_alg::R
    broadening::B

    # Inner constructor to prohibit illegal setting combinations
    function SVDAdjoint(fwd_alg::F, rrule_alg::R, broadening::B) where {F,R,B}
        if fwd_alg isa FixedSVD && isnothing(rrule_alg)
            throw(
                ArgumentError("FixedSVD and nothing (TensorKit rrule) are not compatible")
            )
        end
        return new{F,R,B}(fwd_alg, rrule_alg, broadening)
    end
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

const SVD_FWD_SYMBOLS = IdDict{Symbol,Any}(
    :sdd => TensorKit.SDD,
    :svd => TensorKit.SVD,
    :iterative =>
        (; tol=1e-14, krylovdim=25, kwargs...) ->
            IterSVD(; alg=GKL(; tol, krylovdim), kwargs...),
)
const SVD_RRULE_SYMBOLS = IdDict{Symbol,Type{<:Any}}(
    :gmres => GMRES, :bicgstab => BiCGStab, :arnoldi => Arnoldi
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
        rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg=nothing)) # remove `alg` keyword argument
        rrule_type <: BiCGStab &&
            (rrule_kwargs = Base.structdiff(rrule_kwargs, (; krylovdim=nothing))) # BiCGStab doens't take `krylovdim`
        rrule_type(; rrule_kwargs...)
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

    # update rrule_alg tolerance to be compatible with smallest singular value
    rrule_alg = alg.rrule_alg
    smallest_sval = minimum(minimum(abs.(diag(b))) for (_, b) in blocks(S))
    proper_tol = clamp(rrule_alg.tol, 1e-14, 1e-2 * smallest_sval)
    rrule_alg = @set rrule_alg.tol = proper_tol

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
            minimal_alg = GKL(; tol=proper_tol, verbosity=1)  # Only tolerance is used for gauge sensitivity (# TODO: How do we not hard-code this tolerance?)

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
                rrule_alg,
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
