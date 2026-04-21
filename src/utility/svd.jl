const KrylovKitCRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
$(TYPEDEF)

SVD reverse-rule algorithm which wraps MatrixAlgebraKit's `svd_pullback!`.

## Fields

$(TYPEDFIELDS)

## Constructors

    FullSVDPullback(; kwargs...)

Construct a `FullSVDPullback` algorithm struct from the following keyword arguments:

* `degeneracy_atol::Real=$(Defaults.rrule_degeneracy_atol)` : Broadening amplitude for smoothing divergent term in SVD derivative in case of (pseudo) degenerate singular values.
* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct FullSVDPullback
    degeneracy_atol::Real = Defaults.rrule_degeneracy_atol
    verbosity::Int = 0
end

"""
$(TYPEDEF)

SVD reverse-rule algorithm which wraps MatrixAlgebraKit's `svd_trunc_pullback!`.

## Fields

$(TYPEDFIELDS)

## Constructors

    TruncSVDPullback(; kwargs...)

Construct a `TruncSVDPullback` algorithm struct from the following keyword arguments:

* `degeneracy_atol::Real=$(Defaults.rrule_degeneracy_atol)` : Broadening amplitude for smoothing divergent term in SVD derivative in case of (pseudo) degenerate singular values.
* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct TruncSVDPullback
    degeneracy_atol::Real = Defaults.rrule_degeneracy_atol
    verbosity::Int = 0
end

"""
$(TYPEDEF)

Wrapper for a SVD algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.

## Fields

$(TYPEDFIELDS)

## Constructors

    SVDAdjoint(; kwargs...)

Construct a `SVDAdjoint` algorithm struct based on the following keyword arguments:

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.svd_fwd_alg))`: SVD
  algorithm of the forward pass which can either be passed as an `Algorithm` instance or a
  `NamedTuple` where the algorithm is specified by the `alg` keyword.
  The available SVD algorithms can be divided into two categories:
    - "Dense" SVD algorithms which compute a truncated SVD through the truncation of a full
      [`MatrixAlgebraKit.svd_compact!`](@extref) decomposition.
      Available algorithms are:
        - `:DefaultAlgorithm` : MatrixAlgebraKit's [default SVD algorithm](@extref MatrixAlgebraKit.DefaultAlgorithm) for a given matrix type.
        - `:DivideAndConquer` : MatrixAlgebraKit's [`DivideAndConquer`](@extref)
        - `:QRIteration` : MatrixAlgebraKit's [`QRIteration`](@extref)
        - `:Bisection` : MatrixAlgebraKit's [`Bisection`](@extref)
        - `:Jacobi` : MatrixAlgebraKit's [`Jacobi`](@extref)
        - `:SVDViaPolar` : MatrixAlgebraKit's [`SVDViaPolar`](@extref)
        - `:SafeDivideAndConquer` : MatrixAlgebraKit's [`SafeDivideAndConquer`](@extref)
    - "Sparse" SVD algorithms which directly compute a truncated SVD without access to the
      full decomposition. Available algorithms are:
        - `:iterative` : Iterative Krylov-based SVD only computing the specifed number of
          singular values and vectors, see [`IterSVD`](@ref)
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:notrunc)` : Truncation strategy for the truncated SVD, which controls the spaces of the output. Here, `alg` can be one of the following:
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate to the maximal virtual dimension of `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=:$(Defaults.svd_rrule_alg))`:
  Reverse-rule algorithm for differentiating the SVD. Can be supplied by an `Algorithm`
  instance directly or as a `NamedTuple` where `alg` is one of the following:
    - `:full` : MatrixAlgebraKit's [`svd_pullback!`](@extref) that requires access to the full spectrum
    - `:trunc` : MatrixAlgebraKit's [`svd_trunc_pullback!`](@extref) solving a Sylvester equation on the truncated subspace
    - `:GMRES` : GMRES iterative linear solver, see [`KrylovKit.GMRES`](@extref)
    - `:BiCGStab` : BiCGStab iterative linear solver, see [`KrylovKit.BiCGStab`](@extref)
    - `:Arnoldi` : Arnoldi Krylov algorithm, see the [`KrylovKit.Arnoldi`](@extref)

!!! note
    Manually specifying a `rrule_alg` is considered expert-mode usage, and should only be done when full control over the implementation is desired.
    For all regular use cases, the default reverse rule algorithms, automatically chosen based on the forward algorithm, should be sufficient.
"""
struct SVDAdjoint{F, R, T}
    fwd_alg::F
    rrule_alg::R
    trunc::T
end

const SVD_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :DefaultAlgorithm => DefaultAlgorithm,
    :DivideAndConquer => DivideAndConquer,
    :QRIteration => QRIteration,
    :Bisection => Bisection,
    :Jacobi => Jacobi,
    :SVDViaPolar => SVDViaPolar,
    :SafeDivideAndConquer => SafeDivideAndConquer,
    :iterative => (; tol = 1.0e-14, krylovdim = 25, kwargs...) -> IterSVD(; alg = GKL(; tol, krylovdim), kwargs...),
)
const SVD_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :full => FullSVDPullback, :trunc => TruncSVDPullback,
    :GMRES => GMRES, :BiCGStab => BiCGStab, :Arnoldi => Arnoldi
)

_default_svd_rrule_alg(::MatrixAlgebraKit.Algorithm) = :full

function SVDAdjoint(; fwd_alg = (;), rrule_alg = (;), trunc = (; alg = :notrunc))
    # parse forward SVD algorithm
    fwd_algorithm = if fwd_alg isa NamedTuple
        fwd_kwargs = (; alg = Defaults.svd_fwd_alg, fwd_alg...) # overwrite with specified kwargs
        haskey(SVD_FWD_SYMBOLS, fwd_kwargs.alg) ||
            throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
        fwd_type = SVD_FWD_SYMBOLS[fwd_kwargs.alg]
        fwd_kwargs = Base.structdiff(fwd_kwargs, (; alg = nothing)) # remove `alg` keyword argument
        fwd_type(; fwd_kwargs...)
    else
        fwd_alg
    end

    # parse reverse-rule SVD algorithm
    rrule_algorithm = if rrule_alg isa NamedTuple
        rrule_kwargs = (;
            alg = _default_svd_rrule_alg(fwd_algorithm), # default rrule depends on forward algorithm
            tol = Defaults.svd_rrule_tol,
            krylovdim = Defaults.svd_rrule_min_krylovdim,
            degeneracy_atol = Defaults.rrule_degeneracy_atol,
            verbosity = Defaults.svd_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(SVD_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = SVD_RRULE_SYMBOLS[rrule_kwargs.alg]

        # IterSVD is incompatible with tsvd rrule -> default to Arnoldi
        if rrule_type <: FullSVDPullback && fwd_algorithm isa IterSVD
            rrule_type = Arnoldi
        end

        if rrule_type <: Union{FullSVDPullback, TruncSVDPullback}
            rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg = nothing, tol = 0.0, krylovdim = 0)) # remove `alg`, `tol` and `krylovdim` keyword arguments
        else
            rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg = nothing, degeneracy_atol = 0.0)) # remove `alg` and `degeneracy_atol` keyword arguments
            rrule_type <: BiCGStab &&
                (rrule_kwargs = Base.structdiff(rrule_kwargs, (; krylovdim = nothing))) # BiCGStab doesn't take `krylovdim`
        end
        rrule_type(; rrule_kwargs...)
    else
        rrule_alg
    end

    # parse truncation scheme
    truncation_strategy = if trunc isa TruncationStrategy
        trunc
    elseif trunc isa NamedTuple
        _TruncationStrategy(; trunc...)
    else
        throw(ArgumentError("unknown trunc $trunc"))
    end

    return SVDAdjoint(fwd_algorithm, rrule_algorithm, truncation_strategy)
end

"""
    svd_trunc(t, alg::SVDAdjoint; trunc=notrunc())
    svd_trunc!(t, alg::SVDAdjoint; trunc=notrunc())

Wrapper around `svd_trunc(!)` which dispatches on the `SVDAdjoint` algorithm.
This is needed since a custom adjoint may be defined, depending on the `alg`.
E.g., for `IterSVD` the adjoint for a truncated SVD from `KrylovKit.svdsolve` is used.
"""
MatrixAlgebraKit.svd_trunc(t, alg::SVDAdjoint) = svd_trunc!(copy(t), alg)
function MatrixAlgebraKit.svd_trunc!(t, alg::SVDAdjoint)
    return svd_trunc!(t, TruncatedAlgorithm(alg.fwd_alg, alg.trunc))
end
function MatrixAlgebraKit.svd_trunc!(t::AdjointTensorMap, alg::SVDAdjoint)
    u, s, vt, ϵ = svd_trunc!(adjoint(t), alg)
    return adjoint(vt), adjoint(s), adjoint(u), ϵ
end

#
## Forward algorithms
#

"""
$(TYPEDEF)

Iterative SVD solver based on KrylovKit's GKL algorithm, adapted to (symmetric) tensors.
The number of targeted singular values is set via the `truncspace` in `ProjectorAlg`.
In particular, this make it possible to specify the targeted singular values block-wise.
In case the symmetry block is too small as compared to the number of singular values, or
the iterative SVD didn't converge, the algorithm falls back to a dense SVD.

## Fields

$(TYPEDFIELDS)

## Constructors

    IterSVD(; kwargs...)

Construct an `IterSVD` algorithm struct based on the following keyword arguments:

* `alg::KrylovKit.GKL=KrylovKit.GKL(; tol=1e-14, krylovdim=25)` : GKL algorithm struct for block-wise iterative SVD.
* `fallback_threshold::Float64=Inf` : Threshold for `howmany / minimum(size(block))` above which (if the block is too small) the algorithm falls back to TensorKit's dense SVD.
* `start_vector=deterministic_start_vector` : Function providing the initial vector for the iterative SVD algorithm.
"""
@kwdef struct IterSVD
    alg::KrylovKit.GKL = KrylovKit.GKL(; tol = 1.0e-14, krylovdim = 25)
    fallback_threshold::Float64 = Inf
    start_vector = deterministic_start_vector
end
_default_svd_rrule_alg(::IterSVD) = :trunc

random_start_vector(t::AbstractMatrix) = randn(scalartype(t), size(t, 1))
deterministic_start_vector(t::AbstractMatrix) = ones(scalartype(t), size(t, 1))

# Compute SVD data block-wise using KrylovKit algorithm
# TODO: redefine _empty_svdtensors, _create_svdtensors
function MatrixAlgebraKit.svd_trunc!(f, alg::TruncatedAlgorithm{<:IterSVD})
    fwd_alg = alg.alg
    trunc = alg.trunc
    U, S, V = if isempty(blocksectors(f))
        # early return
        truncation_error = zero(real(scalartype(f)))
        MatrixAlgebraKit.initialize_output(svd_compact!, f, QRIteration()) # specified algorithm doesn't matter here
    else
        SVDdata, dims = _compute_svddata!(f, fwd_alg, trunc)
        _create_svdtensors(f, SVDdata, dims)
    end

    truncation_error =
        trunc isa NoTruncation ? abs(zero(scalartype(f))) : norm(U * S * V - f)

    return U, S, V, truncation_error
end

# Copy from TensorKit v0.14 internal functions
function _create_svdtensors(t::TensorMap, SVDdata, dims)
    T = scalartype(t)
    S = spacetype(t)
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    Σ = DiagonalTensorMap{Tr, S, A}(undef, W)

    U = similar(t, codomain(t) ← W)
    V⁺ = similar(t, W ← domain(t))
    for (c, (Uc, Σc, V⁺c)) in SVDdata
        r = Base.OneTo(dims[c])
        copy!(block(U, c), view(Uc, :, r))
        copy!(block(Σ, c), Diagonal(view(Σc, r)))
        copy!(block(V⁺, c), view(V⁺c, r, :))
    end
    return U, Σ, V⁺
end

# Interface between KrylovKit's svdsolve and TensorMap data
function _compute_svddata!(
        f, alg::IterSVD, trunc::Union{NoTruncation, TruncationSpace}
    )
    InnerProductStyle(f) === EuclideanInnerProduct() || throw_invalid_innerproduct(:full!)
    I = sectortype(f)
    dims = SectorDict{I, Int}()

    sectors = trunc isa NoTruncation ? blocksectors(f) : blocksectors(trunc.space)
    generator = Base.Iterators.map(sectors) do c
        b = block(f, c)
        howmany = trunc isa NoTruncation ? minimum(size(b)) : blockdim(trunc.space, c)

        if howmany / minimum(size(b)) > alg.fallback_threshold  # Use dense SVD for small blocks
            U, S, V = svd_compact!(b; alg = Defaults.svd_fwd_alg)
            S = S.diag # extracts diagonal as Vector instead of Diagonal to make compatible with S of svdsolve
            U = U[:, 1:howmany]
            V = V[1:howmany, :]
        else
            x₀ = alg.start_vector(b)
            svd_alg = alg.alg
            if howmany > alg.alg.krylovdim
                svd_alg = @set svd_alg.krylovdim = round(Int, howmany * 1.2)
            end
            S, lvecs, rvecs, info = svdsolve(b, x₀, howmany, :LR, svd_alg)
            if info.converged < howmany  # Fall back to dense SVD if not properly converged
                @warn "Iterative SVD did not converge for block $c, falling back to dense SVD"
                U, S, V = svd_compact!(b; alg = Defaults.svd_fwd_alg)
                S = S.diag
                U = U[:, 1:howmany]
                V = V[1:howmany, :]
            else  # Slice in case more values were converged than requested
                U = stack(view(lvecs, 1:howmany))
                V = stack(conj, view(rvecs, 1:howmany); dims = 1)
            end
        end

        # make it deterministic-ish
        MatrixAlgebraKit.gaugefix!(svd_trunc!, U, V)

        resize!(S, howmany)
        dims[c] = length(S)
        return c => (U, S, V)
    end

    SVDdata = SectorDict(generator)
    return SVDdata, dims
end

#
## Reverse-rule algorithms
#

# svd_trunc! rrule wrapping MatrixAlgebraKit's svd_pullback!
# https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/b76c7bb60014ecfead6925d0df6cb4b8d7c2668a/src/pullbacks/svd.jl#L33
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        t::AbstractTensorMap,
        alg::SVDAdjoint{F, R}
    ) where {F <: MatrixAlgebraKit.Algorithm, R <: FullSVDPullback}
    # TODO: filter out any decomposition algorithm that doesn't give access to the full spectrum

    # requires access to the full decomposition
    U, S, V⁺ = svd_compact!(t, alg.fwd_alg)
    (Ũ, S̃, Ṽ⁺), inds = truncate(svd_trunc!, (U, S, V⁺), alg.trunc)
    truncerror = truncation_error(diagview(S), inds)

    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function svd_trunc!_full_pullback(ΔUSV′)
        ΔUSV = unthunk.(ΔUSV′)
        Δt = svd_pullback!(
            zeros(scalartype(t), space(t)), t, (U, S, V⁺), ΔUSV, inds;
            gauge_atol = gtol(ΔUSV), degeneracy_atol = alg.rrule_alg.degeneracy_atol,
        )
        return NoTangent(), Δt, NoTangent()
    end
    function svd_trunc!_full_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (Ũ, S̃, Ṽ⁺, truncerror), svd_trunc!_full_pullback
end

# svd_trunc! rrule wrapping MatrixAlgebraKit's svd_trunc_pullback! (also works for IterSVD)
# https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/b76c7bb60014ecfead6925d0df6cb4b8d7c2668a/src/pullbacks/svd.jl#L143
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        t,
        alg::SVDAdjoint{F, R},
    ) where {F, R <: TruncSVDPullback}
    U, S, V⁺, ϵ = svd_trunc(t, alg)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function svd_trunc!_trunc_pullback(ΔUSV′)
        ΔUSV = unthunk.(ΔUSV′)
        Δf = svd_trunc_pullback!(
            zeros(scalartype(t), space(t)), t, (U, S, V⁺), ΔUSV;
            gauge_atol = gtol(ΔUSV), degeneracy_atol = alg.rrule_alg.degeneracy_atol,
        )
        return NoTangent(), Δf, NoTangent()
    end
    function svd_trunc!_trunc_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V⁺, ϵ), svd_trunc!_trunc_pullback
end

# KrylovKit rrule compatible with TensorMaps & function handles
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        f,
        alg::SVDAdjoint{F, R}
    ) where {F, R <: Union{GMRES, BiCGStab, Arnoldi}}
    U, S, V, ϵ = svd_trunc(f, alg)

    # update rrule_alg tolerance to be compatible with smallest singular value
    rrule_alg = alg.rrule_alg
    smallest_sval = minimum(((_, b),) -> minimum(diag(b)), blocks(S))
    proper_tol = clamp(rrule_alg.tol, eps(scalartype(S))^(3 / 4), 1.0e-2 * smallest_sval)
    rrule_alg = @set rrule_alg.tol = proper_tol

    function svd_trunc!_itersvd_pullback(ΔUSVi)
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
            minimal_alg = GKL(; tol = rrule_alg.tol, verbosity = 1)  # Tolerance is used for gauge sensitivity, verbosity is used for warnings

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
                rrule_alg,
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
    function svd_trunc!_itersvd_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V, ϵ), svd_trunc!_itersvd_pullback
end
