const KrylovKitCRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
$(TYPEDEF)

SVD reverse-rule algorithm which wraps MatrixAlgebraKit's `svd_pullback!`.

## Fields

$(TYPEDFIELDS)

## Constructors

    FullSVDPullback(; kwargs...)

Construct a `FullSVDPullback` algorithm struct from the following keyword arguments:

* `degeneracy_tol::Real=$(Defaults.rrule_degeneracy_tol)` : Broadening amplitude for smoothing divergent term in SVD derivative in case of (pseudo) degenerate singular values.
* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct FullSVDPullback
    degeneracy_tol::Real = Defaults.rrule_degeneracy_tol
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

* `degeneracy_tol::Real=$(Defaults.rrule_degeneracy_tol)` : Broadening amplitude for smoothing divergent term in SVD derivative in case of (pseudo) degenerate singular values.
* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct TruncSVDPullback
    degeneracy_tol::Real = Defaults.rrule_degeneracy_tol
    verbosity::Int = 0
end

"""
$(TYPEDEF)

Wrapper for a SVD algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
If `isnothing(rrule_alg)`, Zygote differentiates the forward call automatically.

## Fields

$(TYPEDFIELDS)

## Constructors

    SVDAdjoint(; kwargs...)

Construct a `SVDAdjoint` algorithm struct based on the following keyword arguments:

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.svd_fwd_alg))`: SVD algorithm of the forward pass which can either be passed as an `Algorithm` instance or a `NamedTuple` where `alg` is one of the following:
    - `:divideandconquer` : MatrixAlgebraKit's `LAPACK_DivideAndConquer`
    - `:qriteration` : MatrixAlgebraKit's `LAPACK_QRIteration`
    - `:bisection` : MatrixAlgebraKit's `LAPACK_Bisection`
    - `:jacobi` : MatrixAlgebraKit's `LAPACK_Jacobi`
    - `:gkl` : Iterative SVD only computing the specifed number of singular values and vectors, see [`IterSVD`](@ref)
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.svd_rrule_alg))`: Reverse-rule algorithm for differentiating the SVD. Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
    - `:full` : MatrixAlgebraKit's `svd_pullback!` that requires access to the full spectrum
    - `:trunc` : MatrixAlgebraKit's `svd_trunc_pullback!` solving a Sylvester equation on the truncated subspace
    - `:gmres` : GMRES iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.GMRES) for details
    - `:bicgstab` : BiCGStab iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BiCGStab) for details
    - `:arnoldi` : Arnoldi Krylov algorithm, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Arnoldi) for details
"""
struct SVDAdjoint{F, R}
    fwd_alg::F
    rrule_alg::R
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

const SVD_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :divideandconquer => LAPACK_DivideAndConquer,
    :qriteration => LAPACK_QRIteration,
    :bisection => LAPACK_Bisection,
    :jacobi => LAPACK_Jacobi,
    :gkl => (; tol = 1.0e-14, krylovdim = 25, kwargs...) -> IterSVD(; alg = GKL(; tol, krylovdim), kwargs...),
)
const SVD_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :full => FullSVDPullback, :trunc => TruncSVDPullback,
    :gmres => GMRES, :bicgstab => BiCGStab, :arnoldi => Arnoldi
)

function SVDAdjoint(; fwd_alg = (;), rrule_alg = (;))
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
            alg = Defaults.svd_rrule_alg,
            tol = Defaults.svd_rrule_tol,
            krylovdim = Defaults.svd_rrule_min_krylovdim,
            degeneracy_tol = Defaults.rrule_degeneracy_tol,
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
            rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg = nothing, degeneracy_tol = 0.0)) # remove `alg` and `degeneracy_tol` keyword arguments
            rrule_type <: BiCGStab &&
                (rrule_kwargs = Base.structdiff(rrule_kwargs, (; krylovdim = nothing))) # BiCGStab doesn't take `krylovdim`
        end
        rrule_type(; rrule_kwargs...)
    else
        rrule_alg
    end

    return SVDAdjoint(fwd_algorithm, rrule_algorithm)
end

"""
    svd_trunc(t, alg::SVDAdjoint; trunc=notrunc())
    svd_trunc!(t, alg::SVDAdjoint; trunc=notrunc())

Wrapper around `svd_trunc(!)` which dispatches on the `SVDAdjoint` algorithm.
This is needed since a custom adjoint may be defined, depending on the `alg`.
E.g., for `IterSVD` the adjoint for a truncated SVD from `KrylovKit.svdsolve` is used.
"""
MatrixAlgebraKit.svd_trunc(t, alg::SVDAdjoint; kwargs...) = svd_trunc!(copy(t), alg; kwargs...)
function MatrixAlgebraKit.svd_trunc!(t, alg::SVDAdjoint; trunc = notrunc())
    return _svd_trunc!(t, alg.fwd_alg, trunc)
end
function MatrixAlgebraKit.svd_trunc!(
        t::AdjointTensorMap, alg::SVDAdjoint; trunc = notrunc()
    )
    u, s, vt, info = svd_trunc!(adjoint(t), alg; trunc)
    return adjoint(vt), adjoint(s), adjoint(u), info
end

#
## Forward algorithms
#

# Truncated SVD but also return full U, S and V to make it compatible with :fixed mode
function _svd_trunc!(
        t::TensorMap,
        alg::Union{LAPACK_DivideAndConquer, LAPACK_QRIteration},
        trunc::TruncationStrategy,
    )
    U, S, V⁺ = svd_compact!(t; alg)
    (Ũ, S̃, Ṽ⁺), ind = truncate(svd_trunc!, (U, S, V⁺), trunc)
    truncerror = truncation_error(diagview(S), ind)

    # construct info NamedTuple
    condnum = cond(S)
    info = (;
        truncation_error = truncerror, condition_number = condnum,
        U_full = U, S_full = S, V_full = V⁺,
        truncation_indices = ind,
    )
    return Ũ, S̃, Ṽ⁺, info
end

"""
$(TYPEDEF)

SVD struct containing a pre-computed decomposition or even multiple ones. Additionally, it
can contain the untruncated full decomposition as well. The call to `svd_trunc` just returns the
pre-computed U, S and V. In the reverse pass, the SVD adjoint is computed with these exact
U, S, and V and, potentially, the full decompositions if the adjoints needs access to them.

## Fields

$(TYPEDFIELDS)
"""
struct FixedSVD{Ut, St, Vt, Utf, Stf, Vtf, It}
    U::Ut
    S::St
    V::Vt
    U_full::Utf
    S_full::Stf
    V_full::Vtf
    truncation_indices::It
end

# check whether the full U, S and V are supplied
function isfullsvd(alg::FixedSVD)
    if isnothing(alg.U_full) || isnothing(alg.S_full) || isnothing(alg.V_full) || isnothing(alg.truncation_indices)
        return false
    else
        return true
    end
end

# Return pre-computed SVD
function _svd_trunc!(_, alg::FixedSVD, ::TruncationStrategy)
    info = (;
        truncation_error = zero(real(scalartype(alg.S))),
        condition_number = cond(alg.S),
        U_full = alg.U_full,
        S_full = alg.S_full,
        V_full = alg.V_full,
        truncation_indices = alg.truncation_indices,
    )
    return alg.U, alg.S, alg.V, info
end

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
* `start_vector=random_start_vector` : Function providing the initial vector for the iterative SVD algorithm.
"""
@kwdef struct IterSVD
    alg::KrylovKit.GKL = KrylovKit.GKL(; tol = 1.0e-14, krylovdim = 25)
    fallback_threshold::Float64 = Inf
    start_vector = random_start_vector
end

function random_start_vector(t::AbstractMatrix)
    return randn(scalartype(t), size(t, 1))
end

# Compute SVD data block-wise using KrylovKit algorithm
# TODO: redefine _empty_svdtensors, _create_svdtensors
function _svd_trunc!(f, alg::IterSVD, trunc::TruncationStrategy)
    U, S, V = if isempty(blocksectors(f))
        # early return
        truncation_error = zero(real(scalartype(f)))
        MatrixAlgebraKit.initialize_output(svd_compact!, f, LAPACK_QRIteration()) # specified algorithm doesn't matter here
    else
        SVDdata, dims = _compute_svddata!(f, alg, trunc)
        _create_svdtensors(f, SVDdata, dims)
    end

    # construct info NamedTuple
    truncation_error =
        trunc isa NoTruncation ? abs(zero(scalartype(f))) : norm(U * S * V - f)
    condition_number = cond(S)
    info = (;
        truncation_error, condition_number, U_full = nothing, S_full = nothing, V_full = nothing,
    )

    return U, S, V, info
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
            U, S, V = svd_compact!(b, LAPACK_DivideAndConquer())
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
                U, S, V = svd_compact!(b, LAPACK_DivideAndConquer())
                S = S.diag
                U = U[:, 1:howmany]
                V = V[1:howmany, :]
            else  # Slice in case more values were converged than requested
                U = stack(view(lvecs, 1:howmany))
                V = stack(conj, view(rvecs, 1:howmany); dims = 1)
            end
        end

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
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        t::AbstractTensorMap,
        alg::SVDAdjoint{F, R};
        trunc::TruncationStrategy = notrunc(),
    ) where {F, R <: FullSVDPullback}
    @assert !(alg.fwd_alg isa IterSVD) "IterSVD is not compatible with FullSVDPullback"

    Ũ, S̃, Ṽ⁺, info = svd_trunc(t, alg; trunc)
    U, S, V⁺, inds = info.U_full, info.S_full, info.V_full, info.truncation_indices # untruncated decomposition
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function svd_trunc!_full_pullback(ΔUSV′)
        ΔUSV = unthunk.(ΔUSV′)
        Δt = svd_pullback!(
            zeros(scalartype(t), space(t)), t, (U, S, V⁺), ΔUSV, inds;
            gauge_atol = gtol(ΔUSV), degeneracy_atol = alg.rrule_alg.degeneracy_tol,
        )
        return NoTangent(), Δt, NoTangent()
    end
    function svd_trunc!_full_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (Ũ, S̃, Ṽ⁺, info), svd_trunc!_full_pullback
end

# svd_trunc! rrule wrapping MatrixAlgebraKit's svd_trunc_pullback! (also works for IterSVD)
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        t,
        alg::SVDAdjoint{F, R};
        trunc::TruncationStrategy = notrunc(),
    ) where {F, R <: TruncSVDPullback}
    U, S, V⁺, info = svd_trunc(t, alg; trunc)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function svd_trunc!_trunc_pullback(ΔUSV′)
        ΔUSV = unthunk.(ΔUSV′)
        Δf = svd_trunc_pullback!(
            zeros(scalartype(t), space(t)), t, (U, S, V⁺), ΔUSV;
            gauge_atol = gtol(ΔUSV), degeneracy_atol = alg.rrule_alg.degeneracy_tol,
        )
        return NoTangent(), Δf, NoTangent()
    end
    function svd_trunc!_trunc_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (U, S, V⁺, info), svd_trunc!_trunc_pullback
end

# KrylovKit rrule compatible with TensorMaps & function handles
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        f,
        alg::SVDAdjoint{F, R};
        trunc::TruncationStrategy = notrunc(),
    ) where {F, R <: Union{GMRES, BiCGStab, Arnoldi}}
    U, S, V, info = svd_trunc(f, alg; trunc)

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

    return (U, S, V, info), svd_trunc!_itersvd_pullback
end
