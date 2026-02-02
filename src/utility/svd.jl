const KrylovKitCRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
$(TYPEDEF)

SVD reverse-rule algorithm which uses a modified version of TensorKit's `tsvd!` reverse-rule
allowing for Lorentzian broadening and output verbosity control.

## Fields

$(TYPEDFIELDS)

## Constructors

    FullSVDReverseRule(; kwargs...)

Construct a `FullSVDReverseRule` algorithm struct from the following keyword arguments:

* `broadening::Float64=$(Defaults.svd_rrule_broadening)` : Lorentzian broadening amplitude for smoothing divergent term in SVD derivative in case of (pseudo) degenerate singular values.
* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct FullSVDReverseRule
    broadening::Float64 = Defaults.svd_rrule_broadening
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
    - `:sdd` : MatrixAlgebraKit's `LAPACK_DivideAndConquer`
    - `:svd` : MatrixAlgebraKit's `LAPACK_QRIteration`
    - `:iterative` : Iterative SVD only computing the specifed number of singular values and vectors, see [`IterSVD`](@ref)
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.svd_rrule_alg))`: Reverse-rule algorithm for differentiating the SVD. Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
    - `:full` : Uses a modified version of TensorKit's reverse-rule for `tsvd` which doesn't solve any linear problem and instead requires access to the full SVD, see [`FullSVDReverseRule`](@ref).
    - `:gmres` : GMRES iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.GMRES) for details
    - `:bicgstab` : BiCGStab iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BiCGStab) for details
    - `:arnoldi` : Arnoldi Krylov algorithm, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Arnoldi) for details
"""
struct SVDAdjoint{F, R}
    fwd_alg::F
    rrule_alg::R
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

const SVD_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :sdd => LAPACK_DivideAndConquer,
    :svd => LAPACK_QRIteration,
    :iterative =>
        (; tol = 1.0e-14, krylovdim = 25, kwargs...) ->
    IterSVD(; alg = GKL(; tol, krylovdim), kwargs...),
)
const SVD_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :full => FullSVDReverseRule, :gmres => GMRES, :bicgstab => BiCGStab, :arnoldi => Arnoldi
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
            broadening = Defaults.svd_rrule_broadening,
            verbosity = Defaults.svd_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(SVD_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = SVD_RRULE_SYMBOLS[rrule_kwargs.alg]

        # IterSVD is incompatible with tsvd rrule -> default to Arnoldi
        if rrule_type <: FullSVDReverseRule && fwd_algorithm isa IterSVD
            rrule_type = Arnoldi
        end

        if rrule_type <: FullSVDReverseRule
            rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg = nothing, tol = 0.0, krylovdim = 0)) # remove `alg`, `tol` and `krylovdim` keyword arguments
        else
            rrule_kwargs = Base.structdiff(rrule_kwargs, (; alg = nothing, broadening = 0.0)) # remove `alg` and `broadening` keyword arguments
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

## Forward algorithms

# Truncated SVD but also return full U, S and V to make it compatible with :fixed mode
function _svd_trunc!(
        t::TensorMap,
        alg::Union{LAPACK_DivideAndConquer, LAPACK_QRIteration},
        trunc::TruncationStrategy,
    )
    U, S, V⁺ = svd_compact!(t; alg)
    Ũ, S̃, Ṽ⁺, truncerror = _truncate_compact((U, S, V⁺), trunc)

    # construct info NamedTuple
    condnum = cond(S)
    info = (;
        truncation_error = truncerror, condition_number = condnum, U_full = U, S_full = S, V_full = V⁺,
    )
    return Ũ, S̃, Ṽ⁺, info
end

# hacky way of computing the truncation error for current version of svd_trunc!
# TODO: replace once TensorKit updates to new MatrixAlgebraKit which returns truncation error as well
function _truncate_compact((U, S, V⁺), trunc::TruncationStrategy)
    if !(trunc isa NoTruncation) && !isempty(blocksectors(S))
        Ũ, S̃, Ṽ⁺ = truncate(svd_trunc!, (U, S, V⁺), trunc)[1]
        truncerror = sqrt(norm(S)^2 - norm(S̃)^2)
        return Ũ, S̃, Ṽ⁺, truncerror
    else
        return U, S, V⁺, zero(real(scalartype(S)))
    end
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
struct FixedSVD{Ut, St, Vt, Utf, Stf, Vtf}
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
function _svd_trunc!(_, alg::FixedSVD, ::TruncationStrategy)
    info = (;
        truncation_error = zero(real(scalartype(alg.S))),
        condition_number = cond(alg.S),
        U_full = alg.U_full,
        S_full = alg.S_full,
        V_full = alg.V_full,
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

## Reverse-rule algorithms

# TensorKit.tsvd! rrule with info NamedTuple return value
function ChainRulesCore.rrule(
        ::typeof(svd_trunc!),
        t::AbstractTensorMap,
        alg::SVDAdjoint{F, R};
        trunc::TruncationStrategy = notrunc(),
    ) where {F, R <: FullSVDReverseRule}
    @assert !(alg.fwd_alg isa IterSVD) "IterSVD is not compatible with full SVD reverse-rule"
    Ũ, S̃, Ṽ⁺, info = svd_trunc(t, alg; trunc)
    U, S, V⁺ = info.U_full, info.S_full, info.V_full # untruncated SVD decomposition

    smallest_sval = minimum(((_, b),) -> minimum(diag(b)), blocks(S̃))
    pullback_tol = clamp(
        smallest_sval, eps(scalartype(S̃))^(3 / 4), eps(scalartype(S̃))^(1 / 2)
    )

    function svd_trunc!_full_pullback(ΔUSVi)
        ΔU, ΔS, ΔV⁺, = unthunk.(ΔUSVi)
        Δt = similar(t)
        for (c, b) in blocks(Δt)
            Uc, Sc, V⁺c = block(U, c), block(S, c), block(V⁺, c)
            ΔUc, ΔSc, ΔV⁺c = block(ΔU, c), block(ΔS, c), block(ΔV⁺, c)
            Sdc = view(Sc, diagind(Sc))
            ΔSdc = (ΔSc isa AbstractZero) ? ΔSc : view(ΔSc, diagind(ΔSc))
            svd_pullback!(
                b,
                Uc,
                Sdc,
                V⁺c,
                ΔUc,
                ΔSdc,
                ΔV⁺c;
                tol = pullback_tol,
                broadening = alg.rrule_alg.broadening,
                verbosity = alg.rrule_alg.verbosity,
            )
        end
        return NoTangent(), Δt, NoTangent()
    end
    function svd_trunc!_full_pullback(::Tuple{ZeroTangent, ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (Ũ, S̃, Ṽ⁺, info), svd_trunc!_full_pullback
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

# scalar inverses with a cutoff tolerance
_safe_inv(x, tol) = abs(x) < tol ? zero(x) : inv(x)

# compute inverse singular value difference contribution to SVD gradient with broadening ε
function _broadened_inv_S(S::AbstractVector{T}, tol, ε = 0) where {T}
    F = similar(S, (axes(S, 1), axes(S, 1)))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        F[i, j] = if i == j
            zero(T)
        else
            Δsᵢⱼ = S[j] - S[i]
            ε > 0 ? _lorentz_broaden(Δsᵢⱼ, ε) : _safe_inv(Δsᵢⱼ, tol)
        end
    end
    return F
end

# Lorentzian broadening for divergent term in SVD rrule, see
# https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.013237
function _lorentz_broaden(x, ε = eps(real(scalartype(x)))^(3 / 4))
    return x / (x^2 + ε)
end

function _default_pullback_gaugetol(x)
    n = norm(x, Inf)
    return eps(eltype(n))^(3 / 4) * max(n, one(n))
end

# SVD_pullback: pullback implementation for general (possibly truncated) SVD
#
# This is a modified version of TensorKit's pullback
# https://github.com/Jutho/TensorKit.jl/blob/fa1551472ac74d7f2a61bdb2135cf418c8c53378/ext/TensorKitChainRulesCoreExt/factorizations.jl#L190)
# with support for Lorentzian broadening and improved verbosity control
#
# Arguments are U, S and Vd of full (non-truncated, but still thin) SVD, as well as
# cotangent ΔU, ΔS, ΔVd variables of truncated SVD
#
# Checks whether the cotangent variables are such that they would couple to gauge-dependent
# degrees of freedom (phases of singular vectors), and prints a warning if this is the case
#
# An implementation that only uses U, S, and Vd from truncated SVD is also possible, but
# requires solving a Sylvester equation, which does not seem to be supported on GPUs.
#
# Other implementation considerations for GPU compatibility:
# no scalar indexing, lots of broadcasting and views
#
function svd_pullback!(
        ΔA::AbstractMatrix,
        U::AbstractMatrix,
        S::AbstractVector,
        Vd::AbstractMatrix,
        ΔU,
        ΔS,
        ΔVd;
        tol::Real = _default_pullback_gaugetol(S),
        broadening::Real = 0,
        verbosity = 1,
    )

    # Basic size checks and determination
    m, n = size(U, 1), size(Vd, 2)
    size(U, 2) == size(Vd, 1) == length(S) == min(m, n) || throw(DimensionMismatch())
    p = -1
    if !(ΔU isa AbstractZero)
        m == size(ΔU, 1) || throw(DimensionMismatch())
        p = size(ΔU, 2)
    end
    if !(ΔVd isa AbstractZero)
        n == size(ΔVd, 2) || throw(DimensionMismatch())
        if p == -1
            p = size(ΔVd, 1)
        else
            p == size(ΔVd, 1) || throw(DimensionMismatch())
        end
    end
    if !(ΔS isa AbstractZero)
        if p == -1
            p = length(ΔS)
        else
            p == length(ΔS) || throw(DimensionMismatch())
        end
    end
    Up = view(U, :, 1:p)
    Vp = view(Vd, 1:p, :)'
    Sp = view(S, 1:p)

    # rank
    r = searchsortedlast(S, tol; rev = true)

    # compute antihermitian part of projection of ΔU and ΔV onto U and V
    # also already subtract this projection from ΔU and ΔV
    if !(ΔU isa AbstractZero)
        UΔU = Up' * ΔU
        aUΔU = rmul!(UΔU - UΔU', 1 / 2)
        if m > p
            ΔU -= Up * UΔU
        end
    else
        aUΔU = fill!(similar(U, (p, p)), 0)
    end
    if !(ΔVd isa AbstractZero)
        VΔV = Vp' * ΔVd'
        aVΔV = rmul!(VΔV - VΔV', 1 / 2)
        if n > p
            ΔVd -= VΔV' * Vp'
        end
    else
        aVΔV = fill!(similar(Vd, (p, p)), 0)
    end

    # check whether cotangents arise from gauge-invariance objective function
    mask = abs.(Sp' .- Sp) .< tol
    Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
    if p > r
        rprange = (r + 1):p
        Δgauge = max(Δgauge, norm(view(aUΔU, rprange, rprange), Inf))
        Δgauge = max(Δgauge, norm(view(aVΔV, rprange, rprange), Inf))
    end
    if verbosity == 1 && Δgauge > tol # warn if verbosity is 1
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    elseif verbosity ≥ 2 # always info for debugging purposes
        @info "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
    end

    inv_S_minus = _broadened_inv_S(Sp, tol, broadening) # possibly divergent/broadened contribution
    UdΔAV = @. (aUΔU + aVΔV) * inv_S_minus + (aUΔU - aVΔV) * _safe_inv(Sp' .+ Sp, tol)
    if !(ΔS isa ZeroTangent)
        UdΔAV[diagind(UdΔAV)] .+= real.(ΔS)
        # in principle, ΔS is real, but maybe not if coming from an anyonic tensor
    end
    mul!(ΔA, Up, UdΔAV * Vp')

    if r > p # contribution from truncation
        Ur = view(U, :, (p + 1):r)
        Vr = view(Vd, (p + 1):r, :)'
        Sr = view(S, (p + 1):r)

        if !(ΔU isa AbstractZero)
            UrΔU = Ur' * ΔU
            if m > r
                ΔU -= Ur * UrΔU # subtract this part from ΔU
            end
        else
            UrΔU = fill!(similar(U, (r - p, p)), 0)
        end
        if !(ΔVd isa AbstractZero)
            VrΔV = Vr' * ΔVd'
            if n > r
                ΔVd -= VrΔV' * Vr' # subtract this part from ΔV
            end
        else
            VrΔV = fill!(similar(Vd, (r - p, p)), 0)
        end

        X = @. (1 // 2) * (
            (UrΔU + VrΔV) * _safe_inv(Sp' - Sr, tol) +
                (UrΔU - VrΔV) * _safe_inv(Sp' + Sr, tol)
        )
        Y = @. (1 // 2) * (
            (UrΔU + VrΔV) * _safe_inv(Sp' - Sr, tol) -
                (UrΔU - VrΔV) * _safe_inv(Sp' + Sr, tol)
        )

        # ΔA += Ur * X * Vp' + Up * Y' * Vr'
        mul!(ΔA, Ur, X * Vp', 1, 1)
        mul!(ΔA, Up * Y', Vr', 1, 1)
    end

    if m > max(r, p) && !(ΔU isa AbstractZero) # remaining ΔU is already orthogonal to U[:,1:max(p,r)]
        # ΔA += (ΔU .* _safe_inv.(Sp', tol)) * Vp'
        mul!(ΔA, ΔU .* _safe_inv.(Sp', tol), Vp', 1, 1)
    end
    if n > max(r, p) && !(ΔVd isa AbstractZero) # remaining ΔV is already orthogonal to V[:,1:max(p,r)]
        # ΔA += U * (_safe_inv.(Sp, tol) .* ΔVd)
        mul!(ΔA, Up, _safe_inv.(Sp, tol) .* ΔVd, 1, 1)
    end
    return ΔA
end
