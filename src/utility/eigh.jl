using MatrixAlgebraKit: TruncationStrategy, NoTruncation, LAPACK_EighAlgorithm, truncate
using MatrixAlgebraKit: eigh_pullback!, eigh_trunc_pullback!, findtruncated, diagview
using TensorKit: AdjointTensorMap, SectorDict, Factorizations.TruncationSpace,
    throw_invalid_innerproduct, similarstoragetype
using KrylovKit: Lanczos, BlockLanczos
const KrylovKitCRCExt = Base.get_extension(KrylovKit, :KrylovKitChainRulesCoreExt)

"""
$(TYPEDEF)

Eigh reverse-rule algorithm which wraps MatrixAlgebraKit's `eigh_pullback!`.

## Fields

$(TYPEDFIELDS)

## Constructors

    FullEighPullback(; kwargs...)

Construct a `FullEighPullback` algorithm struct from the following keyword arguments:

* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct FullEighPullback
    verbosity::Int = 1
end

"""
$(TYPEDEF)

Truncated eigh reverse-rule algorithm which wraps MatrixAlgebraKit's `eigh_trunc_pullback!`.

## Fields

$(TYPEDFIELDS)

## Constructors

    TruncEighPullback(; kwargs...)

Construct a `TruncEighPullback` algorithm struct from the following keyword arguments:

* `verbosity::Int=0` : Suppresses all output if `≤0`, prints gauge dependency warnings if `1`, and always prints gauge dependency if `≥2`.
"""
@kwdef struct TruncEighPullback
    verbosity::Int = 1
end

# TODO: should this be same struct as SVDAdjoint?
"""
$(TYPEDEF)

Wrapper for a eigenvalue decomposition algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
If `isnothing(rrule_alg)`, Zygote differentiates the forward call automatically.

## Fields

$(TYPEDFIELDS)

## Constructors

    EighAdjoint(; kwargs...)

Construct a `EighAdjoint` algorithm struct based on the following keyword arguments:

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.eigh_fwd_alg))`: Eig algorithm of the forward pass which can either be passed as an `Algorithm` instance or a `NamedTuple` where `alg` is one of the following:
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.eigh_rrule_alg))`: Reverse-rule algorithm for differentiating the eigenvalue decomposition. Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
"""
struct EighAdjoint{F, R}
    fwd_alg::F
    rrule_alg::R
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

const EIGH_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :qriteration => LAPACK_QRIteration,
    :bisection => LAPACK_Bisection,
    :divideandconquer => LAPACK_DivideAndConquer,
    :multiple => LAPACK_MultipleRelativelyRobustRepresentations,
    :lanczos =>
        (; tol = 1.0e-14, krylovdim = 30, kwargs...) ->
    IterEig(; alg = Lanczos(; tol, krylovdim), kwargs...),
    :blocklanczos =>
        (; tol = 1.0e-14, krylovdim = 30, kwargs...) ->
    IterEig(; alg = BlockLanczos(; tol, krylovdim), kwargs...),
)
const EIGH_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :full => FullEighPullback, :trunc => TruncEighPullback,
)

function EighAdjoint(; fwd_alg = (;), rrule_alg = (;))
    # parse forward algorithm
    fwd_algorithm = if fwd_alg isa NamedTuple
        fwd_kwargs = (; alg = Defaults.eigh_fwd_alg, fwd_alg...) # overwrite with specified kwargs
        haskey(EIGH_FWD_SYMBOLS, fwd_kwargs.alg) ||
            throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
        fwd_type = EIGH_FWD_SYMBOLS[fwd_kwargs.alg]
        fwd_kwargs = Base.structdiff(fwd_kwargs, (; alg = nothing)) # remove `alg` keyword argument
        fwd_type(; fwd_kwargs...)
    else
        fwd_alg
    end

    # parse reverse-rule algorithm
    rrule_algorithm = if rrule_alg isa NamedTuple
        rrule_kwargs = (;
            alg = Defaults.eigh_rrule_alg,
            verbosity = Defaults.eigh_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(EIGH_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = EIGH_RRULE_SYMBOLS[rrule_kwargs.alg]
        if rrule_type <: Union{FullEighPullback, TruncEighPullback}
            rrule_kwargs = (; rrule_kwargs.verbosity)
        end

        rrule_type(; rrule_kwargs...)
    else
        rrule_alg
    end

    return EighAdjoint(fwd_algorithm, rrule_algorithm)
end

"""
    eigh_trunc(t, alg::EighAdjoint; trunc=notrunc())
    eigh_trunc!(t, alg::EighAdjoint; trunc=notrunc())

Wrapper around `eigh_trunc(!)` which dispatches on the `EighAdjoint` algorithm.
This is needed since a custom adjoint may be defined, depending on the `alg`.
"""
MatrixAlgebraKit.eigh_trunc(t, alg::EighAdjoint; kwargs...) = eigh_trunc!(copy(t), alg; kwargs...)
function MatrixAlgebraKit.eigh_trunc!(t, alg::EighAdjoint; trunc = notrunc())
    return _eigh_trunc!(t, alg.fwd_alg, trunc)
end
function MatrixAlgebraKit.eigh_trunc!(
        t::AdjointTensorMap, alg::EighAdjoint; trunc = notrunc()
    )
    D, U, info = eigh_trunc!(adjoint(t), alg; trunc)
    return adjoint(D), adjoint(U), info
end

## Forward algorithms

# Truncated eigh but also return full D and U to make it compatible with :fixed mode
function _eigh_trunc!(
        t::TensorMap,
        alg::LAPACK_EighAlgorithm,
        trunc::TruncationStrategy,
    )
    D, U = eigh_full!(t; alg)
    D̃, Ũ, truncerror = _truncate_eigh((D, U), trunc)

    # construct info NamedTuple
    condnum = cond(D)
    info = (;
        truncation_error = truncerror, condition_number = condnum, D_full = D, U_full = U,
    )
    return D̃, Ũ, info
end

# hacky way of computing the truncation error for current version of eigh_trunc!
# TODO: replace once TensorKit updates to new MatrixAlgebraKit which returns truncation error as well
function _truncate_eigh((D, U), trunc::TruncationStrategy)
    if !(trunc isa NoTruncation) && !isempty(blocksectors(D))
        D̃, Ũ = truncate(eigh_trunc!, (D, U), trunc)[1]
        truncerror = sqrt(abs(norm(D)^2 - norm(D̃)^2))
        return D̃, Ũ, truncerror
    else
        return D, U, zero(real(scalartype(D)))
    end
end

"""
$(TYPEDEF)

Eigenvalue decomposition struct containing a pre-computed decomposition or even multiple ones.
Additionally, it can contain the untruncated full decomposition as well. The call to
`eigh_trunc`/`eig_trunc` just returns the pre-computed D and U. In the reverse pass,
the adjoint is computed with these exact D and U and, potentially, the full decompositions
if the adjoints needs access to them.

## Fields

$(TYPEDFIELDS)
"""
struct FixedEig{Dt, Ut, Dtf, Utf}
    D::Dt
    U::Ut
    D_full::Dtf
    U_full::Utf
end

# check whether the full D and U are supplied
function isfulleig(alg::FixedEig)
    if isnothing(alg.D_full) || isnothing(alg.U_full)
        return false
    else
        return true
    end
end

# Return pre-computed decomposition
function _eigh_trunc!(_, alg::FixedEig, ::TruncationStrategy)
    info = (;
        truncation_error = zero(real(scalartype(alg.D))),
        condition_number = cond(alg.D),
        D_full = alg.D_full,
        U_full = alg.U_full,
    )
    return alg.D, alg.U, info
end


"""
$(TYPEDEF)

Iterative eigenvalue solver based on KrylovKit's `eigsolve`, adapted to (symmetric) tensors.
The number of targeted eigenvalues is set via the `truncspace` in `ProjectorAlg`.
In particular, this makes it possible to specify the targeted eigenvalues block-wise.
In case the symmetry block is too small as compared to the number of singular values, or
the iterative decomposition didn't converge, the algorithm falls back to a dense `eigh`/`eigh`.

## Fields

$(TYPEDFIELDS)

## Constructors

    IterEig(; kwargs...)

Construct an `IterEig` algorithm struct based on the following keyword arguments:

* `alg=KrylovKit.Lanczos(; tol=1e-14, krylovdim=25)` : KrylovKit algorithm struct for iterative eigenvalue decomposition.
* `fallback_threshold::Float64=Inf` : Threshold for `howmany / minimum(size(block))` above which (if the block is too small) the algorithm falls back to a dense decomposition.
* `start_vector=random_start_vector` : Function providing the initial vector for the iterative algorithm.
"""
@kwdef struct IterEig
    alg = KrylovKit.Lanczos(; tol = 1.0e-14, krylovdim = 25)
    fallback_threshold::Float64 = Inf
    start_vector = random_start_vector
end

# Compute eigh data block-wise using KrylovKit algorithm
function _eigh_trunc!(f, alg::IterEig, trunc::TruncationStrategy)
    D, U = if isempty(blocksectors(f))
        # early return
        truncation_error = zero(real(scalartype(f)))
        MatrixAlgebraKit.initialize_output(eigh_full!, f, LAPACK_QRIteration()) # specified algorithm doesn't matter here
    else
        eighdata, dims = _compute_eighdata!(f, alg, trunc)
        _create_eightensors(f, eighdata, dims)
    end

    # construct info NamedTuple
    truncation_error =
        trunc isa NoTruncation ? abs(zero(scalartype(f))) : norm(U * D * U' - f)
    condition_number = cond(D)
    info = (; truncation_error, condition_number, D_full = nothing, U_full = nothing)

    return D, U, info
end

# Obtain sparse decomposition from block-wise eigsolve calls
function _compute_eighdata!(
        f, alg::IterEig, trunc::Union{NoTruncation, TruncationSpace}
    )
    InnerProductStyle(f) === EuclideanInnerProduct() || throw_invalid_innerproduct(:eigh_trunc!)
    domain(f) == codomain(f) ||
        throw(SpaceMismatch("`eigh!` requires domain and codomain to be the same"))
    I = sectortype(f)
    dims = SectorDict{I, Int}()

    sectors = trunc isa NoTruncation ? blocksectors(f) : blocksectors(trunc.space)
    generator = Base.Iterators.map(sectors) do c
        b = block(f, c)
        howmany = trunc isa NoTruncation ? minimum(size(b)) : blockdim(trunc.space, c)

        if howmany / minimum(size(b)) > alg.fallback_threshold  # Use dense decomposition for small blocks
            D, U = eigh_full!(b, LAPACK_QRIteration())
            lm_ordering = sortperm(abs.(D.diag); rev = true) # order values and vectors consistently with eigsolve
            D = D.diag[lm_ordering] # extracts diagonal as Vector instead of Diagonal to make compatible with D of svdsolve
            U = view(U, lm_ordering)[:, 1:howmany]
        else
            x₀ = alg.start_vector(b)
            eig_alg = alg.alg
            if howmany > alg.alg.krylovdim
                eig_alg = @set eig_alg.krylovdim = round(Int, howmany * 1.2)
            end
            D, lvecs, info = eigsolve(b, x₀, howmany, :LM, eig_alg)
            if info.converged < howmany  # Fall back to dense SVD if not properly converged
                @warn "Iterative eigendecomposition did not converge for block $c, falling back to eigh_full"
                D, U = eigh_full!(b, LAPACK_QRIteration())
                lm_ordering = sortperm(abs.(D.diag); rev = true)
                D = D.diag[lm_ordering]
                U = view(U, lm_ordering)[:, 1:howmany]
            else  # Slice in case more values were converged than requested
                U = stack(view(lvecs, 1:howmany))
            end
        end

        resize!(D, howmany)
        dims[c] = length(D)
        return c => (D, U)
    end

    eigdata = SectorDict(generator)
    return eigdata, dims
end

# Create eigh TensorMaps from sparse SectorDict
function _create_eightensors(t::TensorMap, eighdata, dims)
    InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:eigh!)

    T = scalartype(t)
    S = spacetype(t)
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    D = DiagonalTensorMap{Tr, S, A}(undef, W)
    U = similar(t, domain(t) ← W)
    for (c, (Dc, Uc)) in eighdata
        r = Base.OneTo(dims[c])
        copy!(block(D, c), Diagonal(view(Dc, r)))
        copy!(block(U, c), view(Uc, :, r))
    end
    return D, U
end

## Reverse-rule algorithms
function _get_pullback_gauge_tol(verbosity::Int)
    if verbosity ≤ 0 # never print gauge sensitivity
        return (_) -> Inf
    elseif verbosity == 1 # print gauge sensitivity above default atol
        MatrixAlgebraKit.default_pullback_gaugetol
    else # always print gauge sensitivity
        return (_) -> 0.0
    end
end

# eigh_trunc! rrule wrapping MatrixAlgebraKit's eigh_pullback!
function ChainRulesCore.rrule(
        ::typeof(eigh_trunc!),
        t::AbstractTensorMap,
        alg::EighAdjoint{F, R};
        trunc::TruncationStrategy = notrunc(),
    ) where {F <: Union{<:LAPACK_EighAlgorithm, <:FixedEig}, R <: FullEighPullback}
    D̃, Ũ, info = eigh_trunc(t, alg; trunc)
    D, U = info.D_full, info.U_full # untruncated decomposition
    inds = findtruncated(diagview(D), truncspace(only(domain(D̃))))
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function eigh_trunc!_full_pullback(ΔDU)
        Δt = eigh_pullback!( # TODO: does this work by now?
            zeros(scalartype(t), space(t)), t, (D, U), ΔDU, inds;
            gauge_atol = gtol(ΔDU)
        )
        return NoTangent(), Δt, NoTangent()
    end
    function eigh_trunc!_full_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (D̃, Ũ, info), eigh_trunc!_full_pullback
end

# eigh_trunc! rrule wrapping MatrixAlgebraKit's eigh_trunc_pullback! (also works for IterEig)
function ChainRulesCore.rrule(
        ::typeof(eigh_trunc!),
        t,
        alg::EighAdjoint{F, R};
        trunc::TruncationStrategy = notrunc(),
    ) where {F <: Union{<:LAPACK_EighAlgorithm, <:FixedEig, IterEig}, R <: TruncEighPullback}
    D, U, info = eigh_trunc(t, alg; trunc)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function eigh_trunc!_trunc_pullback(ΔDU)
        Δf = eigh_trunc_pullback!(
            zeros(scalartype(t), space(t)), t, (D, U), ΔDU;
            gauge_atol = gtol(ΔDU)
        )
        return NoTangent(), Δf, NoTangent()
    end
    function eigh_trunc!_trunc_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (D, U, info), eigh_trunc!_trunc_pullback
end
