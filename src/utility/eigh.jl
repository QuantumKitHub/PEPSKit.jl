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
    degeneracy_atol::Real = Defaults.rrule_degeneracy_atol
    verbosity::Int = 0
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
    degeneracy_atol::Real = Defaults.rrule_degeneracy_atol
    verbosity::Int = 0
end

"""
$(TYPEDEF)

Wrapper for a eigenvalue decomposition algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.

## Fields

$(TYPEDFIELDS)

## Constructors

    EighAdjoint(; kwargs...)

Construct a `EighAdjoint` algorithm struct based on the following keyword arguments:

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.eigh_fwd_alg))`: Eigh
  algorithm of the forward pass which can either be passed as an `Algorithm` instance or a
  `NamedTuple` where the algorithm is specified by the `alg` keyword.
  The available Eigh algorithms can be divided into two categories:
    - "Dense" Eigh algorithms which compute a truncated Eigh through the truncation of a full
      [`MatrixAlgebraKit.eigh_full!`](@extref) decomposition.
      Available algorithms are:
        - `:DefaultAlgorithm` : MatrixAlgebraKit's [default Eigh algorithm](@extref MatrixAlgebraKit.DefaultAlgorithm) for a given matrix type.
        - `:DivideAndConquer` : MatrixAlgebraKit's [`DivideAndConquer`](@extref MatrixAlgebraKit.DivideAndConquer)
        - `:QRIteration` : MatrixAlgebraKit's [`QRIteration`](@extref MatrixAlgebraKit.QRIteration)
        - `:Bisection` : MatrixAlgebraKit's [`Bisection`](@extref MatrixAlgebraKit.Bisection)
        - `:Jacobi` : MatrixAlgebraKit's [`Jacobi`](@extref MatrixAlgebraKit.Jacobi)
        - `:RobustRepresentations` : MatrixAlgebraKit's [`RobustRepresentations`](@extref MatrixAlgebraKit.RobustRepresentations)
    - "Sparse" Eigh algorithms which directly compute a truncated Eigh without access to the
      full decomposition. Available algorithms are:
        - `:Lanczos` : Lanczos algorithm for symmetric/Hermitian matrices, see [`KrylovKit.Lanczos`](@extref)
        - `:BlockLanczos` : Block version of `:Lanczos` for repeated extremal eigenvalues, see [`KrylovKit.BlockLanczos`](@extref)
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:notrunc)` : Truncation strategy for the truncated eigh, which controls the spaces of the output. Here, `alg` can be one of the following:
    - `:notrunc` : No eigenvalues are truncated.
    - `:truncerror` : Additionally supply error threshold `η`; truncate such that the 2-norm of the truncated eigenvalues is smaller than `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate to the maximal virtual dimension of `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply eigenvalue magnitude cutoff `η`; truncate such that the magnitude of every retained eigenvalue is larger than `η`
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.eigh_rrule_alg))`:
  Reverse-rule algorithm for differentiating the eigenvalue decomposition. Can be supplied
  by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the
  following:
    - `:full` : MatrixAlgebraKit's [`eigh_pullback!`](@extref MatrixAlgebraKit.eigh_pullback!) that requires access to the full spectrum
    - `:trunc` : MatrixAlgebraKit's [`eigh_trunc_pullback!`](@extref MatrixAlgebraKit.eigh_trunc_pullback!) solving a Sylvester equation on the truncated subspace

!!! note
    Manually specifying a `rrule_alg` is considered expert-mode usage, and should only be done when full control over the implementation is desired.
    For all regular use cases, the default reverse rule algorithms, automatically chosen based on the forward algorithm, should be sufficient.
"""
struct EighAdjoint{F, R, T}
    fwd_alg::F
    rrule_alg::R
    trunc::T
end

const EIGH_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :DefaultAlgorithm => DefaultAlgorithm,
    :QRIteration => QRIteration,
    :Bisection => Bisection,
    :DivideAndConquer => DivideAndConquer,
    :Jacobi => Jacobi,
    :RobustRepresentations => RobustRepresentations,
    :Lanczos => (; tol = 1.0e-14, krylovdim = 30, kwargs...) -> IterEigh(; alg = Lanczos(; tol, krylovdim), kwargs...),
    :BlockLanczos => (; tol = 1.0e-14, krylovdim = 30, kwargs...) -> IterEigh(; alg = BlockLanczos(; tol, krylovdim), kwargs...),
)
const EIGH_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :full => FullEighPullback, :trunc => TruncEighPullback,
)

_default_eigh_rrule_alg(::MatrixAlgebraKit.Algorithm) = :full

function EighAdjoint(; fwd_alg = (;), rrule_alg = (;), trunc = (; alg = :notrunc))
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
            alg = _default_eigh_rrule_alg(fwd_algorithm),
            degeneracy_atol = Defaults.rrule_degeneracy_atol,
            verbosity = Defaults.eigh_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(EIGH_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = EIGH_RRULE_SYMBOLS[rrule_kwargs.alg]
        if rrule_type <: Union{FullEighPullback, TruncEighPullback}
            rrule_kwargs = (; rrule_kwargs.degeneracy_atol, rrule_kwargs.verbosity)
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

    return EighAdjoint(fwd_algorithm, rrule_algorithm, truncation_strategy)
end

"""
    eigh_trunc(t, alg::EighAdjoint; trunc=notrunc())
    eigh_trunc!(t, alg::EighAdjoint; trunc=notrunc())

Wrapper around `eigh_trunc(!)` which dispatches on the `EighAdjoint` algorithm.
This is needed since a custom adjoint may be defined, depending on the `alg`.
"""
MatrixAlgebraKit.eigh_trunc(t, alg::EighAdjoint) = eigh_trunc!(copy(t), alg)
function MatrixAlgebraKit.eigh_trunc!(t, alg::EighAdjoint)
    return eigh_trunc!(t, TruncatedAlgorithm(alg.fwd_alg, alg.trunc))
end
function MatrixAlgebraKit.eigh_trunc!(t::AdjointTensorMap, alg::EighAdjoint)
    D, V, ϵ = eigh_trunc!(adjoint(t), alg; trunc)
    return adjoint(D), adjoint(V), ϵ
end

#
## Forward algorithms
#

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

    IterEigh(; kwargs...)

Construct an `IterEigh` algorithm struct based on the following keyword arguments:

* `alg=KrylovKit.Lanczos(; tol=1e-14, krylovdim=25)` : KrylovKit algorithm struct for iterative eigenvalue decomposition.
* `fallback_threshold::Float64=Inf` : Threshold for `howmany / minimum(size(block))` above which (if the block is too small) the algorithm falls back to a dense decomposition.
* `start_vector=deterministic_start_vector` : Function providing the initial vector for the iterative algorithm.
"""
@kwdef struct IterEigh
    alg = KrylovKit.Lanczos(; tol = 1.0e-14, krylovdim = 25)
    fallback_threshold::Float64 = Inf
    start_vector = deterministic_start_vector
end
_default_eigh_rrule_alg(::IterEigh) = :trunc

# Compute eigh data block-wise using KrylovKit algorithm
function MatrixAlgebraKit.eigh_trunc!(f, alg::TruncatedAlgorithm{<:IterEigh})
    D, V = if isempty(blocksectors(f))
        # early return
        truncation_error = zero(real(scalartype(f)))
        MatrixAlgebraKit.initialize_output(eigh_full!, f, QRIteration()) # specified algorithm doesn't matter here
    else
        eighdata, dims = _compute_eighdata!(f, alg.alg, alg.trunc)
        _create_eightensors(f, eighdata, dims)
    end

    # construct info NamedTuple
    truncation_error =
        alg.trunc isa NoTruncation ? abs(zero(scalartype(f))) : norm(V * D * V' - f)

    return D, V, truncation_error
end

# Obtain sparse decomposition from block-wise eigsolve calls
function _compute_eighdata!(
        f, alg::IterEigh, trunc::Union{NoTruncation, TruncationSpace}
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
            D, V = eigh_full!(b, QRIteration())
            lm_ordering = sortperm(abs.(D.diag); rev = true) # order values and vectors consistently with eigsolve
            D = D.diag[lm_ordering] # extracts diagonal as Vector instead of Diagonal to make compatible with D of svdsolve
            V = stack(eachcol(V)[lm_ordering])[:, 1:howmany]
        else
            x₀ = alg.start_vector(b)
            eig_alg = alg.alg
            if howmany > alg.alg.krylovdim
                eig_alg = @set eig_alg.krylovdim = round(Int, howmany * 1.2)
            end
            D, lvecs, info = eigsolve(b, x₀, howmany, :LM, eig_alg)
            if info.converged < howmany  # Fall back to dense SVD if not properly converged
                @warn "Iterative eigendecomposition did not converge for block $c, falling back to eigh_full"
                D, V = eigh_full!(b, QRIteration())
                lm_ordering = sortperm(abs.(D.diag); rev = true)
                D = D.diag[lm_ordering]
                V = stack(eachcol(V)[lm_ordering])[:, 1:howmany]
            else  # Slice in case more values were converged than requested
                V = stack(view(lvecs, 1:howmany))
            end
        end

        # make it deterministic-ish
        MatrixAlgebraKit.gaugefix!(eigh_full!, V)

        resize!(D, howmany)
        dims[c] = length(D)
        return c => (D, V)
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
    V = similar(t, domain(t) ← W)
    for (c, (Dc, Vc)) in eighdata
        r = Base.OneTo(dims[c])
        copy!(block(D, c), Diagonal(view(Dc, r)))
        copy!(block(V, c), view(Vc, :, r))
    end
    return D, V
end

#
## Reverse-rule algorithms
#

function _get_pullback_gauge_tol(verbosity::Int)
    if verbosity ≤ 0 # never print gauge sensitivity
        return (_) -> Inf
    elseif verbosity == 1 # print gauge sensitivity above default atol
        MatrixAlgebraKit.default_pullback_gauge_atol
    else # always print gauge sensitivity
        return (_) -> 0.0
    end
end

# eigh_trunc! rrule wrapping MatrixAlgebraKit's eigh_pullback!
# https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/b76c7bb60014ecfead6925d0df6cb4b8d7c2668a/src/pullbacks/eigh.jl#L34
function ChainRulesCore.rrule(
        ::typeof(eigh_trunc!),
        t::AbstractTensorMap,
        alg::EighAdjoint{<:MatrixAlgebraKit.Algorithm, <:FullEighPullback}
    )

    D, V = eigh_full!(t; alg.fwd_alg)
    (D̃, Ṽ), inds = truncate(eigh_trunc!, (D, V), alg.trunc)
    truncerror = truncation_error(diagview(D), inds)

    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function eigh_trunc!_full_pullback(ΔDV)
        Δt = eigh_pullback!(
            zeros(scalartype(t), space(t)), t, (D, V), ΔDV, inds;
            gauge_atol = gtol(ΔDV), degeneracy_atol = alg.rrule_alg.degeneracy_atol,
        )
        return NoTangent(), Δt, NoTangent()
    end
    function eigh_trunc!_full_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (D̃, Ṽ, truncerror), eigh_trunc!_full_pullback
end

# eigh_trunc! rrule wrapping MatrixAlgebraKit's eigh_trunc_pullback! (also works for IterEigh)
# https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/b76c7bb60014ecfead6925d0df6cb4b8d7c2668a/src/pullbacks/eigh.jl#L113
function ChainRulesCore.rrule(
        ::typeof(eigh_trunc!),
        t,
        alg::EighAdjoint{<:Any, <:TruncEighPullback}
    )
    D, V, truncerror = eigh_trunc(t, alg)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function eigh_trunc!_trunc_pullback(ΔDV)
        Δf = eigh_trunc_pullback!(
            zeros(scalartype(t), space(t)), t, (D, V), ΔDV;
            gauge_atol = gtol(ΔDV), degeneracy_atol = alg.rrule_alg.degeneracy_atol,
        )
        return NoTangent(), Δf, NoTangent()
    end
    function eigh_trunc!_trunc_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return (D, V, truncerror), eigh_trunc!_trunc_pullback
end
