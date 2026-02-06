"""
$(TYPEDEF)

CTMRG algorithm assuming a C₄ᵥ-symmetric PEPS, i.e. invariance under 90° spatial rotation and
Hermitian reflection. This requires a single-site unit cell. The projector is obtained from
`eigh` decomposing the Hermitian enlarged corner.

## Fields

$(TYPEDFIELDS)

## Constructors

    C4vCTMRG(; kwargs...)

Construct a C₄ᵥ CTMRG algorithm struct based on keyword arguments.
For a full description, see [`leading_boundary`](@ref). The supported keywords are:

* `tol::Real=$(Defaults.ctmrg_tol)`
* `maxiter::Int=$(Defaults.ctmrg_maxiter)`
* `miniter::Int=$(Defaults.ctmrg_miniter)`
* `verbosity::Int=$(Defaults.ctmrg_verbosity)`
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))`
* `decomposition_alg::Union{<:EighAdjoint,NamedTuple}`
* `projector_alg::Symbol=:$(Defaults.projector_alg_c4v)`
"""
struct C4vCTMRG{P <: ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function C4vCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg = :c4v, kwargs...)
end
CTMRG_SYMBOLS[:c4v] = C4vCTMRG

"""
$(TYPEDEF)

Projector algorithm implementing the `eigh` decomposition of a Hermitian enlarged corner.

## Fields

$(TYPEDFIELDS)

## Constructors

    C4vEighProjector(; kwargs...)

Construct the C₄ᵥ `eigh`-based projector algorithm based on the following keyword arguments:

* `decomposition_alg::Union{<:EighAdjoint,NamedTuple}=EighAdjoint()` : `eigh` algorithm including the reverse rule. See [`EighAdjoint`](@ref).
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation strategy for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `verbosity::Int=$(Defaults.projector_verbosity)` : Projector output verbosity which can be:
    0. Suppress output information
    1. Print singular value degeneracy warnings
"""
struct C4vEighProjector{S <: EighAdjoint, T} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    verbosity::Int
end
function C4vEighProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :c4v_eigh, kwargs...)
end
PROJECTOR_SYMBOLS[:c4v_eigh] = C4vEighProjector

"""
$(TYPEDEF)

Projector algorithm implementing the `qr` decomposition of a column-enlarged corner.

## Fields

$(TYPEDFIELDS)

## Constructors

    C4vQRProjector(; kwargs...)

Construct the C₄ᵥ `qr`-based projector algorithm
based on the following keyword arguments:

* `decomposition_alg::Union{<:QRAdjoint,NamedTuple}=QRAdjoint()` : `qr` algorithm including the reverse rule. See [`QRAdjoint`](@ref).
"""
struct C4vQRProjector{S, T} <: ProjectorAlgorithm
    # TODO: support all `left_orth` algorithms
    decomposition_alg::S
    # TODO: remove unused attributes
    trunc::T
    verbosity::Int
end
function C4vQRProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :c4v_qr, kwargs...)
end
PROJECTOR_SYMBOLS[:c4v_qr] = C4vQRProjector

#
## C4v-symmetric CTMRG iteration (called through `leading_boundary`)
#

function ctmrg_iteration(
        network,
        env::CTMRGEnv,
        alg::C4vCTMRG,
    )
    if isa(alg.projector_alg, C4vEighProjector)
        enlarged_corner = c4v_enlarge(network, env, alg.projector_alg)
        corner′, projector, info = c4v_projector!(enlarged_corner, alg.projector_alg)
        edge′ = c4v_renormalize_edge(network, env, projector)
        return CTMRGEnv(corner′, edge′), info
    elseif isa(alg.projector_alg, C4vQRProjector)
        enlarged_corner = c4v_enlarge(env, alg.projector_alg)
        projector, info = c4v_projector!(enlarged_corner, alg.projector_alg)
        edge′ = c4v_renormalize_edge(network, env, projector)
        corner′ = c4v_qr_renormalize_corner(edge′, projector, info.R)
        return CTMRGEnv(corner′, edge′), info
    else
        throw(ArgumentError("Invalid C4v projector algorithm."))
    end
end

"""
    c4v_enlarge(network, env, ::C4vEighProjector)

Compute the normalized and Hermitian-symmetrized C₄ᵥ enlarged corner.
"""
function c4v_enlarge(network, env, ::C4vEighProjector)
    enlarged_corner = TensorMap(EnlargedCorner(network, env, (NORTHWEST, 1, 1)))
    # TODO: replace by `project_hermitian`
    enlarged_corner = 0.5 * (enlarged_corner + enlarged_corner')
    return enlarged_corner / norm(enlarged_corner)
end
"""
    c4v_enlarge(env, ::C4vQRProjector)

Compute the normalized column-enlarged northeast corner for C₄ᵥ QR-CTMRG.
"""
function c4v_enlarge(env, ::C4vQRProjector)
    return TensorMap(ColumnEnlargedCorner(env, (NORTHWEST, 1, 1)))
end

"""
    c4v_projector!(enlarged_corner, alg::C4vEighProjector)

Compute the C₄ᵥ projector from `eigh` decomposing the Hermitian `enlarged_corner`.
Also return the normalized eigenvalues as the new corner tensor.
"""
function c4v_projector!(enlarged_corner, alg::C4vEighProjector)
    trunc = truncation_strategy(alg, enlarged_corner)
    D, V, info = eigh_trunc!(enlarged_corner, decomposition_algorithm(alg); trunc)

    # Check for degenerate eigenvalues
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(D)
            vals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(D))
            @warn("degenerate eigenvalues detected: ", vals)
        end
    end

    return D / norm(D), V, (; D, V, info...)
end
"""
    c4v_projector!(enlarged_corner, alg::C4vQRProjector)

Compute the C₄ᵥ projector by decomposing the column-enlarged corner with `left_orth`.
```
                   R--←--
                   ↓
    C-←-E-←-  =  [~Q~]    
    ↓   |        ↓   |
```
"""
function c4v_projector!(enlarged_corner, ::C4vQRProjector)
    # TODO: support all `left_orth` algorithms
    Q, R = left_orth!(enlarged_corner)
    return Q, (; Q, R)
end

"""
    c4v_renormalize_edge(network, env, projector)

Renormalize the single edge tensor.
```
        |~~~|-←-E-←-|~~~|
    -←--| P'|   |   | P |--←-
        |~~~|---A---|~~~|
                |
```
"""
# TODO: possible missing twists for fermions
function c4v_renormalize_edge(network, env, projector)
    new_edge = renormalize_north_edge(env.edges[1], projector, projector', network[1, 1])
    # additional Hermitian projection step for numerical stability
    new_edge = _project_hermitian(new_edge)
    return new_edge / norm(new_edge)
end

"""
    c4v_qr_renormalize_corner(new_edge, projector, R)

Renormalize the single corner tensor
```
    C-←-E-←-|~~~|
    |   |   | P |-←-
    E---A---|~~~|
    |   |
    [~P']
      ↓
```
Using the already calculated QR decomposition
```
                   R--←--
                   ↓
    C-←-E-←-  =  [~P~]    
    ↓   |        ↓   |
```
we rewrite the renormalized corner as
```
    R-←-|~~~|
    ↓   | P |-←-
    E′--|~~~|
    ↓
```
which reuses the renormalized edge `E′` (`new_edge`).
(Credit: https://github.com/qiyang-ustc/QRCTM/blob/dd160116c3d7b02076691ceaf0a9833511ae532d/heisenberg.py#L80)
"""
# TODO: possible missing twists for fermions
function c4v_qr_renormalize_corner(new_edge::CTMRG_PEPS_EdgeTensor, projector, R)
    @tensor new_corner[χ; χ′] :=
        physical_flip(new_edge)[χ Dt Db; χ1] * R[χ1; χ2] * projector[χ2 Dt Db; χ′]
    new_corner = _project_hermitian(new_corner)
    return new_corner / norm(new_corner)
end
function c4v_qr_renormalize_corner(new_edge::CTMRG_PF_EdgeTensor, projector, R)
    @tensor new_corner[χ; χ′] :=
        physical_flip(new_edge)[χ D; χ1] * R[χ1; χ2] * projector[χ2 D; χ′]
    new_corner = _project_hermitian(new_corner)
    return new_corner / norm(new_corner)
end
# TODO: PEPS-PEPO-PEPS sandwich

# TODO: this should eventually be the constructor for a new C4vCTMRGEnv type
function CTMRGEnv(
        corner::AbstractTensorMap{T, S, 1, 1}, edge::AbstractTensorMap{T′, S, N, 1}
    ) where {T, T′, S, N}
    corners = fill(corner, 4, 1, 1)
    edge_SW = physical_flip(edge)
    edges = reshape([edge, edge, edge_SW, edge_SW], (4, 1, 1))
    return CTMRGEnv(corners, edges)
end

#
## utility
#

# TODO: re-examine these for fermions

# Adjoint of an edge tensor, but permutes the physical spaces back into the codomain.
# Intuitively, this conjugates a tensor and then reinterprets its 'direction' as an edge tensor.
function _dag(A::AbstractTensorMap{T, S, N, 1}) where {T, S, N}
    return permute(A', ((1, (3:(N + 1))...), (2,)))
end

function physical_flip(A::AbstractTensorMap{T, S, N, 1}) where {T, S, N}
    return flip(A, 2:N)
end

# call it `_project_hermitian` to avoid type piracy with MAK's exported project_hermitian
function _project_hermitian(E::AbstractTensorMap{T, S, N, 1}) where {T, S, N}
    E´ = (E + physical_flip(_dag(E))) / 2
    return E´
end
function _project_hermitian(C::AbstractTensorMap{T, S, 1, 1}) where {T, S}
    C´ = (C + C') / 2
    return C´
end

# TODO: check symmetry directly on InfiniteSquareNetwork
function check_symmetry(state, ::RotateReflect; atol = 1.0e-10)
    @assert length(state) == 1 "check_symmetry only works for single site unit cells"
    @assert norm(state[1] - _fit_spaces(rotl90(state[1]), state[1])) /
        norm(state[1]) < atol "not rotation invariant"
    @assert norm(state[1] - _fit_spaces(herm_depth(state[1]), state[1])) /
        norm(state[1]) < atol "not hermitian-reflection invariant"
    return nothing
end

#
## environment initialization
#

"""
    initialize_random_c4v_env([f=randn, T=scalartype(state)], state, Venv::ElementarySpace)

Initialize a C₄ᵥ-symmetric `CTMRGEnv` on virtual spaces `Venv` with random entries created
by `f` and scalartype `T`.
"""
function initialize_random_c4v_env(state, Venv::ElementarySpace)
    return initialize_random_c4v_env(randn, scalartype(state), state, Venv)
end
function initialize_random_c4v_env(f, T, state::InfinitePEPS, Venv::ElementarySpace)
    Vpeps = north_virtualspace(state, 1, 1)'
    return initialize_random_c4v_env(f, T, Vpeps ⊗ Vpeps', Venv)
end
function initialize_random_c4v_env(f, T, state::InfinitePartitionFunction, Venv::ElementarySpace)
    Vpf = north_virtualspace(state, 1, 1)'
    return initialize_random_c4v_env(f, T, Vpf, Venv)
end
function initialize_random_c4v_env(f, T, Vstate::VectorSpace, Venv::ElementarySpace)
    corner₀ = DiagonalTensorMap(randn(real(T), Venv ← Venv))
    edge₀ = f(T, Venv ⊗ Vstate ← Venv)
    edge₀ = _project_hermitian(edge₀)
    return CTMRGEnv(corner₀, edge₀)
end

"""
    initialize_singlet_c4v_env([T=scalartype(state)], state::InfinitePEPS, Venv::ElementarySpace)

Initialize a C₄ᵥ-symmetric `CTMRGEnv` with a singlet corner of dimension `dim(Venv)` and an
identity edge from `id(T, Venv ⊗ Vpeps)`.
"""
function initialize_singlet_c4v_env(state::InfinitePEPS, Venv::ElementarySpace)
    return initialize_singlet_c4v_env(scalartype(state), state, Venv)
end
function initialize_singlet_c4v_env(T, state::InfinitePEPS, Venv::ElementarySpace)
    Vpeps = north_virtualspace(state, 1, 1)'
    return initialize_singlet_c4v_env(T, Vpeps, Venv)
end
function initialize_singlet_c4v_env(T, Vpeps::ElementarySpace, Venv::ElementarySpace)
    corner₀ = DiagonalTensorMap(zeros(real(T), Venv ← Venv))
    corner₀.data[1] = one(real(T))
    edge₀ = permute(id(T, Venv ⊗ Vpeps), ((1, 2, 4), (3,)))
    return CTMRGEnv(corner₀, edge₀)
end
