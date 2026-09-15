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
* `decomposition_alg::Union{NamedTuple,<:EighAdjoint,<:QRAdjoint}=(;)`
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
    return CTMRGAlgorithm(; alg = :C4vCTMRG, kwargs...)
end
CTMRG_SYMBOLS[:C4vCTMRG] = C4vCTMRG

function check_input(
        ::typeof(leading_boundary), network::InfiniteSquareNetwork, env::CTMRGEnv, alg::C4vCTMRG; atol = 1.0e-10
    )
    # check unit cell size
    length(network) == 1 || throw(ArgumentError("C4v CTMRG is only compatible with single-site unit cells."))
    O = network[1, 1]
    # check for fermionic braiding statistics
    BraidingStyle(sectortype(spacetype(network))) != Bosonic() &&
        throw(ArgumentError("C4v CTMRG is currently only implemented for networks consisting of tensors with bosonic braiding."))
    # check sufficient condition for equality and duality of spaces that we assume
    west_virtualspace(O) == _elementwise_dual(north_virtualspace(O)) ||
        throw(ArgumentError("C4v CTMRG requires south and west virtual space to be the dual of north and east virtual space."))
    # check rotation invariance of the local tensors, with the exact spaceflips we assume
    is_rotation_invariant = try
        _isapprox_localsandwich(O, flip_virtualspace(_rotl90_localsandwich(O), [EAST, WEST]); atol)
    catch
        false # _isapprox_localsandwich errors if the symmetry action changes the spaces (e.g. in case of non self-dual irreps)
    end
    is_rotation_invariant || @warn("The local tensors are not invariant under 90° rotation. In general, C4v CTMRG is not expected to work in this case.")
    # check the hermitian reflection invariance of the local tensors, with the exact spaceflips we assume
    is_herm_reflection_invariant = try
        _isapprox_localsandwich(O, flip_physicalspace(flip_virtualspace(herm_depth(O), [EAST, WEST])); atol)
    catch
        false
    end
    is_herm_reflection_invariant || @warn("The local tensors are not invariant under hermitian reflection. In general, C4v CTMRG is not expected to work in this case.")
    # TODO: check compatibility of network and environment spaces in general?
    return nothing
end

# The corners in C₄ᵥ CTMRG are identical and the four edges are related by
# rotation, so all edges produce the same singular values. `eigh_vals` is about
# as expensive as `svd_vals` on CPU but much cheaper on GPU, and gives effectively
# the same information. For the QR projector, off-diagonal elements are still
# present until CTMRG converges, so we need a fallback for the non-diagonal case.
corner_spectrum(C::AbstractTensorMap, ::C4vCTMRG) = eigh_vals(C)

"""
    convergence_tensors(env::CTMRGEnv, alg::C4vCTMRG) -> (corners, edges)

The corners and edges whose singular values determine convergence.

For C₄ᵥ-symmetric CTMRG, only one corner and one edge are needed since
the corners are identical, and the edges have identical singular values.
"""
convergence_tensors(env::CTMRGEnv, ::C4vCTMRG) =
    (view(env.corners, 1:1, :, :), view(env.edges, 1:1, :, :))

function ctmrg_iteration(network, env::CTMRGEnv, alg::C4vCTMRG)
    projector_alg = alg.projector_alg
    enlarged_corner = c4v_enlarge(network, env, projector_alg)
    projector, info = c4v_projector!(enlarged_corner, projector_alg)
    edge′ = c4v_renormalize_edge(network, env, projector)
    corner′ = c4v_renormalize_corner(edge′, projector, info, projector_alg)
    return CTMRGEnv(corner′, edge′), info
end

"""Report unsupported projector algorithms when enlarging a C₄ᵥ corner."""
function c4v_enlarge(network, env, alg::ProjectorAlgorithm)
    throw(ArgumentError("Unknown C4v projector algorithm $(typeof(alg))"))
end

"""
Renormalize the single C₄ᵥ edge tensor.
```
        |~~~|-←-E-←-|~~~|
    -←--| P'|   |   | P |--←-
        |~~~|---A---|~~~|
                |
```
"""
function c4v_renormalize_edge(network, env, projector)
    # TODO: possible missing twists for fermions
    new_edge = renormalize_north_edge(env.edges[1], projector, projector', network[1, 1])
    # additional Hermitian projection step for numerical stability
    new_edge = _project_hermitian(new_edge)
    return new_edge / norm(new_edge)
end

"""
Renormalize the single C₄ᵥ corner tensor.
```
    C-←-E-←-|~~~|
    |   |   | P |-←-
    E---A---|~~~|
    |   |
    [~P']
    ↓
```

For `C4vEighProjector`, it is already given by truncated eigenvalues of the enlarged corner.

For `C4vQRProjector`, we can use the already calculated QR decomposition
```
                   R--←--
                   ↓
    C-←-E-←-  =  [~P~]
    ↓   |        ↓   |
```
to rewrite the renormalized corner as
```
    R-←-|~~~|
    ↓   | P |-←-
    E′--|~~~|
    ↓
```
which reuses the renormalized edge `E′` (`new_edge`).
(Credit: https://github.com/qiyang-ustc/QRCTM/blob/dd160116c3d7b02076691ceaf0a9833511ae532d/heisenberg.py#L80)
"""
function c4v_renormalize_corner(new_edge::CTMRGEdgeTensor, projector, info, ::C4vEighProjector)
    return info.D / norm(info.D)
end
function c4v_renormalize_corner(new_edge::CTMRGEdgeTensor, projector, info, ::C4vQRProjector)
    # TODO: possible missing twists for fermions
    # contract updated edge and R
    edge′ = physical_flip(new_edge)
    ER = edge′ * twistdual(info.R, 1)
    # contract (edge′, R) with projector
    new_corner = contract_edges(ER, projector)
    new_corner = project_hermitian(new_corner)
    return new_corner / norm(new_corner)
end

"""Contract two CTMRG edge tensors into a corner tensor."""
@generated function contract_edges(
        EL::CTMRGEdgeTensor{T, S, N}, ER::CTMRGEdgeTensor{T, S, N}
    ) where {T, S, N}
    C´_e = tensorexpr(:C´, -1, -2)
    EL_e = tensorexpr(:EL, (-1, (2:N)...), 1)
    ER_e = tensorexpr(:ER, 1:N, -2)
    return macroexpand(@__MODULE__, :(return @tensor $C´_e := $EL_e * $ER_e))
end

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
