const NNGate{T, S} = AbstractTensorMap{T, S, 2, 2}

"""
Apply 1-site `gate` on the PEPS or PEPO tensor `a`.
"""
function _apply_sitegate(
        a::PEPSTensor, gate::AbstractTensorMap{T, S, 1, 1}; purified::Bool = true
    ) where {T, S}
    @assert purified
    return gate * a
end

function _apply_sitegate(
        a::PEPOTensor, gate::AbstractTensorMap{T, S, 1, 1}; purified::Bool = true
    ) where {T, S}
    @plansor a′[p1 p2; n e s w] := gate[p1; p] * a[p p2; n e s w]
    if !purified
        @plansor a′[p1 p2; n e s w] := a′[p1 p; n e s w] * gate[p; p2]
    end
    return a′
end

# --- 2-site gates on nearest neighbor bonds ---------------------------------------------
#
# The bond between `s₁` and `s₂` is updated in the native PEPS/PEPO leg order, using the
# bond direction to locate the legs. Each site tensor is split as `T = Q * R`, where the
# reduced tensor `R` holds the physical leg acted on by the gate and the bond leg. The gate is
# applied to `R₁ -- R₂` and split again by a truncated SVD, after which `Q` and the new `R`
# are contracted back into the native leg order.
#
# Temporaries allocated with `allocator` never outlive the function that allocates them:
# each function below frees its own temporaries and resets `allocator` before returning.

"""
    _orient_bond(state, sites, gate) -> (s₁, k₁), (s₂, k₂), gate

Order the two `sites` of a nearest neighbor bond and the virtual axes `k₁`, `k₂` of their
bond legs, such that the bond leg of `s₁` is non-dual. Putting `s₁` on the codomain side
of the SVD then reproduces the original arrow on the bond, so nothing has to be flipped.
"""
function _orient_bond(state::InfiniteState, sites::Vector{CartesianIndex{2}}, gate::NNGate)
    s₁, s₂ = sites
    k₁ = first(_nn_bonddir(s₁, s₂))
    k₂ = mod1(k₁ + 2, 4)
    T₁ = state[s₁]
    # `space` of a domain leg returns its dual
    isdual(space(T₁, numout(T₁) + k₁)) && return (s₁, k₁), (s₂, k₂), gate
    return (s₂, k₂), (s₁, k₁), permute(gate, ((2, 1), (4, 3)))
end

"""
    _bond_qr(T, k, gate_ax, [allocator]) -> Q, R, p

Split `T = Q * R`, where the reduced tensor `R : x ← p ⊗ v` holds the physical axis `gate_ax`
and the virtual axis `k` (the bond) of `T`, while `Q` holds all other axes. `p` is the
permutation that brings the product back to the leg order of `T`. For example, for a
PEPSTensor `T[p; n e s w]` with the bond `v = e` on its east axis (`k = 2`),
```
        n                       n
        |                       |
    w - T - v       =       w - Q ← x ← R - v
        | ↘                     |        ↘
        s   p                   s         p
```
For a PEPOTensor, the physical axis that is not `gate_ax` stays in `Q`.
The permuted copy of `T` that is factorized is a temporary allocated with `allocator`.
"""
function _bond_qr(
        T::Union{PEPSTensor, PEPOTensor}, k::Int, gate_ax::Int, allocator = DefaultAllocator()
    )
    Np, M = numout(T), numind(T)
    r = (gate_ax, Np + k)
    q = TupleTools.deleteat(ntuple(identity, M), r)
    cp = allocator_checkpoint!(allocator)
    T′ = tensoralloc_add(scalartype(T), T, (q, r), false, Val(true), allocator)
    tensoradd!(T′, T, (q, r), false, One(), Zero(), DefaultBackend(), allocator)
    Q, R = left_orth!(T′; positive = true)
    tensorfree!(T′, allocator)
    allocator_reset!(allocator, cp)
    return Q, R, _inv_mpo_perm((q..., r...), Val(Np))
end

"""
    _undo_bond_qr(Q, X, pX, p, [allocator])

Inverse of [`_bond_qr`](@ref): contract `Q` with `X`, whose legs selected by `pX` are
ordered as `x ← p ⊗ v`, into the leg order specified by `p`.
"""
function _undo_bond_qr(
        Q::AbstractTensorMap, X::AbstractTensorMap, pX, p, allocator = DefaultAllocator()
    )
    N = numind(Q)
    pQ = (ntuple(identity, N - 1), (N,))
    cp = allocator_checkpoint!(allocator)
    T = tensorcontract(Q, pQ, false, X, pX, false, p, One(), DefaultBackend(), allocator)
    allocator_reset!(allocator, cp)
    return T
end

"""
    _apply_bond_gate(R₁, R₂, gate, trunc, [allocator]) -> U, S, Vᴴ, ϵ

Apply the 2-site `gate` to the reduced tensors `R₁ : x₁ ← p₁ ⊗ v` and `R₂ : x₂ ← p₂ ⊗ v'`
```
    x₁ ← R₁ --- v --- R₂ → x₂           p₁            p₂
         ↘            ↘                 ↓             ↓
          p₁           p₂               |----gate-----|
          ↓            ↓        or      ↓             ↓
          |----gate----|                p₁            p₂
          ↓            ↓                ↘             ↘
          p₁           p₂          x₁ ← R₁ --- v --- R₂ → x₂
```
where the second form is used when the physical legs are dual (the second physical axis of
a PEPO), and split the result as `U S Vᴴ` with a truncated SVD. The square root of `S` is
absorbed into `U : x₁ ⊗ p₁ ← v′` and `Vᴴ : v′ ← x₂ ⊗ p₂`.
"""
function _apply_bond_gate(
        R₁::AbstractTensorMap, R₂::AbstractTensorMap, gate::NNGate,
        trunc::TruncationStrategy, allocator = DefaultAllocator()
    )
    θ = if isdual(space(R₁, 2))
        @tensor allocator = allocator θ[-1 -2; -3 -4] :=
            R₁[-1; 1 3] * R₂[-3; 2 3] * gate[1 2; -2 -4]
    else
        @tensor allocator = allocator θ[-1 -2; -3 -4] :=
            R₁[-1; 1 3] * R₂[-3; 2 3] * gate[-2 -4; 1 2]
    end
    U, S, Vᴴ, ϵ = svd_trunc!(θ; trunc)
    absorb_s!(U, S, Vᴴ)
    return U, S, Vᴴ, ϵ
end
