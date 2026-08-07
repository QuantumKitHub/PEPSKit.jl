#
# Characteristic equations used in implicit differentiation of CMTRG contractions
#

## Convenience aliases

const CornerTensor{S, N} = AbstractTensorMap{T, S, 1, 1} where {T}
const EdgeTensor{S, N} = AbstractTensorMap{T, S, N, 1} where {T}
const LeftProjector{S, N} = AbstractTensorMap{T, S, N, 1} where {T}
const RightProjector{S, N} = AbstractTensorMap{T, S, 1, N} where {T}

## C4v symmetric case

# partial contractions for different networks

# north edge and its left projector
function contract_EnVd(
        En::EdgeTensor{S, 3}, Vd::LeftProjector{S, 3}, O::PEPSSandwich
    ) where {S}
    @autoopt @tensor EnVd[χ_NNW D_W_a D_W_b D_S_a D_S_b; χ_E] :=
        Vd[χ_NNE D_E_a D_E_b; χ_E] *
        En[χ_NNW D_N_a D_N_b; χ_NNE] *
        ket(O)[d; D_N_a D_E_a D_S_a D_W_a] *
        conj(bra(O)[d; D_N_b D_E_b D_S_b D_W_b])
    return EnVd
end
function contract_EnVd(
        En::EdgeTensor{S, 2}, Vd::LeftProjector{S, 2}, O::PartitionFunctionTensor
    ) where {S}
    @autoopt @tensor EnVd[χ_NNW D_W D_S; χ_E] :=
        Vd[χ_NNE D_E; χ_E] *
        En[χ_NNW D_N; χ_NNE] *
        O[D_W D_S; D_N D_E]
    return EnVd
end

# northwest enlarged corner with its left projector
function contract_EwCEnVd(Ew::EdgeTensor{S, 3}, C::CornerTensor{S}, EnVd::LeftProjector{S, 5}) where {S}
    @autoopt @tensor EwCEnVd[χ_WSW D_S_a D_S_b; χ_E] :=
        Ew[χ_WSW D_W_a D_W_b; χ_WNW] *
        C[χ_WNW; χ_NNW] *
        EnVd[χ_NNW D_W_a D_W_b D_S_a D_S_b; χ_E]
    return EwCEnVd
end
function contract_EwCEnVd(Ew::EdgeTensor{S, 2}, C::CornerTensor{S}, EnVd::LeftProjector{S, 3}) where {S}
    @autoopt @tensor EwCEnVd[χ_WSW D_S; χ_E] :=
        Ew[χ_WSW D_W; χ_WNW] *
        C[χ_WNW; χ_NNW] *
        EnVd[χ_NNW D_W D_S; χ_E]
    return EwCEnVd
end

# north edge renormalization
function contract_E´(Ud::RightProjector{S, 3}, EnVd::LeftProjector{S, 5}) where {S}
    @tensor E´[-1 -2 -3; -4] := Ud[-1; 1 2 3] * EnVd[1 2 3 -2 -3; -4]
    return E´
end
function contract_E´(Ud::RightProjector{S, 2}, EnVd::LeftProjector{S, 3}) where {S}
    @tensor E´[-1 -2; -3] := Ud[-1; 1 2] * EnVd[1 2 -2; -3]
    return E´
end

"""
    generate_symmetric_characteristic_equation(
        Cfp::CornerTensor,
        Efp::EdgeTensor, # unused
        Ufp::LeftProjector,
        ULfp::LeftProjector,
    )

Takes the fixed-point values of the corner tensor `Cfp`, edge tensor `Efp`, left isometry
`Ufp` and its left null space `ULfp` corresponding to a converged C4v CTMRG contraction, and
generates a function ``F(s, C, E, u)`` which characterizes the convergence of the C4v CTMRG
algorithm in terms of the characteristic equation ``F(s, C, E, u) = 0``.
Here, ``s`` corresponds to a state variable (e.g. an `InfinitePEPS` that is being optimized),
and ``(C, E, u)`` represents a C4v symmetric contraction environment.
``C`` and ``E`` directly represent the corner and edge tensors, while ``u`` parametrizes
a differentiable projector ``U`` as ``U = U_{fp} + U_{L,fp} * u``.

``F`` returns a tuple of three tensors, corresponding to an equation for ``C``, ``E`` and
``u`` respectively:
```
    C---E---|~~~|
    |   |   | U |---  - λC * --C-- 
    E---O---|~~~|
    |   |
    [ U†]
      |
```
```
    |~~~|---E---|~~~|
 ---| U†|   |   | U |---  - λE *  --E--
    |~~~|---O---|~~~|               |
            |
```
```
    C---E---|~~~|
    |   |   | U |---Cfp^{-1}  - λC * --u-- 
    E---O---|~~~|
    |   |
    [ ULfp†]
      |
```
where ``λ_C`` and ``λ_E`` which are defined as the inner product of the first term in the
first two contractions given here with ``C`` and ``E`` respectively.
"""
function generate_symmetric_characteristic_equation(
        Cfp::CornerTensor,
        Efp::EdgeTensor, # unused
        Ufp::LeftProjector,
        ULfp::LeftProjector,
    )

    iC = sdiag_pow(real(DiagonalTensorMap(Cfp)), -1)
    ULd = ULfp'

    function symmetric_characteristic_equation(state, C, E, u)
        network = InfiniteSquareNetwork(state)
        O = network[1, 1]

        # project input
        C = _project_hermitian(C)
        E = _project_hermitian(E)

        # assuming 'eigenvalue decomposition' of enlarged corner as ECE = U * S * V
        # where now sectretly V = U†
        U = Ufp + ULfp * u # left isometry
        Ep = physical_flip(E) # edge (west or south)

        # then the 'right projector' (which goes on the left side) is U†
        Ud = U'

        # evaluate partial contractions to reuse
        # north edge and its left projector
        EnVd = contract_EnVd(E, U, O)
        EwCEnVd = contract_EwCEnVd(Ep, C, EnVd)

        # F1: corner
        C´ = Ud * EwCEnVd
        C´ = _project_hermitian(C´) # project output
        λ_C = dot(C, C´)
        F1 = C´ / λ_C - C

        # F2: edge
        E´ = contract_E´(Ud, EnVd)
        E´ = _project_hermitian(E´) # project output
        λ_E = dot(E, E´)
        F2 = E´ / λ_E - E

        # F3: u
        ULdEwCEnVd = ULd * EwCEnVd
        F3 = (ULdEwCEnVd * iC) / λ_C - u

        return F1, F2, F3
    end

    return symmetric_characteristic_equation
end
