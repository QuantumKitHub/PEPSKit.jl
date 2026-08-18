#
# Characteristic equations used in implicit differentiation of CMTRG contractions
#

## Convenience aliases

const CornerTensor{S, N} = AbstractTensorMap{T, S, 1, 1} where {T}
const EdgeTensor{S, N} = AbstractTensorMap{T, S, N, 1} where {T}
const LeftProjector{S, N} = AbstractTensorMap{T, S, N, 1} where {T}
const RightProjector{S, N} = AbstractTensorMap{T, S, 1, N} where {T}

const CornerTensors{TC} = Array{TC, 3} where {TC <: CornerTensor}
const EdgeTensors{TE} = Array{TE, 3} where {TE <: EdgeTensor}
const LeftProjectors{TP} = Array{TP, 3} where {TP <: LeftProjector}
const RightProjectors{TP} = Array{TP, 3} where {TP <: RightProjector}


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

# northwest enlarged corner with its left projector
function contract_EwCEnVd(Ew::EdgeTensor{S, 3}, C::CornerTensor{S}, EnVd::LeftProjector{S, 5}) where {S}
    @autoopt @tensor EwCEnVd[χ_WSW D_S_a D_S_b; χ_E] :=
        Ew[χ_WSW D_W_a D_W_b; χ_WNW] *
        C[χ_WNW; χ_NNW] *
        EnVd[χ_NNW D_W_a D_W_b D_S_a D_S_b; χ_E]
    return EwCEnVd
end

# north edge renormalization
# NOTE: will need to be rewritten in terms of tensor multiplication for (future) fermion support
function contract_E´(Ud::RightProjector{S, 3}, EnVd::LeftProjector{S, 5}) where {S}
    @tensor E´[-1 -2 -3; -4] := Ud[-1; 1 2 3] * EnVd[1 2 3 -2 -3; -4]
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
        C = project_hermitian(C)
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
        C´ = project_hermitian(C´) # project output
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


## Generic asymmetric case

# Index helpers
# -------------

# unit cell indices for absorbing S⁻¹ into left and right projectors
function _proj_sinv_indices(coordinate, nrows, ncols)
    dir, r, c = coordinate
    r′, c′ = if dir == NORTH
        _prev(r, nrows), c
    elseif dir == EAST
        r, _next(c, ncols)
    elseif dir == SOUTH
        _next(r, nrows), c
    elseif dir == WEST
        r, _prev(c, ncols)
    end
    return dir, r′, c′
end
# unit cell indices for absorbing fourthroot of inverse S^2 into U isometry
function _leftvec_invfroot_indices(coordinate, nrows, ncols)
    dir, r, c = coordinate
    r′, c′ = if dir == NORTH
        _next(r, nrows), _prev(c, ncols)
    elseif dir == EAST
        _prev(r, nrows), _prev(c, ncols)
    elseif dir == SOUTH
        _prev(r, nrows), _next(c, ncols)
    elseif dir == WEST
        _next(r, nrows), _next(c, ncols)
    end
    return _prev(dir, 4), r′, c′
end
# unit cell indices for absorbing forthroot of inverse S^2 into V isometry
function _rightvec_invfroot_indices(coordinate, nrows, ncols)
    dir, r, c = coordinate
    r′, c′ = if dir == NORTH
        r, _next(_next(c, ncols), ncols)
    elseif dir == EAST
        _next(_next(r, nrows), nrows), c
    elseif dir == SOUTH
        r, _prev(_prev(c, ncols), ncols)
    elseif dir == WEST
        _prev(_prev(r, nrows), nrows), c
    end
    return _next(dir, 4), r′, c′
end

# corner coordinate relative to enlarged corner position
function _above_left(co, nrows, ncols)
    dir, r, c = co
    if dir == 1
        return dir, _prev(r, nrows), _prev(c, ncols)
    elseif dir == 2
        return dir, _prev(r, nrows), _next(c, ncols)
    elseif dir == 3
        return dir, _next(r, nrows), _next(c, ncols)
    elseif dir == 4
        return dir, _next(r, nrows), _prev(c, ncols)
    end
end

# edge coordinate relative to enlarged corner position
function _left(co, nrows, ncols)
    dir, r, c = co
    if dir == 1
        return _prev(dir, 4), r, _prev(c, ncols)
    elseif dir == 2
        return _prev(dir, 4), _prev(r, nrows), c
    elseif dir == 3
        return _prev(dir, 4), r, _next(c, ncols)
    elseif dir == 4
        return _prev(dir, 4), _next(r, nrows), c
    end
end
function _above(co, nrows, ncols)
    dir, r, c = co
    if dir == 1
        return dir, _prev(r, nrows), c
    elseif dir == 2
        return dir, r, _next(c, ncols)
    elseif dir == 3
        return dir, _next(r, nrows), c
    elseif dir == 4
        return dir, r, _prev(c, ncols)
    end
end

# projector coordinate relative to enlarged corner position
function _left_projector(co, nrows, ncols)
    dir, r, c = co
    if dir == 1
        return dir, r, _prev(c, ncols)
    elseif dir == 2
        return dir, _prev(r, nrows), c
    elseif dir == 3
        return dir, r, _next(c, ncols)
    elseif dir == 4
        return dir, _next(r, nrows), c
    end
end

# Custom roots
# ------------

# square root of a diagonal TensorMap, but with complex non-diagonal adjoint
squareroot(t::AbstractTensorMap) = sdiag_pow(t, 0.5)

function ChainRulesCore.rrule(
        ::typeof(squareroot), t::AbstractTensorMap{T, S, 1, 1}
    ) where {T, S}
    domain(t) == codomain(t) ||
        error("Square root of a tensor only exists when domain == codomain.")
    P_t = ProjectTo(t) # does this projection project down to a diagonal variation, if t is diagonal?
    C = squareroot(t)

    function squareroot_pullback(ΔC_)
        ΔC = unthunk(ΔC_)
        F = similar(t)
        for (c, b) in blocks(C)
            copyto!(block(F, c), _squareroot_pullback(b))
        end
        return NoTangent(), P_t(_elementwise_mult(ΔC, F))
    end
    return C, squareroot_pullback
end
function _squareroot_pullback(C::AbstractMatrix)
    Fdata = similar(C)
    for j in axes(Fdata, 2), i in axes(Fdata, 1)
        # Taking the diagonal only is okay, when dA is diagonal anyway: Fdata[i, i] = 1 / (2 * conj(C[i, i]))
        Fdata[i, j] = 1 / conj(C[i, i] + C[j, j])
    end
    return Fdata
end

# take fourth root of diagonal TensorMap, but with complex non-diagonal adjoint
fourthroot(t::AbstractTensorMap) = sdiag_pow(t, 0.25)

function ChainRulesCore.rrule(
        ::typeof(fourthroot), t::AbstractTensorMap{T, S, 1, 1}
    ) where {T, S}
    domain(t) == codomain(t) ||
        error("Square root of a tensor only exist when domain == codomain.")
    P_t = ProjectTo(t) # does this projection project down to a diagonal variation, if t is diagonal?
    C = fourthroot(t)

    function fourthroot_pullback(ΔC_)
        ΔC = unthunk(ΔC_)
        F = similar(t)
        for (c, b) in blocks(C)
            copyto!(block(F, c), _fourthroot_pullback(block(ΔC, c), b))
        end
        return NoTangent(), P_t(F)
    end
    return C, fourthroot_pullback
end
function _fourthroot_pullback(ΔC::AbstractMatrix, C::AbstractMatrix)
    # Taking the diagonal only is okay, when dA is diagonal anyway: Fdata[i, i] = 1 / (4 * conj(C[i, i]^3))
    # However, for Q-deformed CTMRG, we need the full version:
    Cd = diagview(C) # column
    Cdt = transpose(Cd) # row
    F = @. ΔC / conj(Cd^3 + Cd * Cdt^2 + Cd^2 * Cdt + Cdt^3)
    return F
end

# Util
# ----

function eachcoordinate(tensor_unitcell::Array{<:AbstractTensorMap, 3})
    return collect(Iterators.product(axes(tensor_unitcell)...))
end

# add rrule for twistdual through dummy out-of-place function
function _twistdual(t::AbstractTensorMap, i::Int)
    isdual(space(t, i)) || return t
    return twist(t, i)
end
function _twistdual(t::AbstractTensorMap, is)
    is′ = filter(i -> isdual(space(t, i)), is)
    return twist(t, is′)
end
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(twistdual), t::AbstractTensorMap, i
    )
    tout, twistdual_pullback = rrule_via_ad(config, _twistdual, t, i)
    return tout, twistdual_pullback
end

# add rrule for twistnondual through dummy out-of-place function
function _twistnondual(t::AbstractTensorMap, i::Int)
    !isdual(space(t, i)) || return t
    return twist(t, i)
end
function _twistnondual(t::AbstractTensorMap, is)
    is′ = filter(i -> !isdual(space(t, i)), is)
    return twist(t, is′)
end
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(twistnondual), t::AbstractTensorMap, i
    )
    tout, twistnondual_pullback = rrule_via_ad(config, _twistnondual, t, i)
    return tout, twistnondual_pullback
end

function absorb_left(
        E::AbstractTensorMap{T, S}, C::CornerTensor{S}
    ) where {T, S}
    pC = (codomainind(C), domainind(C))
    pE = ((codomainind(E)[1],), (codomainind(E)[2:end]..., domainind(E)...))
    pCE = (codomainind(E), domainind(E))
    return tensorcontract(C, pC, false, E, pE, false, pCE)
end
function absorb_right(
        P::AbstractTensorMap{T, S}, C::CornerTensor{S}
    ) where {T, S}
    pP = ((codomainind(P)..., domainind(P)[2:end]...), (domainind(P)[1],))
    pC = (codomainind(C), domainind(C))
    pPC = (codomainind(P), (domainind(P)[end], domainind(P)[1:(end - 1)]...))
    return tensorcontract(P, pP, false, C, pC, false, pPC)
end
# specialized versions; TODO: probably remove, this is a terrible idea for fermionic tensors...
function absorb_right(E::EdgeTensor{S}, C::CornerTensor{S}) where {S}
    return E * C
end
function absorb_left_right(T::AbstractTensorMap, CL::CornerTensor, CR::CornerTensor)
    return absorb_right(absorb_left(T, CL), CR)
end

# Partial contractions
# --------------------

function contract_EPL(
        E::EdgeTensor{S, 3}, PL::LeftProjector{S, 3}, O::PEPSSandwich,
    ) where {S}
    @autoopt @tensor eipl[χ_W D_W_above D_W_below D_S_above D_S_below; χ_S] :=
        PL[χ_N D_E_above D_E_below; χ_S] *
        E[χ_W D_N_above D_N_below; χ_N] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
    return eipl
end

function contract_EiCiEPL(
        EiCi::EdgeTensor{S, 3}, EPL::LeftProjector{S, 5},
    ) where {S}
    @tensor ecepl[-1 -2 -3; -4] :=
        EPL[1 2 3 -2 -3; -4] *
        EiCi[-1 2 3; 1]
    return ecepl
end

function contract_PREPL(
        PR::RightProjector{S, 3}, EPL::LeftProjector{S, 5},
    ) where {S}
    @tensor E´[-1 -2 -3; -4] :=
        PR[-1; 1 2 3] * EPL[1 2 3 -2 -3; -4]
    return E´
end

# contract two projectors into bond matrix; proxy for multiplication, but with fermion consistency
function _contract_PR_PL(PR::RightProjector{S, N}, PL::LeftProjector{S, N}) where {S, N}
    pPR = (codomainind(PR), domainind(PR))
    pPL = (codomainind(PL), domainind(PL))
    pPLR = ((1,), (2,))
    return tensorcontract(PR, pPR, false, PL, pPL, false, pPLR)
end

function _contract_PR_M(PR::RightProjector{S, N}, M::AbstractTensorMap{T, S, N, N}) where {T, S, N}
    pPR = (codomainind(PR), domainind(PR))
    pM = (codomainind(M), domainind(M))
    return tensorcontract(PR, pPR, false, M, pM, false, pPR)
end

# computation of characteristic equations by constructing full enlarged corners
function contract_halfinfinite_characteristic_equation(
        C::CornerTensors, E::EdgeTensors,
        is::CornerTensors, s::CornerTensors,
        u::CornerTensors, v::CornerTensors,
        n::InfiniteSquareNetwork,
        Ud::RightProjectors, Vd::LeftProjectors,
        iCi::CornerTensors,
        ULd::RightProjectors, VRd::LeftProjectors,
        iSfp::CornerTensors,
    )
    coordinates = eachcoordinate(n, 1:4)
    nrows, ncols = size(n)

    # precompute rotated local sandwiches, enlarged corners, and projectors
    Or = map(coordinates) do co
        dir, r, c = co
        return _rotate_north_localsandwich(n[r, c], dir)
    end
    EC = map(coordinates) do co
        return TensorMap(EnlargedCorner(n, CTMRGEnv(iCi, E), co))
    end
    PR = map(coordinates) do co
        co′ = _proj_sinv_indices(co, nrows, ncols)
        return absorb_right(Ud[co...] * EC[co...], is[co′...])
    end
    PLpart = map(coordinates) do co
        co′ = _next_coordinate(co, nrows, ncols)
        return EC[co′...] * Vd[co...]
    end
    PL = map(coordinates) do co
        co′ = _proj_sinv_indices(co, nrows, ncols)
        return absorb_left(PLpart[co...], is[co′...])
    end

    # prepare partial contractions with projectors for F1 and F2

    # absorb corner into right side of edge
    EiCi = map(coordinates) do co
        return E[_left(co, nrows, ncols)...] * iCi[_above_left(co, nrows, ncols)...]
    end
    # pre-contract the top edge and its sandwich into the left projector
    EPL = map(coordinates) do co
        return contract_EPL(E[_above(co, nrows, ncols)...], PL[co...], Or[co...])
    end
    # pre-contract the top-left corner and left edge into the previous to complete the
    # top-left enlarged corner
    EiCiEPL = map(coordinates) do co
        return contract_EiCiEPL(EiCi[co...], EPL[co...])
    end

    # corners
    F1 = map(coordinates) do co
        co´ = _prev_coordinate(co, nrows, ncols)
        C´ = _contract_PR_PL(PR[co´...], EiCiEPL[co...])
        λC = dot(C[co...], C´)
        return C´ / λC - C[co...]
    end

    # edges
    F2 = map(coordinates) do co
        E´ = contract_PREPL(PR[_left_projector(co, nrows, ncols)...], EPL[co...])
        λ_E = dot(E[co...], E´)
        return E´ / λ_E - E[co...]
    end

    # halfinfinite environment
    F345 = map(coordinates) do co
        s´ = _contract_PR_PL(PR[co...], PLpart[co...])
        λs = dot(s[co...], s´)
        fp4 = s´ / λs - s[co...]

        co´ = _next_coordinate(co, nrows, ncols)
        fp3 = ((ULd[co...] * EiCiEPL[co...]) * iSfp[co...]) / λs - u[co...]
        fp5 = (iSfp[co...] * (_contract_PR_M(PR[co...], EC[co´...]) * VRd[co...])) / λs - v[co...]

        return fp3, fp4, fp5
    end
    F3 = map(x -> x[1], F345)
    F4 = map(x -> x[2], F345)
    F5 = map(x -> x[3], F345)

    return F1, F2, F3, F4, F5
end


# Combine into characteristic equations
# -------------------------------------

"""
    generate_halfinfinite_characteristic_equation(
        ::CTMRGAlgorithm{<:HalfInfiniteProjector},
        iSfp::CornerTensors,
        Ufp::LeftProjectors,
        Vfp::RightProjectors,
        ULfp::LeftProjectors,
        VRfp::RightProjectors,
    )

Takes the fixed-point values of the inverse singular values `iSfp`, the left and right isometries `Ufp`
and `Vfp`, and their null spaces `ULfp` and `VRfp` corresponding to a converged CTMRG contraction,
and generates a function ``F(s, C, E, u, S, v)`` which characterizes the convergence of the CTMRG algorithm in terms of the characteristic equation ``F(s, C, E, u, S, v) = 0``. Here, ``s`` corresponds to a
state variable (e.g. an `InfinitePEPS` that is being optimized), and ``(C, E, u, S, v)`` represents a CTMRG
contraction environment on a generic unit cell meaning that all tensors have a directional and
unit cell index. ``C`` and ``E`` directly represent the corner and edge tensors, while ``u`` and ``v``
parametrize differentiable projectors ``U = U_{fp} + U_{L,fp} u`` and ``V = V_{fp} + V_{L,fp} V``, and ``S`` denotes the singular values of the decomposed environment.

``F`` returns a tuple of five tensor arrays, corresponding to equations for ``C``, ``E``, ``u``, ``S`` and
``v``, respectively, as shown in Eqs. (76)-(80) in [arXiv:2607.15030](@cite burgelman_implicit_2026).
"""
function generate_halfinfinite_characteristic_equation(
        iSfp::CornerTensors,
        Ufp::LeftProjectors,
        Vfp::RightProjectors,
        ULfp::LeftProjectors,
        VRfp::RightProjectors,
    )

    iSfp = real.(DiagonalTensorMap.(iSfp)) # use as constant preconditioner?
    coordinates = eachcoordinate(iSfp)
    nrows, ncols = size(iSfp)[2:3]

    # the main routine which uses both the singular values and their inverses
    function asymmetric_characteristic_equation(state, C, E, u, s, v)
        ## Prepare all the objects we need in the right parametrization
        is = map(inv, s)

        # outspace variation parametrization of isometries
        U = map(coordinates) do co
            return Ufp[co...] + ULfp[co...] * u[co...]
        end
        V = map(coordinates) do co
            return Vfp[co...] + v[co...] * VRfp[co...]
        end

        isqsR = map(fourthroot, adjoint.(is) .* is) # root that goes into the left projector
        isqsL = map(fourthroot, is .* adjoint.(is)) # root that goes into the right projector

        # pre-dagger the isometries, absorb the square roots
        Ud = map(coordinates) do co
            co′ = _leftvec_invfroot_indices(co, nrows, ncols)
            absorb_right(U[co...]', isqsR[co′...])
        end
        Vd = map(coordinates) do co
            co′ = _rightvec_invfroot_indices(co, nrows, ncols)
            absorb_left(V[co...]', isqsL[co′...])
        end
        ULd = map(coordinates) do co
            co′ = _leftvec_invfroot_indices(co, nrows, ncols)
            absorb_right(ULfp[co...]', isqsR[co′...])
        end
        VRd = map(coordinates) do co
            co′ = _rightvec_invfroot_indices(co, nrows, ncols)
            absorb_left(VRfp[co...]', isqsL[co′...])
        end

        # pre-contract full inverses into corners from both sides
        iCi = map(coordinates) do co
            co′ = _prev_coordinate(co, nrows, ncols)
            return is[co′...] * C[co...] * is[co...]
        end

        ## Perform the actual contractions
        F1, F2, F3, F4, F5 = contract_halfinfinite_characteristic_equation(
            C, E,
            is, s,
            u, v,
            InfiniteSquareNetwork(state),
            Ud, Vd,
            iCi,
            ULd, VRd,
            iSfp,
        )

        return F1, F2, F3, F4, F5
    end

    return asymmetric_characteristic_equation
end

function generate_fullinfinite_characteristic_equation(
        iSfp::CornerTensors,
        Ufp::LeftProjectors,
        Vfp::RightProjectors,
        ULfp::LeftProjectors,
        VRfp::RightProjectors,
    )

    throw(ArgumentError("Characteristic equations for CTMRGAlgorithm{<:FullInfiniteProjector} are not yet implemented."))
end
