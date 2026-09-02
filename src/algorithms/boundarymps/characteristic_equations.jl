#
# Characteristic equation used in implicit differentiation of symmetric boundary MPS
# contractions
#

"""
    generate_boundary_mps_characteristic_equation(
        env::SymmetricBoundaryMPSEnv, (VLfp, VRfp)::Tuple{<:EdgeTensor, <:RightProjector}
    )

Takes the fixed-point values of a converged symmetric boundary MPS contraction, given by the
environment `env`, along with the left null space `VLfp` of its left-gauged MPS tensor and
the right null space `VRfp` of its right-gauged MPS tensor, and generates a function
``F(s, l, r, c, gl, gr)`` which characterizes the convergence of the boundary MPS contraction
in terms of the characteristic equation ``F(s, l, r, c, gl, gr) = 0``.

Here, ``s`` corresponds to a state variable (e.g. an `InfinitePEPS` that is being optimized),
``c``, ``gl`` and ``gr`` directly represent the MPS bond tensor and the left and right
environments, while ``l`` and ``r`` parametrize differentiable left- and
right-gauged MPS tensors as ``AL = AL_{fp} + V_{L,fp} * l`` and
``AR = AR_{fp} + (r * V_{R,fp})``.

``F`` returns a tuple of five tensors, corresponding to an equation for each of ``l``, ``r``,
``c``, ``gl`` and ``gr``. The first three are obtained by projecting the effective site
operator applied to the center-gauged MPS tensor onto the respective tangent directions, and
the last two express that the environments are fixed points of the MPS-network-MPS transfer
matrix. All equations are normalized by the eigenvalue ``λ`` of the effective site operator.

See also [`generate_symmetric_characteristic_equation`](@ref).
"""
function generate_boundary_mps_characteristic_equation(
        env::SymmetricBoundaryMPSEnv, (VLfp, VRfp)::Tuple{TE, TP}
    ) where {TE <: EdgeTensor, TP <: RightProjector}
    ALfp, ARfp, Cfp, = _unpack(env)

    # constant preconditioner
    # NOTE: this relies on the bond tensor having been diagonalized, which
    # `leading_boundary` takes care of
    iCfp = sdiag_pow(real(DiagonalTensorMap(Cfp)), -1)

    function boundary_mps_characteristic_equation(state, l, r, c, gl, gr)
        network = InfiniteSquareNetwork(state)
        O = network[1, 1]

        # prepare appropriately parametrized MPS tensors
        AL = ALfp + VLfp * l
        AR = ARfp + repartition_left(r * VRfp)
        ARR = repartition_right(AR) # 'right isometry' form

        # construct the center tensor in a symmetric way
        AC = (absorb_right(AL, c) + absorb_left(AR, c)) / 2

        # main partial contractions to reuse
        AC´L = ∂AC(AC, gl, O, gr) # 'left isometry' form
        AC´R = repartition_right(AC´L) # 'right isometry' form
        λ = dot(AC, AC´L)

        F1 = VLfp' * AC´L * iCfp / λ - l
        F2 = iCfp * AC´R * VRfp' / λ - r
        F3 = (AL' * AC´L + AC´R * ARR') / (2 * λ) - c
        F4 = MPSKit.transfer_left(gl, O, AL, AL) / λ - gl
        F5 = MPSKit.transfer_right(gr, O, AR, AR) / λ - gr

        return F1, F2, F3, F4, F5
    end

    return boundary_mps_characteristic_equation
end
