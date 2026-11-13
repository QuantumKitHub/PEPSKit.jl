#
# Characteristic equations used in implicit differentiation of boundary MPS contractions
#

# All equations act on matrices of tensors in the positions of a `BoundaryMPSEnv`; the single
# boundary MPS of a C4v contraction is the 1×1 case.

# the inverses of the diagonal bond tensors of a fixed point, used as a constant
# preconditioner of the isometry equations
function _inverse_bonds(C)
    all(x -> all(b -> isdiag(last(b)), blocks(x)), C) ||
        throw(ArgumentError("the characteristic equations require diagonal bond tensors"))
    return map(x -> sdiag_pow(real(DiagonalTensorMap(x)), -1), C)
end

# The equations of the isometries and bond tensors of one boundary MPS. With
# `AC´[r, c] = ∂AC(AC[r - 1, c]; gl[r, c - 1], O[r, c], gr[r, c + 1])` and
# `λ[r, c] = ⟨AC[r, c], AC´[r, c]⟩`,
#
#     F1 = VL[r, c]' AC´[r, c] C[r, c]⁻¹ / λ[r, c] - l[r, c]
#     F2 = C[r, c - 1]⁻¹ AC´[r, c] VR[r, c]' / λ[r, c] - r[r, c]
#     F3 = (AL[r, c]' AC´[r, c] / λ[r, c] + AC´[r, c + 1] ARR[r, c + 1]' / λ[r, c + 1]) / 2 - c[r, c]
#
# Returns `(F1, F2, F3)` and `λ`.
function _center_equations(n, AL, AR, ARR, c, gl, gr, l, r, (VL, VR), iCfp; getsite)
    dims = size(AL)
    AC = _cellmap((i, j) -> _center_tensor(AL, AR, c, i, j), dims)
    AC´ = _cellmap(dims) do i, j
        return ∂AC(_at(AC, i - 1, j), _at(gl, i, j - 1), getsite(n, i, j), _at(gr, i, j + 1))
    end
    AC´R = map(repartition_right, AC´)
    λ = map(dot, AC, AC´)
    F1 = _cellmap(dims) do i, j
        return VL[i, j]' * AC´[i, j] * iCfp[i, j] / λ[i, j] - l[i, j]
    end
    F2 = _cellmap(dims) do i, j
        return _at(iCfp, i, j - 1) * AC´R[i, j] * VR[i, j]' / λ[i, j] - r[i, j]
    end
    F3 = _cellmap(dims) do i, j
        return (
            AL[i, j]' * AC´[i, j] / λ[i, j] +
                _at(AC´R, i, j + 1) * _at(ARR, i, j + 1)' / _at(λ, i, j + 1)
        ) / 2 - c[i, j]
    end
    return (F1, F2, F3), λ
end

"""
    generate_boundary_mps_characteristic_equation(
        (ALfp, ARfp, Cfp, GLfp, GRfp), (VLfp, VRfp); getsite = getindex
    )

Takes the fixed-point values of a boundary MPS and its environments, given as matrices of
tensors in the positions of the top MPS of a [`BoundaryMPSEnv`](@ref) with diagonal bond
tensors, together with the left null spaces `VLfp` of its left-gauged tensors and the right
null spaces `VRfp` of its right-gauged tensors, and generates a function ``F(n, l, r, c, gl, gr)`` which characterizes the
convergence of the boundary MPS contraction of the network ``n`` in terms of the
characteristic equation ``F(n, l, r, c, gl, gr) = 0``.

Here, ``c``, ``gl`` and ``gr`` directly represent the bond tensors and the left and right
environments, while ``l`` and ``r`` parametrize differentiable left- and right-gauged MPS
tensors as ``A_L = A_{L,fp} + V_{L,fp} l`` and ``A_R = A_{R,fp} + r V_{R,fp}``.

``F`` returns a tuple of five matrices of tensors, corresponding to an equation for each of
``l``, ``r``, ``c``, ``gl`` and ``gr`` on every site. The first three project the effective
site operator applied to the center-gauged MPS tensor onto the respective tangent directions,
and the last two express that the environments are fixed points of the MPS-network-MPS
transfer matrices,
```
    F4[r, c] = T_L(gl[r, c - 1]; O[r, c], AL[r - 1, c], AL[r, c]) / λ[r, c] - gl[r, c]
    F5[r, c] = T_R(gr[r, c + 1]; O[r, c], AR[r - 1, c], AR[r, c]) / λ[r, c] - gr[r, c]
```
All equations are normalized by the eigenvalue ``λ[r, c]`` of the effective site operator,
which is also that of the transfer matrices through site ``(r, c)`` when the environments are
normalized per bond, as MPSKit does.

The local sandwiches are read out of ``n`` through `getsite`, so a caller can choose whether
they are differentiated at all; see [`PEPSKit.constant_site`](@ref).
"""
function generate_boundary_mps_characteristic_equation(
        (ALfp, ARfp, Cfp, GLfp, GRfp), (VLfp, VRfp); getsite = getindex,
    )
    iCfp = _inverse_bonds(Cfp)

    function boundary_mps_characteristic_equation(n, l, r, c, gl, gr)
        # prepare appropriately parametrized MPS tensors
        AL = map((A, V, x) -> A + V * x, ALfp, VLfp, l)
        AR = map((A, V, x) -> A + repartition_left(x * V), ARfp, VRfp, r)
        ARR = map(repartition_right, AR) # 'right isometry' form

        (F1, F2, F3), λ = _center_equations(
            n, AL, AR, ARR, c, gl, gr, l, r, (VLfp, VRfp), iCfp; getsite
        )
        F4 = _cellmap(size(AL)) do i, j
            gl´ = MPSKit.transfer_left(
                _at(gl, i, j - 1), getsite(n, i, j), _at(AL, i - 1, j), AL[i, j]
            )
            return gl´ / λ[i, j] - gl[i, j]
        end
        F5 = _cellmap(size(AL)) do i, j
            gr´ = MPSKit.transfer_right(
                _at(gr, i, j + 1), getsite(n, i, j), _at(AR, i - 1, j), AR[i, j]
            )
            return gr´ / λ[i, j] - gr[i, j]
        end
        return F1, F2, F3, F4, F5
    end

    return boundary_mps_characteristic_equation
end

"""
    generate_mixed_environment_characteristic_equation(
        env::BoundaryMPSEnv, (VLt, VRt, VLb, VRb); getsite = getindex
    )

Takes a converged asymmetric boundary MPS environment `env`, together with the null spaces of
the left- and right-gauged tensors of its top and bottom MPS, and generates a function
``F(n, l_t, r_t, c_t, l_b, r_b, c_b, gl, gr)`` which characterizes its mixed environments
``gl`` and ``gr`` through ``F = 0``, with the top and bottom MPS parametrized as in
[`generate_boundary_mps_characteristic_equation`](@ref). With
```
    gl´[r, c] = E_L(gl[r, c - 1]; O[r, c], ALt[r - 1, c], ARb[r + 1, c])
    λ[r, c]   = tr(cb[r + 1, c] ∂C(ct[r - 1, c]; gl´[r, c], gr[r, c + 1]))
```
it returns the two equations
```
    F1[r, c] = gl´[r, c] / λ[r, c] - gl[r, c]
    F2[r, c] = E_R(gr[r, c + 1]; O[r, c], ARt[r - 1, c], ALb[r + 1, c]) / λ[r, c] - gr[r, c]
```
where ``λ[r, c]`` is the eigenvalue of both mixed transfer matrices through site ``(r, c)``
when the mixed environments are normalized as in [`PEPSKit.mixed_environments`](@ref).
"""
function generate_mixed_environment_characteristic_equation(
        env::BoundaryMPSEnv, (VLt, VRt, VLb, VRb); getsite = getindex
    )
    function mixed_environment_characteristic_equation(n, lt, rt, ct, lb, rb, cb, gl, gr)
        # prepare appropriately parametrized MPS tensors
        ALt = map((A, V, x) -> A + V * x, env.ALt, VLt, lt)
        ARt = map((A, V, x) -> A + repartition_left(x * V), env.ARt, VRt, rt)
        ALb = map((A, V, x) -> A + V * x, env.ALb, VLb, lb)
        ARb = map((A, V, x) -> A + repartition_left(x * V), env.ARb, VRb, rb)
        dims = size(ALt)
        gl´ = _cellmap(dims) do i, j
            return edge_transfer_left(
                _at(gl, i, j - 1), getsite(n, i, j), _at(ALt, i - 1, j), _at(ARb, i + 1, j)
            )
        end
        λ = _cellmap(dims) do i, j
            return _mixed_bond_value(_at(cb, i + 1, j), _at(ct, i - 1, j), gl´[i, j], _at(gr, i, j + 1))
        end
        F1 = map((x, μ, y) -> x / μ - y, gl´, λ, gl)
        F2 = _cellmap(dims) do i, j
            gr´ = edge_transfer_right(
                _at(gr, i, j + 1), getsite(n, i, j), _at(ARt, i - 1, j), _at(ALb, i + 1, j)
            )
            return gr´ / λ[i, j] - gr[i, j]
        end
        return F1, F2
    end

    return mixed_environment_characteristic_equation
end

"""
    generate_full_boundary_mps_characteristic_equation(
        env::BoundaryMPSEnv, (VLt, VRt, VLb, VRb); getsite = getindex
    )

Takes a converged asymmetric boundary MPS environment `env`, together with the null spaces of
the left- and right-gauged tensors of its top and bottom MPS, and generates a function
``F(n, l_t, r_t, c_t, gl_t, gr_t, l_b, r_b, c_b, gl_b, gr_b, gl, gr)`` which characterizes
its convergence through ``F = 0``. It returns the twelve equations of the top MPS on the
network, of the bottom MPS on the network rotated by 180 degrees (with the variables and
equations mapped to the original frame by [`PEPSKit.rotate_bottom`](@ref)) and of the mixed
environments; see [`generate_boundary_mps_characteristic_equation`](@ref) and
[`generate_mixed_environment_characteristic_equation`](@ref).
"""
function generate_full_boundary_mps_characteristic_equation(
        env::BoundaryMPSEnv, (VLt, VRt, VLb, VRb); getsite = getindex
    )
    F_top = generate_boundary_mps_characteristic_equation(_top(env), (VLt, VRt); getsite)
    F_bottom = generate_boundary_mps_characteristic_equation(
        _bottom(env), (_rotate_sites(VLb), _rotate_sites(VRb)); getsite
    )
    F_mixed = generate_mixed_environment_characteristic_equation(env, (VLt, VRt, VLb, VRb); getsite)

    function full_boundary_mps_characteristic_equation(
            n, lt, rt, ct, glt, grt, lb, rb, cb, glb, grb, gl, gr,
        )
        F_b = F_bottom(rot180(n), rotate_bottom((lb, rb, cb, glb, grb))...)
        return (
            F_top(n, lt, rt, ct, glt, grt)...,
            rotate_bottom(F_b)...,
            F_mixed(n, lt, rt, ct, lb, rb, cb, gl, gr)...,
        )
    end

    return full_boundary_mps_characteristic_equation
end

# The largest residual of every characteristic equation over the unit cell. The isometry
# equations carry the inverse bond tensors, so near a fixed point their residuals are of the
# order of the convergence error divided by the smallest Schmidt value. To compare them with
# the tolerance, this preconditioner is undone by multiplying them with the bond tensors.
# `bonds` lists, for every boundary MPS, the positions of its two isometry equations in `FS`,
# its bond tensors and the map of its equations to the frame of these bond tensors.
function _characteristic_residuals(FS, bonds)
    nrms = Any[map(norm, F) for F in FS]
    for (il, ir, C, frame) in bonds
        Fl, Fr = frame(FS[il]), frame(FS[ir])
        nrms[il] = map((f, c) -> norm(f * c), Fl, C)
        nrms[ir] = _cellmap((r, c) -> norm(_at(C, r, c - 1) * Fr[r, c]), size(C))
    end
    return map(maximum, Tuple(nrms))
end
function _check_characteristic_residuals(FS, tol, bonds)
    F_nrms = _characteristic_residuals(FS, bonds)
    any(F_nrms .> tol) &&
        @warn "Characteristic equations not satisfied, still using the gradient: $F_nrms"
    return nothing
end
