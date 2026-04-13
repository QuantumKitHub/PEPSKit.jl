# Column-enlarged corner contractions
# ----------------------------

# Northwest corner
# ----------------
"""
$(SIGNATURES)

Contract the half-enlarged northwest corner of the CTMRG environment.
```
    C₁-←-E₁-←-
    ↓    |
```
"""
@generated function column_enlarge_northwest_corner(
        C_northwest::CTMRGCornerTensor, edge::CTMRGEdgeTensor{T, S, N}
    ) where {T, S, N}
    CE_e = tensorexpr(:CE, -(1:N), -(N + 1))
    C_e = tensorexpr(:C_northwest, -1, 1)
    E_e = tensorexpr(:edge, (1, -(2:N)...), -(N + 1))
    return macroexpand(@__MODULE__, :(return @tensor $CE_e := $C_e * $E_e))
end

# TODO: Other column-enlarged corners when QR-CTMRG for arbitrary unit cell is possible
