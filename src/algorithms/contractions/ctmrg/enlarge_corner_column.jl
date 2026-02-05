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
function column_enlarge_northwest_corner(
        C_northwest::CTMRGCornerTensor, edge::CTMRG_PEPS_EdgeTensor
    )
    return @tensor corner[χS Dt Db; χE] := C_northwest[χS; χ] * edge[χ Dt Db; χE]
end
function column_enlarge_northwest_corner(
        C_northwest::CTMRGCornerTensor, edge::CTMRG_PF_EdgeTensor
    )
    return @tensor corner[χS D; χE] := C_northwest[χS; χ] * edge[χ D; χE]
end
# TODO: PEPS-PEPO-PEPS sandwich

# TODO: Other column-enlarged corners when QR-CTMRG for arbitrary unit cell is possible
