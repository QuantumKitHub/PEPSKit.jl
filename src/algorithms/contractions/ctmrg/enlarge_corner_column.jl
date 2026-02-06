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
        C_northwest::CTMRGCornerTensor, edge::CTMRGEdgeTensor
    )
    pC = (codomainind(C_northwest), domainind(C_northwest))
    pE = ((codomainind(edge)[1],), (codomainind(edge)[2:end]..., domainind(edge)...))
    pCE = (codomainind(edge), domainind(edge))
    return tensorcontract(C_northwest, pC, false, edge, pE, false, pCE)
end

# TODO: Other column-enlarged corners when QR-CTMRG for arbitrary unit cell is possible
