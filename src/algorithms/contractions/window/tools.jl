"""
Bundle the zip-up contraction and optional DMRG refinement
algorithms used after each finite window MPO-MPS contraction.
"""
struct WindowApprox{Z, D}
    zipup::Z
    dmrg::D
end

"""
Convert a south-boundary MPS tensor into the stored bra representation expected by `dot`.
"""
_bra_mps_tensor(A::GenericMPSTensor) = copy(permute(A', ((2, 3), (1,))))

"""
Construct the planar adjoint of a finite MPO while restoring MPSKit's local MPO leg partition.
"""
function _adjoint_mpo(W::FiniteMPO)
    return FiniteMPO(map(A -> transpose(A', ((3, 1), (4, 2)); copy = true), parent(W)))
end
