"""
Get identity operator on the physical space
"""
function _getid_from_rho(rho::AbstractTensorMap)
    Pspace = codomain(rho)[1]
    if isdual(Pspace)
        Pspace = adjoint(Pspace)
    end
    return TensorKit.id(Pspace)
end

"""
Measure `<op>` using 1-site rho
"""
function meas_site(op::AbstractTensorMap, rho1::AbstractTensorMap)
    Id = _getid_from_rho(rho1)
    val = ncon((rho1, op), ([1, 2], [1, 2]))
    nrm = ncon((rho1, Id), ([1, 2], [1, 2]))
    meas = first(blocks(val / nrm))[2][1]
    return meas
end

"""
Measure `<op1 op2>` using 2-site rho
"""
function meas_bond(op1::AbstractTensorMap, op2::AbstractTensorMap, rho2::AbstractTensorMap)
    return meas_bond(op1 ⊗ op2, rho2)
end

"""
Measure `<gate>` using 2-site rho
"""
function meas_bond(gate::AbstractTensorMap, rho2::AbstractTensorMap)
    Id = _getid_from_rho(rho2)
    val = ncon((rho2, gate), ([1, 2, 3, 4], [1, 2, 3, 4]))
    nrm = ncon((rho2, Id ⊗ Id), ([1, 2, 3, 4], [1, 2, 3, 4]))
    meas = first(blocks(val / nrm))[2][1]
    return meas
end
