const BondEnv{T, S} = AbstractTensorMap{T, S, 2, 2} where {T <: Number, S <: ElementarySpace}
const BondEnv3site{T, S} = AbstractTensorMap{T, S, 4, 4} where {T <: Number, S <: ElementarySpace}
const Hair{T, S} = AbstractTensor{T, S, 2} where {T <: Number, S <: ElementarySpace}
# Orthogonal tensors obtained PEPSTensor/PEPOTensor
# with one physical leg factored out by `_qr_bond`
const PEPSOrth{T, S} = AbstractTensor{T, S, 4} where {T <: Number, S <: ElementarySpace}
const PEPOOrth{T, S} = AbstractTensor{T, S, 5} where {T <: Number, S <: ElementarySpace}

"Convert tensor `t` connected by the bond to be truncated to a `PEPSTensor`."
_prepare_site_tensor(t::PEPSTensor) = t
_prepare_site_tensor(t::PEPOTensor) = first(fuse_physicalspaces(t))
_prepare_site_tensor(t::PEPSOrth) = permute(insertleftunit(t, 1), ((1,), (2, 3, 4, 5)))
_prepare_site_tensor(t::PEPOOrth) = permute(t, ((1,), (2, 3, 4, 5)))
