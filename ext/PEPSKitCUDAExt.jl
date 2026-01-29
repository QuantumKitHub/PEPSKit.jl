module PEPSKitCUDAExt

using PEPSKit, CUDA, cuTENSOR, Random
import CUDA: rand as curand, rand! as curand!, randn as curandn, randn! as curandn!

using PEPSKit.TensorKit
import PEPSKit: PEPSTensor, _corner_tensor, _edge_tensor

function PEPSTensor(
        f::typeof(rand),
        ::Type{TA},
        Pspace::S,
        Nspace::S, Espace::S = Nspace, Sspace::S = Nspace', Wspace::S = Espace',
    ) where {S <: ElementarySpace, TA <: CuArray}
    return curand(eltype(TA), Pspace ← Nspace ⊗ Espace ⊗ Sspace ⊗ Wspace)
end

function PEPSTensor(
        f::typeof(randn),
        ::Type{TA},
        Pspace::S,
        Nspace::S, Espace::S = Nspace, Sspace::S = Nspace', Wspace::S = Espace',
    ) where {S <: ElementarySpace, TA <: CuArray}
    return curandn(eltype(TA), Pspace ← Nspace ⊗ Espace ⊗ Sspace ⊗ Wspace)
end

function _corner_tensor(
        f::typeof(rand), ::Type{TA}, left_vspace::S, right_vspace::S = left_vspace
   ) where {T, TA <: CuArray{T}, S <: ElementarySpace}
    return curand(T, left_vspace ← right_vspace)
end

function _edge_tensor(
      f::typeof(randn), ::Type{TA}, left_vspace::S, pspaces::P, right_vspace::S = left_vspace
   ) where {T, TA <: CuArray{T}, S <: ElementarySpace, P <: ProductSpace}
    return curandn(T, left_vspace ⊗ pspaces, right_vspace)
end

end
