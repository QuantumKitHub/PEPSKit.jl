using Random
using Printf
using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit

function _postprocess(
    aR::AbstractTensorMap{S,2,1}, s::AbstractTensorMap{S,1,1}, bL::AbstractTensorMap{S,1,2}
) where {S<:ElementarySpace}
    aR, bL = PEPSKit.absorb_s(aR, s, bL)
    return permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
end

Random.seed!(10)
trscheme = truncerr(1e-10) & truncdim(10)
# Vext, Vint, Vphy = ℂ^200, ℂ^14, ℂ^3
Vext = ℂ[U1Irrep](0 => 90, 1 => 50, -1 => 60)
Vint = ℂ[U1Irrep](0 => 6, 1 => 2, -1 => 3)
Vphy = ℂ[U1Irrep](0 => 1, 1 => 2)
# Vext = ℂ[FermionParity](0 => 100, 1 => 100)
# Vint = ℂ[FermionParity](0 => 7, 1 => 7)
# Vphy = ℂ[FermionParity](0 => 1, 1 => 2)
Vbond = Vint ⊗ Vint
# random positive-definite environment
Z = TensorMap(randn, Float64, Vext, Vbond)
env = Z' * Z
# untruncated bond tensor
aR2bL2 = Tensor(randn, Float64, Vint * Vphy * Vphy * Vint)
aR2, s, bL2 = tsvd(aR2bL2, ((1, 2), (3, 4)))
aR2, bL2 = _postprocess(aR2, s, bL2)
# bond tensor (truncated SVD initialization)
aR0, s, bL0 = tsvd(aR2bL2, ((1, 2), (3, 4)); trunc=trscheme)
aR0, bL0 = _postprocess(aR0, s, bL0)
fid0 = PEPSKit.fidelity(env, PEPSKit._combine_aRbL(aR0, bL0), aR2bL2)
@info "SVD initial fidelity = $fid0."

for (label, alg) in (
    ("ALS", ALSTruncation(; trscheme, maxiter=10, verbose=true, check_int=1)),
    ("FET", FullEnvTruncation(; trscheme, maxiter=10, verbose=true, check_int=1)),
)
    local s
    aR1, s, bL1, info = PEPSKit.bond_optimize(env, aR2, bL2, alg)
    @info "$label improved fidelity = $(info.fid)."
    @test info.fid > fid0
end
