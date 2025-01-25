using AppleAccelerate
using TensorKit
using PEPSKit
using Random
using Printf
using LinearAlgebra
using KrylovKit
using BenchmarkTools

"""
Contract the axis between `aR` and `bL` tensors
"""
function combine_aRbL(
    aR::AbstractTensor{S,3}, bL::AbstractTensor{S,3}
) where {S<:ElementarySpace}
    #= 
            da      db
            ↑       ↑
    ← DX ← aR ← D ← bL → DY →
    =#
    @tensor aRbL[DX, da, db, DY] := aR[DX, da, D] * bL[D, db, DY]
    return aRbL
end

# test
Random.seed!(10)
trscheme = truncerr(1e-10) & truncdim(10)
Vext, Vint, Vphy = ℂ^200, ℂ^14, ℂ^3 
# Vext = ℂ[FermionParity](0 => 50, 1 => 50)
# Vint = ℂ[FermionParity](0 => 5, 1 => 5)
Vbond = Vint ⊗ Vint
Z = TensorMap(randn, Float64, Vext, Vbond)
env = Z' * Z
aR2bL2 = Tensor(randn, Float64, Vint * Vphy * Vphy * Vint)

alg = FullEnvTruncation(; trscheme, maxiter=20, verbose=true, check_int=1)
aR, s, bL = tsvd(aR2bL2, ((1, 2), (3, 4)); trunc=truncerr(1e-15))
aR, bL = PEPSKit.absorb_s(aR, s, bL)
aR, bL = permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
# @btime bond_optimize(env, aR, bL, alg)
aR, s, bL, (cost, fid) = PEPSKit.bond_optimize(env, aR, bL, alg)
