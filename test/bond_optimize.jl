using AppleAccelerate
using TensorKit
using PEPSKit
using Random
using Printf
using LinearAlgebra
using KrylovKit
using BenchmarkTools

# test
Random.seed!(10)
trscheme = truncerr(1e-10) & truncdim(10)
Vext, Vint, Vphy = ℂ^200, ℂ^14, ℂ^3
# Vext = ℂ[FermionParity](0 => 100, 1 => 100)
# Vint = ℂ[FermionParity](0 => 7, 1 => 7)
# Vphy = ℂ[FermionParity](0 => 1, 1 => 2)
# Vext = ℂ[FermionNumber](0 => 80, 1 => 40, -1 => 50)
# Vint = ℂ[FermionNumber](0 => 6, 1 => 4, -1 => 3)
# Vphy = ℂ[FermionNumber](0 => 1, 1 => 2)
Vbond = Vint ⊗ Vint
Z = TensorMap(randn, Float64, Vext, Vbond)
env = Z' * Z
aR2bL2 = Tensor(randn, Float64, Vint * Vphy * Vphy * Vint)

for alg in (
    ALSTruncation(; trscheme, maxiter=10, verbose=true, check_int=1),
    FullEnvTruncation(; trscheme, maxiter=10, verbose=true, check_int=1),
)
    aR, s, bL = tsvd(aR2bL2, ((1, 2), (3, 4)); trunc=truncerr(1e-14))
    aR, bL = PEPSKit.absorb_s(aR, s, bL)
    aR, bL = permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
    aR, s, bL, info = PEPSKit.bond_optimize(env, aR, bL, alg)
    println(space(s, 1))
    println(info)
end
