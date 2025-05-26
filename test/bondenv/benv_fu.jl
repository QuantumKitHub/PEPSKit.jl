using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Nr, Nc = 2, 3
# create random PEPS
Random.seed!(0)
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Nspace = Vect[FermionParity](0 => 2, 1 => 2)
peps = InfinitePEPS(randn, ComplexF64, Pspace, Nspace; unitcell=(Nr, Nc))
normalize!.(peps.A, Inf)
# calculate CTMRG environment
Envspace = Vect[FermionParity](0 => 3, 1 => 3)
ctm_alg = SequentialCTMRG(; tol=1e-10, verbosity=2, trscheme=FixedSpaceTruncation())
env, = leading_boundary(CTMRGEnv(rand, ComplexF64, peps, Envspace), peps, ctm_alg)
for row in 1:Nr, col in 1:Nc
    cp1 = PEPSKit._next(1, Nc)
    A, B = peps.A[row, col], peps.A[row, cp1]
    X, a, b, Y = PEPSKit._qr_bond(A, B)
    # verify that gauge fixing can reduce condition number
    benv = PEPSKit.bondenv_fu(row, col, X, Y, env)
    @assert [isdual(space(benv, ax)) for ax in 1:numind(benv)] == [0, 0, 1, 1]
    @tensor ab[DX DY; da db] := a[DX; da D] * b[D db; DY]
    nrm1 = PEPSKit.inner_prod(benv, ab, ab)
    # gauge fixing
    Z = PEPSKit.positive_approx(benv)
    cond = PEPSKit._condition_number(Z' * Z)
    Z2, a2, b2, (Linv, Rinv) = PEPSKit.fixgauge_benv(Z, a, b)
    benv2 = Z2' * Z2
    cond2 = PEPSKit._condition_number(benv2)
    @test 1 <= cond2 < cond
    @info "benv cond number: (gauge-fixed) $(cond2) ≤ $(cond) (initial)"
end
