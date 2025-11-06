using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using KrylovKit
using Random

Random.seed!(100)
Nr, Nc = 2, 2
# create Hubbard iPEPS using simple update
function get_hubbard_state(t::Float64 = 1.0, U::Float64 = 8.0)
    H = hubbard_model(ComplexF64, Trivial, U1Irrep, InfiniteSquare(Nr, Nc); t, U, mu = U / 2)
    Vphy = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    peps = InfinitePEPS(rand, ComplexF64, Vphy, Vphy; unitcell = (Nr, Nc))
    wts = SUWeight(peps)
    alg = SimpleUpdate(;
        trunc = truncerror(; atol = 1.0e-10) & truncrank(4), check_interval = 2000
    )
    evolver = TimeEvolver(peps, H, 1.0e-2, 10000, alg, wts; tol = 1.0e-8)
    peps, = time_evolve(evolver)
    normalize!.(peps.A, Inf)
    return peps
end

peps = get_hubbard_state()
# calculate CTMRG environment
Envspace = Vect[FermionParity ⊠ U1Irrep](
    (0, 0) => 4, (1, 1 // 2) => 1, (1, -1 // 2) => 1, (0, 1) => 1, (0, -1) => 1
)
ctm_alg = SequentialCTMRG(; tol = 1.0e-10, verbosity = 2, trunc = truncerror(; atol = 1.0e-10) & truncrank(8))
env, = leading_boundary(CTMRGEnv(rand, ComplexF64, peps, Envspace), peps, ctm_alg)
for row in 1:Nr, col in 1:Nc
    cp1 = PEPSKit._next(col, Nc)
    A, B = peps.A[row, col], peps.A[row, cp1]
    X, a, b, Y = PEPSKit._qr_bond(A, B)
    benv = PEPSKit.bondenv_fu(row, col, X, Y, env)
    @assert [isdual(space(benv, ax)) for ax in 1:numind(benv)] == [0, 0, 1, 1]
    Z = PEPSKit.positive_approx(benv)
    # verify that gauge fixing can greatly reduce
    # condition number for physical state bond envs
    cond1 = cond(Z' * Z)
    Z2, a2, b2, (Linv, Rinv) = PEPSKit.fixgauge_benv(Z, a, b)
    benv2 = Z2' * Z2
    cond2 = cond(benv2)
    @test 1 <= cond2 < cond1
    @info "benv cond number: (gauge-fixed) $(cond2) ≤ $(cond1) (initial)"
    # verify gauge fixing is done correctly
    @tensor half[:] := Z[-1; 1 3] * a[1; -2 2] * b[2 -3; 3]
    @tensor half2[:] := Z2[-1; 1 3] * a2[1; -2 2] * b2[2 -3; 3]
    @test half ≈ half2
end
