using Test
using TensorKit
import MPSKitModels: S_zz, σˣ
using PEPSKit
using Printf
using Accessors: @set

const hc = 3.044382
const formatter = Printf.Format("t = %.2f, ⟨σˣ⟩ = %.7e + %.7e im.")
# real time evolution of ⟨σx⟩
# benchmark data from Physical Review B 104, 094411 (2021)
# Figure 6(a) calculated with D = 8 and χ = 32
const data = [
    # 0.01 9.9920027e-1
    0.06 9.7274912e-1
    0.11 9.1973182e-1
    0.16 8.6230618e-1
    0.21 8.1894325e-1
    0.26 8.0003708e-1
    0.31 8.0081082e-1
    0.36 8.0979257e-1
    # 0.41 8.1559623e-1
    # 0.46 8.1541661e-1
    # 0.51 8.1274128e-1
]

# the fully polarized state
peps0 = InfinitePEPS(zeros, ComplexF64, ℂ^2, ℂ^1; unitcell = (2, 2))
for t in peps0.A
    t[1, 1, 1, 1, 1] = 1.0
    t[2, 1, 1, 1, 1] = 1.0
end
lattice = collect(space(t, 1) for t in peps0.A)

# Hamiltonian
op = LocalOperator(lattice, ((1, 1),) => σˣ())
ham = transverse_field_ising(ComplexF64, Trivial, InfiniteSquare(2, 2); J = 1.0, g = hc)

# truncation strategy
Dcut, chi = 4, 16
trunc_peps = truncerror(; atol = 1.0e-10) & truncrank(Dcut)
trunc_env = truncerror(; atol = 1.0e-10) & truncrank(chi)

ctm_alg = SequentialCTMRG(;
    tol = 1.0e-8, maxiter = 50, verbosity = 2,
    trunc = trunc_env, projector_alg = :fullinfinite
)

interval = 5
ntu_alg = NeighbourUpdate(;
    opt_alg = FullEnvTruncation(; trunc = trunc_peps, tol = 1.0e-10),
    bondenv_alg = NNEnv(), imaginary_time = false
)

# do one step of NTU to match benchmark data
peps0, = time_evolve(peps0, ham, 0.01, 6, ntu_alg)
@info "Space of `peps0[1, 1]` = $(space(peps0[1, 1]))."
env0 = CTMRGEnv(ones, ComplexF64, peps0, ℂ^1)
env0, = leading_boundary(env0, peps0, ctm_alg)
# measure magnetization
magx = expectation_value(peps0, op, env0)
@info Printf.format(formatter, 0.06, real(magx), imag(magx))
@test isapprox(magx, data[1, 2]; atol = 0.005)

@testset "Neigborhood tensor update" begin
    peps, env = deepcopy(peps0), deepcopy(env0)
    count = 2
    evolver = TimeEvolver(peps, ham, 0.01, 30, ntu_alg; t0 = 0.06)
    spaces0 = collect(space(t) for t in peps.A)
    for (peps, info) in evolver
        !(evolver.state.iter % interval == 0) && continue
        spaces = collect(space(t) for t in peps.A)
        if spaces0 == spaces
            env, = leading_boundary(env, peps, ctm_alg)
        else
            env = complex(CTMRGEnv(info.wts))
            env, = leading_boundary(env, peps, ctm_alg)
        end
        # monitor the growth of env dimension
        corner = env.corners[1, 1, 1]
        corner_dim = dim.(space(corner, ax) for ax in 1:numind(corner))
        @info "Dimension of env.corner[1, 1, 1] = $(corner_dim)."
        # measure magnetization
        magx = expectation_value(peps, op, env)
        @info Printf.format(formatter, info.t, real(magx), imag(magx))
        @test isapprox(magx, data[count, 2]; atol = 0.005)
        count += 1
        spaces0 = spaces
    end
end
