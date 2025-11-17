using Test
using TensorKit
import MPSKitModels: S_zz, σˣ
using PEPSKit
using Printf
using Accessors: @set

const hc = 3.044382
const formatter = Printf.Format("t = %.2f, ⟨σˣ⟩ = %.7e + %.7e im.")
# real time evolution of ⟨σx⟩
# benchmark data from Physical Review B 104, 094411 (2021) Figure 6(a)
# calculated with D = 8 and χ = 4D = 32
const data = [
    0.01 9.9920027e-1
    0.06 9.7274912e-1
    0.11 9.1973182e-1
    0.16 8.6230618e-1
    0.21 8.1894325e-1
    0.26 8.0003708e-1
    0.31 8.0081082e-1
    # 0.36 8.0979257e-1
    # 0.41 8.1559623e-1
    # 0.46 8.1541661e-1
    # 0.51 8.1274128e-1
]

# redefine tfising Hamiltonian with only 2-site gate
function tfising(
        T::Type{<:Number},
        S::Union{Type{Trivial}, Type{Z2Irrep}},
        lattice::InfiniteSquare;
        J = 1.0,
        g = 1.0,
    )
    ZZ = rmul!(4 * S_zz(T, S), -J)
    X = rmul!(σˣ(T, S), g * -J)
    unit = id(space(X, 1))
    gate = ZZ + (1 / 4) * (unit ⊗ X + X ⊗ unit)
    spaces = fill(domain(X)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces, (neighbor => gate for neighbor in PEPSKit.nearest_neighbours(lattice))...
    )
end

function tfising_fu(g::Float64, Dcut::Int, chi::Int; als = true, use_pinv = true)
    # the fully polarized state
    peps = InfinitePEPS(zeros, ComplexF64, ℂ^2, ℂ^1; unitcell = (2, 2))
    for t in peps.A
        t[1, 1, 1, 1, 1] = 1.0
        t[2, 1, 1, 1, 1] = 1.0
    end
    lattice = collect(space(t, 1) for t in peps.A)
    op = LocalOperator(lattice, ((1, 1),) => σˣ())
    ham = tfising(ComplexF64, Trivial, InfiniteSquare(2, 2); J = 1.0, g = g)

    trunc_peps = truncerror(; atol = 1.0e-10) & truncrank(Dcut)
    opt_alg = if als
        ALSTruncation(; trunc = trunc_peps, tol = 1.0e-10, use_pinv)
    else
        FullEnvTruncation(; trunc = trunc_peps, tol = 1.0e-10)
    end

    trunc_env = truncerror(; atol = 1.0e-10) & truncrank(chi)
    ctm_alg = SequentialCTMRG(;
        tol = 1.0e-8, maxiter = 50, verbosity = 2,
        trunc = trunc_env, projector_alg = :fullinfinite
    )

    env = CTMRGEnv(ones, ComplexF64, peps, ℂ^1)
    env, = leading_boundary(env, peps, ctm_alg)

    # do one extra step at the beginning to match benchmark data
    fu_alg = FullUpdate(; opt_alg, ctm_alg, imaginary_time = false, reconverge_interval = 5)
    evolver = TimeEvolver(peps, ham, 0.01, 30, fu_alg, env)
    peps, env, info = timestep(evolver, peps, env; reconverge_env = true)
    # ensure the recoverged environment is updated to the internal state of `evolver`
    @test env == evolver.state.env
    magx = expectation_value(peps, op, env)
    @info Printf.format(formatter, info.t, real(magx), imag(magx))
    @test isapprox(magx, data[1, 2]; atol = 0.005)

    # reset the number of performed iterations
    state0 = evolver.state
    evolver.state = (@set state0.iter = 0)
    # continue the remaining evolution
    count = 2
    for (peps, env, info) in evolver
        !evolver.state.reconverged && continue
        # monitor the growth of env dimension
        corner = env.corners[1, 1, 1]
        corner_dim = dim.(space(corner, ax) for ax in 1:numind(corner))
        @info "Dimension of env.corner[1, 1, 1] = $(corner_dim)."

        magx = expectation_value(peps, op, env)
        @info Printf.format(formatter, info.t, real(magx), imag(magx))
        @test isapprox(magx, data[count, 2]; atol = 0.005)
        count += 1
    end
    return nothing
end

tfising_fu(hc, 6, 24; als = false, use_pinv = true)
