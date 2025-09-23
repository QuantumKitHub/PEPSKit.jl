using Test
using TensorKit
import MPSKitModels: S_zz, σˣ
using PEPSKit
using Printf
using Random
Random.seed!(0)

const hc = 3.044382
const formatter = Printf.Format("t = %.2f, ⟨σˣ⟩ = %.7e + %.7e im. Time = %.3f s")
# real time evolution of ⟨σx⟩
# benchmark data from Physical Review B 104, 094411 (2021) Figure 6(a)
# calculated with D = 8 and χ = 4 D = 32
const data = [
    0.01 9.9920027e-1
    0.06 9.7274912e-1
    0.11 9.1973182e-1
    0.16 8.6230618e-1
    0.21 8.1894325e-1
    0.26 8.0003708e-1
    0.31 8.0081082e-1
    0.36 8.0979257e-1
    0.41 8.1559623e-1
    0.46 8.1541661e-1
    0.51 8.1274128e-1
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

function tfising_fu(g::Float64, maxiter::Int, Dcut::Int, chi::Int; als = true, use_pinv = false)
    # the fully polarized state
    peps = InfinitePEPS(randn, ComplexF64, 2, 1; unitcell = (2, 2))
    for t in peps.A
        t[1, 1, 1, 1, 1] = 1.0
        t[2, 1, 1, 1, 1] = 1.0
    end
    lattice = collect(space(t, 1) for t in peps.A)
    op = LocalOperator(lattice, ((1, 1),) => σˣ())
    ham = tfising(ComplexF64, Trivial, InfiniteSquare(2, 2); J = 1.0, g = g)

    trscheme_peps = truncerr(1.0e-10) & truncdim(Dcut)
    trscheme_env = truncerr(1.0e-10) & truncdim(chi)
    env = CTMRGEnv(rand, ComplexF64, peps, chi)
    env, = leading_boundary(env, peps; tol = 1.0e-10, verbosity = 2, trscheme = trscheme_env)

    ctm_alg = SequentialCTMRG(;
        tol = 1.0e-9,
        maxiter = 50,
        verbosity = 2,
        trscheme = trscheme_env,
        projector_alg = :fullinfinite,
    )
    opt_alg = ALSTruncation(; trscheme = trscheme_peps, tol = 1.0e-10, use_pinv = true)
    # opt_alg = FullEnvTruncation(; trscheme=trscheme_peps, tol=1e-10)

    # do one extra step at the beginning to match benchmark data
    t = 0.01
    fu_alg = FullUpdate(; dt = 0.01im, niter = 1, opt_alg, ctm_alg)
    time0 = time()
    peps, env, = fu_iter2(peps, ham, fu_alg, env)
    magx = expectation_value(peps, op, env)
    time1 = time()
    @info Printf.format(formatter, t, real(magx), imag(magx), time1 - time0)
    @test isapprox(magx, data[1, 2]; atol = 0.005)

    fu_alg = FullUpdate(; dt = 0.01im, niter = 5, opt_alg, ctm_alg)
    for count in 1:maxiter
        time0 = time()
        peps, env, = fu_iter2(peps, ham, fu_alg, env)
        normalize!.(peps.A, Inf)
        magx = expectation_value(peps, op, env)
        time1 = time()
        t += 0.05
        @info Printf.format(formatter, t, real(magx), imag(magx), time1 - time0)
        @test isapprox(magx, data[count + 1, 2]; atol = 0.005)
        flush(stdout)
        flush(stderr)
    end
    return nothing
end

tfising_fu(hc, 10, 6, 24; als = false, use_pinv = false)
