# Shared helpers for the boundary MPS tests

# An approximation of the ground state of the Heisenberg model in its sublattice-rotated
# form, which has a uniform ground state on any unit cell, and the model. On cells of at
# least 2×2 this is a simple-update state. Simple update does not support a single-site cell,
# so a single-site state is obtained from a short C4v CTMRG optimization of a symmetrized
# random state instead, and is C4v-symmetric.
function heisenberg_state(AT, D, unitcell; seed, χ = 16, c4v_maxiter = 10)
    Random.seed!(seed)
    H = heisenberg_XYZ(InfiniteSquare(unitcell...))
    if unitcell == (1, 1)
        peps₀ = symmetrize!(InfinitePEPS(ComplexSpace(2), ComplexSpace(D)), RotateReflect())
        env₀, = leading_boundary(
            initialize_random_c4v_env(peps₀, ComplexSpace(χ)), peps₀;
            alg = :C4vCTMRG, tol = 1.0e-8, maxiter = 500, verbosity = 0,
        )
        peps, = fixedpoint(
            H, peps₀, env₀;
            boundary_alg = (; alg = :C4vCTMRG, maxiter = 500),
            gradient_alg = (; alg = :ImplicitGradient),
            optimizer_alg = (; tol = 1.0e-6, maxiter = c4v_maxiter),
            verbosity = 0,
        )
    else
        peps₀ = InfinitePEPS(randn, ComplexF64, ComplexSpace(2), ComplexSpace(D); unitcell)
        alg = SimpleUpdate(; trunc = truncrank(D), bipartite = false)
        peps, = time_evolve(peps₀, H, 1.0e-2, 2000, alg, SUWeight(peps₀); tol = 1.0e-8, verbosity = 0)
    end
    return adapt(AT, PEPSKit.peps_normalize(peps)), adapt(AT, H)
end

# a CTMRG environment grown from a product state, converged at environment dimension χ
function grown_ctmrg_env(psi, χ; tol)
    env₀ = initialize_ctmrg_environment(psi, ProductStateInitialization())
    env, = leading_boundary(
        env₀, psi; alg = :SimultaneousCTMRG, trunc = truncrank(χ), tol, maxiter = 1000, verbosity = 0
    )
    return env
end
