using Test
using Random
using LinearAlgebra
using PEPSKit
using Adapt
using MPSKit
using TensorKit

include("utility.jl")

## Test observables evaluated with a C4v boundary MPS environment against CTMRG
# -----------------------------------------------------------------------------------
# The boundary MPS environment absorbs the corners into its gauge center and acts as its own
# bra, so it reaches the generic `expectation_value` machinery through a different set of
# contractions than CTMRG does. The state approximates the ground state of the Heisenberg model
# through a short C4v CTMRG optimization, and the boundary MPS is initialized from a CTMRG
# environment grown from a product state.

sd = 42039482049
D = 3
χ = 16
boundary_tol = 1.0e-10
boundary_maxiter = 500
ctmrg_rtol = 1.0e-4 # boundary MPS and CTMRG at the same χ differ by a truncation error
patch_rtol = 1.0e-10 # fast paths against the generic patch contraction
mps_algs = [:VUMPS, :VOMPS]

## Tests
# ------
function boundary_mps_c4v_observables(AT)
    psi, H = heisenberg_state(AT, D, (1, 1); seed = sd, χ)
    # next-nearest-neighbour terms are evaluated through the generic patch contraction
    H_j1j2 = adapt(AT, j1_j2_model(InfiniteSquare(); J2 = 0.5))

    ctmrg_env = grown_ctmrg_env(psi, χ; tol = boundary_tol)
    N_ref = norm(psi, ctmrg_env)
    E_ref = expectation_value(psi, H, ctmrg_env)
    E_j1j2_ref = expectation_value(psi, H_j1j2, ctmrg_env)

    return @testset "C4v boundary MPS Heisenberg observables, mps_alg=:$mps_alg ($AT)" for mps_alg in mps_algs
        boundary_alg = C4vBoundaryMPS(;
            mps_alg, tol = boundary_tol, maxiter = boundary_maxiter, verbosity = 0,
        )
        env, info = leading_boundary(C4vBoundaryMPSEnv(ctmrg_env, psi), psi, boundary_alg)
        @test info.converged

        # the network value normalizing the expectation value must agree first
        @test norm(psi, env) ≈ N_ref rtol = ctmrg_rtol

        E = expectation_value(psi, H, env)
        @info "C4v boundary MPS (mps_alg=:$mps_alg) vs CTMRG energy" E E_ref
        @test E ≈ E_ref rtol = ctmrg_rtol
        @test expectation_value(psi, H_j1j2, env) ≈ E_j1j2_ref rtol = ctmrg_rtol

        # `cost_function` is the real part of the same quantity, and is what the optimizer sees
        @test cost_function(psi, env, H) ≈ real(E) rtol = 1.0e-12

        # the fast paths agree with the generic patch contraction
        for inds in ([(1, 1)], [(1, 1), (2, 1)], [(1, 1), (1, 2)])
            sites = Tuple(Val.(CartesianIndex.(inds)))
            ρ_fast = reduced_densitymatrix(CartesianIndex.(inds), psi, env)
            ρ_patch = PEPSKit._contract_densitymatrix(sites, (psi, psi), env)
            @test ρ_fast ≈ ρ_patch rtol = patch_rtol
        end
    end
end
