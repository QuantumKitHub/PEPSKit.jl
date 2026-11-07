using Test
using Random
using LinearAlgebra
using PEPSKit
using Adapt
using MPSKit
using TensorKit
using Zygote
using KrylovKit

include("utility.jl")
include("../gradients/gradient_utility.jl")

## Test C4v boundary MPS gradients through implicit differentiation
# -----------------------------------------------------------------------
# The implicit gradient of the energy of an approximate ground state of the Heisenberg model,
# from a short C4v CTMRG optimization, is compared with finite differences along a symmetric
# direction, and a short boundary MPS optimization is run from such a state.

sd = 42039482049
symmetry = RotateReflect()
D = 3
χ = 16
boundary_tol = 1.0e-11
boundary_maxiter = 500
solver_tol = 1.0e-10
fd_rtol = 1.0e-5

# optimization from a D = 2 state after five C4v CTMRG optimization steps, compared with Juraj
# Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat
D_opt = 2
χ_opt = 16
opt_maxiter = 10
E_ref = -0.6602310934799577

## Tests
# ------
function boundary_mps_c4v_gradients(AT)
    return @testset "C4v boundary MPS gradients ($AT)" verbose = true begin
        @testset "Implicit gradient against finite differences, D = $D, χ = $χ" begin
            psi, H = heisenberg_state(AT, D, (1, 1); seed = sd, χ)
            Random.seed!(sd)
            dir = symmetrize!(adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(D))), symmetry)

            boundary_alg = C4vBoundaryMPS(;
                tol = boundary_tol, maxiter = boundary_maxiter, verbosity = 0
            )
            gradient_alg = ImplicitGradient(; solver_alg = (; alg = :GMRES, tol = solver_tol))
            env₀ = C4vBoundaryMPSEnv(grown_ctmrg_env(psi, χ; tol = 1.0e-8), psi)
            env, = leading_boundary(env₀, psi, boundary_alg)

            _, g = energy_and_gradient(psi, env, boundary_alg, gradient_alg, H)
            symmetrize!(g, symmetry)
            dE = gradient_derivative(psi, env, dir, g)
            dE_fd = finite_difference_derivative(psi, env, dir, boundary_alg, H)
            @info "C4v boundary MPS: implicit gradient vs finite differences" dE dE_fd abs(dE - dE_fd) / abs(dE_fd)
            @test dE ≈ dE_fd rtol = fd_rtol
        end

        @testset "Optimization, D = $D_opt, χ = $χ_opt" begin
            psi, H = heisenberg_state(AT, D_opt, (1, 1); seed = sd, χ = χ_opt, c4v_maxiter = 5)
            env₀, = leading_boundary(
                C4vBoundaryMPSEnv(grown_ctmrg_env(psi, χ_opt; tol = 1.0e-8), psi), psi;
                tol = boundary_tol, maxiter = boundary_maxiter,
            )
            E₀ = cost_function(psi, env₀, H)
            peps, env, E, info = fixedpoint(
                H, psi, env₀; optimizer_alg = (; tol = 1.0e-4, maxiter = opt_maxiter), verbosity = 1
            )
            @info "C4v boundary MPS optimization" E₀ E info.gradnorms[1] info.gradnorms[end]
            @test E < E₀
            @test info.gradnorms[end] < info.gradnorms[1]
            @test E ≈ E_ref atol = 1.0e-2
        end
    end
end
