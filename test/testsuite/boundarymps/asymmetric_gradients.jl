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

## Test asymmetric boundary MPS gradients through implicit differentiation
# ------------------------------------------------------------------------
# The implicit gradient of the energy of approximate ground states of the Heisenberg model on
# several unit cells (simple update, or a short C4v CTMRG optimization for the single-site
# cell) is compared with finite differences along a random direction, and a short
# optimization is run on a 2×2 unit cell.

sd = 20261021
# (unit cell, D, χ)
cases = [((1, 1), 3, 16), ((2, 2), 3, 16), ((3, 3), 2, 12)]
boundary_tol = 1.0e-11
boundary_maxiter = 500
solver_tol = 1.0e-10
fd_rtol = 1.0e-5

# optimization from a D = 2 simple-update state on a 2×2 unit cell, compared with the energy
# of Juraj Hasik's single-site D = 2 state
opt_unitcell = (2, 2)
D_opt = 2
χ_opt = 16
opt_maxiter = 5
E_ref = -0.6602310934799577

boundary_alg = BoundaryMPS(; tol = boundary_tol, maxiter = boundary_maxiter, verbosity = 0)
gradient_alg = ImplicitGradient(; solver_alg = (; alg = :GMRES, tol = solver_tol))

## Tests
# ------
function boundary_mps_asymmetric_gradients(AT)
    return @testset "Asymmetric boundary MPS gradients ($AT)" verbose = true begin
        @testset "Implicit gradient against finite differences, $(unitcell) unit cell, D = $D, χ = $χ" for (unitcell, D, χ) in cases
            psi, H = heisenberg_state(AT, D, unitcell; seed = sd, χ)
            Random.seed!(sd)
            dir = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(D); unitcell))

            env₀ = BoundaryMPSEnv(grown_ctmrg_env(psi, χ; tol = 1.0e-8), psi)
            env, = leading_boundary(env₀, psi, boundary_alg)

            _, g = energy_and_gradient(psi, env, boundary_alg, gradient_alg, H)
            dE = gradient_derivative(psi, env, dir, g)
            dE_fd = finite_difference_derivative(psi, env, dir, boundary_alg, H)
            @info "$(unitcell) boundary MPS: implicit gradient vs finite differences" dE dE_fd abs(dE - dE_fd) / abs(dE_fd)
            @test dE ≈ dE_fd rtol = fd_rtol
        end

        @testset "Optimization, $(opt_unitcell) unit cell, D = $D_opt, χ = $χ_opt" begin
            psi, H = heisenberg_state(AT, D_opt, opt_unitcell; seed = sd)
            env₀, = leading_boundary(
                BoundaryMPSEnv(grown_ctmrg_env(psi, χ_opt; tol = 1.0e-8), psi), psi;
                tol = boundary_tol, maxiter = boundary_maxiter,
            )
            E₀ = cost_function(psi, env₀, H)
            peps, env, E, info = fixedpoint(
                H, psi, env₀; optimizer_alg = (; tol = 1.0e-4, maxiter = opt_maxiter), verbosity = 1
            )
            @info "$(opt_unitcell) boundary MPS optimization" E₀ E info.gradnorms[1] info.gradnorms[end]
            @test E < E₀
            @test info.gradnorms[end] < info.gradnorms[1]
            @test E / prod(opt_unitcell) ≈ E_ref atol = 1.0e-2
        end
    end
end
