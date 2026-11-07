using Test
using Random
using LinearAlgebra
using PEPSKit
using Adapt
using MPSKit
using TensorKit

include("utility.jl")

## Test observables evaluated with an asymmetric boundary MPS environment against CTMRG
# -------------------------------------------------------------------------------------
# Approximate ground states of the Heisenberg model on several unit cells (simple update, or a
# short C4v CTMRG optimization for the single-site cell), contracted with a
# top and a bottom boundary MPS initialized from a CTMRG environment grown from a product
# state. Checks that every characteristic equation holds, compares the network value and the
# energies with CTMRG, and checks that a single-site state repeated over a larger unit cell
# gives the single-site values.

sd = 20261021
# (unit cell, D, χ)
cases = [((1, 1), 3, 16), ((2, 2), 3, 16), ((3, 3), 2, 12)]
boundary_tol = 1.0e-10
boundary_maxiter = 500
residual_tol = 1.0e-8
ctmrg_rtol = 1.0e-4 # boundary MPS and CTMRG at the same χ differ by a truncation error
patch_rtol = 1.0e-10 # fast paths against the generic patch contraction
tiling_rtol = 1.0e-10

boundary_alg = BoundaryMPS(; tol = boundary_tol, maxiter = boundary_maxiter, verbosity = 0)

## Tests
# ------
function boundary_mps_asymmetric_observables(AT)
    return @testset "Asymmetric boundary MPS observables ($AT)" verbose = true begin
        @testset "$(unitcell) unit cell, D = $D, χ = $χ" for (unitcell, D, χ) in cases
            psi, H = heisenberg_state(AT, D, unitcell; seed = sd, χ)
            # next-nearest-neighbour terms are evaluated through the generic patch contraction
            H_j1j2 = adapt(AT, j1_j2_model(InfiniteSquare(unitcell...); J2 = 0.5))

            ctmrg_env = grown_ctmrg_env(psi, χ; tol = boundary_tol)
            env, info = leading_boundary(BoundaryMPSEnv(ctmrg_env, psi), psi, boundary_alg)
            @test info.converged

            # the characteristic equations hold on every site
            VLt, VLb = map(PEPSKit.left_null, env.ALt), map(PEPSKit.left_null, env.ALb)
            VRt = map(A -> PEPSKit.right_null(PEPSKit.repartition_right(A)), env.ARt)
            VRb = map(A -> PEPSKit.right_null(PEPSKit.repartition_right(A)), env.ARb)
            lvar(A, V) = zeros(scalartype(A), space(V, numind(V))' ← space(A, numind(A))')
            rvar(A, V) = zeros(scalartype(A), space(A, 1) ← space(V, 1))
            x = (
                map(lvar, env.ALt, VLt), map(rvar, env.ARt, VRt), env.Ct, env.GLt, env.GRt,
                map(lvar, env.ALb, VLb), map(rvar, env.ARb, VRb), env.Cb, env.GLb, env.GRb,
                env.GLm, env.GRm,
            )
            F = PEPSKit.generate_full_boundary_mps_characteristic_equation(env, (VLt, VRt, VLb, VRb))
            bonds = ((1, 2, env.Ct, identity), (6, 7, PEPSKit._bottom(env)[3], PEPSKit._rotate_sites))
            residuals = PEPSKit._characteristic_residuals(F(InfiniteSquareNetwork(psi), x...), bonds)
            @info "$(unitcell) characteristic equation residuals" residuals
            @test all(<(residual_tol), residuals)

            # the network value and the energies agree with CTMRG
            @test norm(psi, env) ≈ norm(psi, ctmrg_env) rtol = ctmrg_rtol
            E, E_ref = expectation_value(psi, H, env), expectation_value(psi, H, ctmrg_env)
            @info "$(unitcell) boundary MPS vs CTMRG energy per site" E / prod(unitcell) E_ref / prod(unitcell)
            @test E ≈ E_ref rtol = ctmrg_rtol
            @test expectation_value(psi, H_j1j2, env) ≈ expectation_value(psi, H_j1j2, ctmrg_env) rtol = ctmrg_rtol

            # the fast paths agree with the generic patch contraction
            for (r, c) in Iterators.product(1:unitcell[1], 1:unitcell[2])
                for inds in ([(r, c)], [(r, c), (r + 1, c)], [(r, c), (r, c + 1)])
                    sites = Tuple(Val.(CartesianIndex.(inds)))
                    ρ_fast = reduced_densitymatrix(CartesianIndex.(inds), psi, env)
                    ρ_patch = PEPSKit._contract_densitymatrix(sites, (psi, psi), env)
                    @test ρ_fast ≈ ρ_patch rtol = patch_rtol
                end
            end
        end

        @testset "Single-site state on a 2×3 unit cell" begin
            psi₁, H₁ = heisenberg_state(AT, 3, (1, 1); seed = sd)
            env₁, = leading_boundary(BoundaryMPSEnv(grown_ctmrg_env(psi₁, 16; tol = boundary_tol), psi₁), psi₁, boundary_alg)
            psi = InfinitePEPS(fill(psi₁[1, 1], 2, 3))
            H = adapt(AT, heisenberg_XYZ(InfiniteSquare(2, 3)))
            env, = leading_boundary(BoundaryMPSEnv(grown_ctmrg_env(psi, 16; tol = boundary_tol), psi), psi, boundary_alg)
            @test expectation_value(psi, H, env) / 6 ≈ expectation_value(psi₁, H₁, env₁) rtol = tiling_rtol
            @test abs(norm(psi, env))^(1 / 6) ≈ abs(norm(psi₁, env₁)) rtol = tiling_rtol
        end
    end
end
