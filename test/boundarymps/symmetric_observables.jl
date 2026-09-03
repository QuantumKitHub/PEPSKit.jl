using Test
using Random
using LinearAlgebra
using PEPSKit
using MPSKit
using TensorKit

sd = 42039482049

## Test observables evaluated with a symmetric boundary MPS environment against CTMRG
# -----------------------------------------------------------------------------------
# The boundary MPS environment absorbs the corners into its gauge center and acts as its own
# bra, so it reaches the generic `expectation_value` machinery through a different set of
# contractions than CTMRG does. Contracting the same symmetric state both ways must give the
# same energy.

χbond = 2
χenv = 16
symmetry = RotateReflect()
Pspace = ComplexSpace(2)
Vspace = ComplexSpace(χbond)
Espace = ComplexSpace(χenv)

H = heisenberg_XYZ(InfiniteSquare())

boundary_tol = 1.0e-10
boundary_maxiter = 400
boundary_verbosity = 0
comp_tol = 1.0e-8

mps_algs = [:VUMPS, :VOMPS]

## Tests
# ------
Random.seed!(sd)
# a symmetric boundary MPS contraction is only meaningful for a fully symmetric network
psi = symmetrize!(InfinitePEPS(Pspace, Vspace), symmetry)

# CTMRG reference
ctmrg_env, = leading_boundary(
    CTMRGEnv(psi, Espace), psi;
    alg = :SimultaneousCTMRG, tol = boundary_tol,
    maxiter = boundary_maxiter, verbosity = boundary_verbosity,
)
norm_ref = norm(psi, ctmrg_env)
E_ref = expectation_value(psi, H, ctmrg_env)

@testset "Symmetric boundary MPS Heisenberg energy, mps_alg=:$mps_alg" for mps_alg in mps_algs
    boundary_alg = SymmetricBoundaryMPS(;
        mps_alg, tol = boundary_tol,
        maxiter = boundary_maxiter, verbosity = boundary_verbosity,
    )
    env, info = leading_boundary(SymmetricBoundaryMPSEnv(psi, Espace), psi, boundary_alg)
    @test info.converged

    # the network value normalizing the expectation value must agree first
    @test norm(psi, env) ≈ norm_ref rtol = comp_tol

    E = expectation_value(psi, H, env)
    @test E ≈ E_ref rtol = comp_tol

    # `cost_function` is the real part of the same quantity, and is what the optimizer sees
    @test cost_function(psi, env, H) ≈ real(E_ref) rtol = comp_tol
end
