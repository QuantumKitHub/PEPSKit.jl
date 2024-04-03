using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit
using PEPSKit

function test_gauge_fixing(
    f, T, P::S, V::S, E::S; χenv::Int=20, unitcell::NTuple{2,Int}=(1, 1)
) where {S<:ElementarySpace}
    ψ = InfinitePEPS(f, T, P, V; unitcell)
    env = CTMRGEnv(ψ; Venv=E)

    ctmalg = CTMRG(;
        trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2
    )
    env = leading_boundary(ψ, ctmalg, env)

    println("Testing gauge fixing for $(sectortype(P)) symmetry and $unitcell unit cell.")

    println("\nBefore gauge-fixing:")
    env′, = PEPSKit.ctmrg_iter(ψ, env, ctmalg)
    @show PEPSKit.check_elementwise_convergence(env, env′)

    println("\nAfter gauge-fixing:")
    envfix = PEPSKit.gauge_fix(env, env′)
    @show PEPSKit.check_elementwise_convergence(env, envfix)
    return println()
end

# Trivial

P = ℂ^2 # physical space
V = ℂ^2 # PEPS virtual space
χenv = 20 # environment truncation dimension
E = ℂ^χenv # environment virtual space

test_gauge_fixing(randn, ComplexF64, P, V, E; χenv, unitcell=(1, 1))
test_gauge_fixing(randn, ComplexF64, P, V, E; χenv, unitcell=(2, 2))
test_gauge_fixing(randn, ComplexF64, P, V, E; χenv, unitcell=(3, 4)) # check gauge-fixing for unit cells > (2, 2)

# Convergence of real CTMRG seems to be more sensitive to initial guess
test_gauge_fixing(randn, Float64, P, V, E; χenv, unitcell=(1, 1))
test_gauge_fixing(randn, Float64, P, V, E; χenv, unitcell=(2, 2))
test_gauge_fixing(randn, Float64, P, V, E; χenv, unitcell=(3, 4))

# Z2

P = Z2Space(0 => 1, 1 => 1) # physical space
V = Z2Space(0 => 2, 1 => 2) # PEPS virtual space
χenv = 20 # environment truncation dimension
E = Z2Space(0 => χenv / 2, 1 => χenv / 2) # environment virtual space

test_gauge_fixing(randn, ComplexF64, P, V, E; χenv, unitcell=(1, 1))
test_gauge_fixing(randn, ComplexF64, P, V, E; χenv, unitcell=(2, 2))

test_gauge_fixing(randn, Float64, P, V, E; χenv, unitcell=(1, 1))
test_gauge_fixing(randn, Float64, P, V, E; χenv, unitcell=(2, 2))
