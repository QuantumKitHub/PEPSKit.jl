using Test
using Printf
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut, χenv = 2, 16
N1, N2 = 2, 2
Random.seed!(0)
Pspace = ℂ^2
Vspace = ℂ^Dcut
Espace = ℂ^χenv
peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end
# Heisenberg model Hamiltonian
# (already only includes nearest neighbor terms)
ham = heisenberg_XYZ(ComplexF64, Trivial, InfiniteSquare(N1, N2); Jx=1.0, Jy=1.0, Jz=1.0)
# convert to real tensors
ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-7, 1e-8, 1e-8, 1e-8]
maxiter = 5000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    Dcut2 = (n == 1) ? Dcut + 2 : Dcut
    trscheme = truncerr(1e-10) & truncdim(Dcut2)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    result = simpleupdate(peps, ham, alg; bipartite=false)
    global peps = result[1]
end
# absort weight into site tensors
peps = InfinitePEPS(peps)
# CTMRG
envs = CTMRGEnv(rand, Float64, peps, Espace)
trscheme = truncerr(1e-9) & truncdim(χenv)
ctm_alg = CTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme, ctmrgscheme=:sequential)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
e_site = costfun(peps, envs, ham) / (N1 * N2)
@info @sprintf("Simple update energy = %.8f\n", e_site)
# benchmark data from Phys. Rev. B 94, 035133 (2016)
@test isapprox(e_site, -0.6594; atol=1e-3)

# continue with auto differentiation
ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; gradtol=1e-3, verbosity=2)
)
result = fixedpoint(peps, ham, opt_alg, envs)
ξ_h, ξ_v, = correlation_length(result.peps, result.env)
e_site2 = result.E / (N1 * N2)
@info @sprintf("Auto diff energy = %.8f\n", e_site)
@test e_site2 ≈ -0.6694421 atol = 1e-2
@test all(@. ξ_h > 0 && ξ_v > 0)
