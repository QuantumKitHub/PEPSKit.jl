using Test
using Printf
using Random
using PEPSKit
using TensorKit
using OptimKit
using KrylovKit
import Statistics: mean
import MPSKitModels: S_x, S_y, S_z, S_exchange
include("utility/measure_heis.jl")
import .MeasureHeis: measure_heis

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut, χenv = 4, 16
N1, N2 = 2, 2
Random.seed!(0)
peps = InfiniteWeightPEPS(rand, Float64, ℂ^2, ℂ^Dcut; unitcell=(N1, N2))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end

# Heisenberg model Hamiltonian
# (only includes nearest neighbor terms)
lattice = InfiniteSquare(N1, N2)
onsite = TensorMap([1.0 0.0; 0.0 1.0], ℂ^2, ℂ^2)
ham = heisenberg_XYZ(lattice; Jx=1.0, Jy=1.0, Jz=1.0)
# convert to real tensors
ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

# Include the onsite operators in two ways
ham_SU = LocalOperator(
    ham.lattice,
    Tuple(
        sites => op + (S_z() ⊗ onsite) / 2 for
        (sites, op) in ham.terms if length(sites) == 2
    )...,
)
ham_CTMRG = LocalOperator(
    ham.lattice,
    Tuple(ind => op for (ind, op) in ham.terms)...,
    ((idx,) => S_z() for idx in vertices(lattice))...,
)

# simple update with ham_SU
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 10000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    Dcut2 = (n == 1 ? Dcut + 1 : Dcut)
    trscheme = truncerr(1e-10) & truncdim(Dcut2)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    result = simpleupdate(peps, ham_SU, alg; bipartite=false)
    global peps = result[1]
end
# absort weight into site tensors
peps = InfinitePEPS(peps)
# CTMRG
envs = CTMRGEnv(rand, Float64, peps, ℂ^χenv)
trscheme = truncerr(1e-9) & truncdim(χenv)
ctm_alg = CTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme, ctmrgscheme=:simultaneous)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
meas = measure_heis(peps, ham_SU, envs)

# CTMRG with ham_CTMRG
psi_init = InfinitePEPS(2, Dcut; unitcell=(N1, N2))
env0 = CTMRGEnv(psi_init, ComplexSpace(χenv));
env_init = leading_boundary(env0, psi_init, ctm_alg);

opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)
result = fixedpoint(psi_init, ham_CTMRG, opt_alg, env_init)

@test isapprox(result.E / (N1 * N2), meas["e_site"], atol=1e-2)