using Test
using Printf
using Random
using PEPSKit
using TensorKit
import Statistics: mean
include("utility/heis.jl")
import .RhoMeasureHeis: gen_gate, measrho_all

# benchmark data is from Phys. Rev. B 94, 035133 (2016)

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
ham = gen_gate()

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-7, 5e-8, 1e-8]
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    Dcut2 = (n == 1 ? Dcut + 1 : Dcut)
    simpleupdate!(peps, ham, dt, Dcut2; bipartite=true, evolstep=10000, wtdiff_tol=tol)
end
# absort weight into site tensors
peps = InfinitePEPS(peps)
# CTMRG
envs = CTMRGEnv(rand, Float64, peps, ℂ^χenv)
trscheme = truncerr(1e-9) & truncdim(χenv)
ctm_alg = CTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme, ctmrgscheme=:sequential)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
rho1ss, rho2sss = calrho_all(envs, peps)
result = measrho_all(rho1ss, rho2sss)
@info @sprintf("Energy = %.8f\n", result["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(result["mag_norm"]))
@test isapprox(result["e_site"], -0.6675; atol=1e-3)
@test isapprox(mean(result["mag_norm"]), 0.3767; atol=1e-3)
