using Test
using Printf
using Random
using PEPSKit
using TensorKit
import Statistics: mean
include("utility/heis.jl")
import .OpsHeis: gen_gate
import .RhoMeasureHeis: measrho_all

# benchmark data for D = 3 is from
# Phys. Rev. B 94, 035133 (2016)

# random initialization of 2x2 iPEPS and CTMRGEnv (using real numbers)
Dcut, χenv = 4, 16
N1, N2 = 2, 2
Pspace, Vspace = ℂ^2, ℂ^Dcut
Random.seed!(0)
peps = InfinitePEPS(rand, Float64, 2, Dcut; unitcell=(N1, N2))
wts = SUWeight(
    collect(id(Vspace) for (row, col) in Iterators.product(1:N1, 1:N2)),
    collect(id(Vspace) for (row, col) in Iterators.product(1:N1, 1:N2)),
)
# normalize peps
for ind in CartesianIndices(peps.A)
    peps.A[ind] /= PEPSKit.maxabs(peps.A[ind])
end
# Heisenberg model Hamiltonian
ham = gen_gate()

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-7, 1e-8, 1e-9]
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    Dcut2 = (n == 1 ? Dcut + 1 : Dcut)
    simpleupdate!(peps, wts, ham, dt, Dcut2; bipartite=true, evolstep=30000, wtdiff_tol=tol)
end
# absort weight into site tensors
absorb_wt!(peps, wts)
# CTMRG
envs = CTMRGEnv(rand, Float64, peps, ℂ^χenv)
trscheme = truncerr(1e-9) & truncdim(χenv)
ctm_alg = CTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme, ctmrgscheme=:sequential)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
rho1ss, rho2sss = calrho_all(envs, peps)
result = measrho_all(rho1ss, rho2sss)
@printf("Energy = %.8f\n", result["e_site"])
@printf("Staggered magnetization = %.8f\n", mean(result["mag_norm"]))
@test isapprox(result["e_site"], -0.6675; atol=1e-3)
@test isapprox(mean(result["mag_norm"]), 0.3767; atol=1e-3)

# continue with full update
dts = [2e-2, 1e-2, 5e-3, 1e-3, 5e-4]
for dt in dts
    fullupdate!(peps, envs, ham, dt, Dcut, χenv; rgmaxiter=5, cheap=true)
end
# measure physical quantities
rho1ss, rho2sss = calrho_all(envs, peps)
result = measrho_all(rho1ss, rho2sss)
@printf("Energy = %.8f\n", result["e_site"])
@printf("Staggered magnetization = %.8f\n", mean(result["mag_norm"]))
@test isapprox(result["e_site"], -0.66875; atol=1e-4)
@test isapprox(mean(result["mag_norm"]), 0.3510; atol=1e-3)
