using Test
using Random
using PEPSKit
using TensorKit
import Statistics: mean
include("utility/heis.jl")
using .OpsHeis, .RhoMeasureHeis

# benchmark data for D = 3 is from
# Phys. Rev. B 94, 035133 (2016)

# random initialization of 2x2 iPEPS and CTMRGEnv
# (using real numbersf)
Dcut = 3
χenv = 24
N1, N2 = 2, 2
Pspace = ℂ^2
Vspace = ℂ^Dcut
Random.seed!(0)
peps = InfinitePEPS(
    collect(
        TensorMap(rand, Float64, Pspace, Vspace ⊗ Vspace ⊗ Vspace' ⊗ Vspace') for
        (row, col) in Iterators.product(1:N1, 1:N2)
    ),
)
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

# simple update energy and magnetization
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-7, 1e-8, 1e-9]
for (dt, tol) in zip(dts, tols)
    simpleupdate!(peps, wts, ham, dt, Dcut; evolstep=30000, wtdiff_tol=tol)
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
display(result)
@test isapprox(result["e_site"], -0.6633; atol=1e-3)
@test isapprox(mean(result["mag_norm"]), 0.3972; atol=1e-3)

# full update energy and magnetization
# e_fu = -0.6654
# mag_fu = 0.3634
