using AppleAccelerate
using Test
using Printf
using Random
using PEPSKit
using TensorKit
import Statistics: mean
include("utility/measure_heis.jl")
import .MeasureHeis: measure_heis

# benchmark data is from Phys. Rev. B 94, 035133 (2016)

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dbond, χenv, symm = 4, 16, Trivial
N1, N2 = 2, 2
Random.seed!(2024)
if symm == Trivial
    Pspace = ℂ^2
    Vspace = ℂ^Dbond
    Espace = ℂ^χenv
elseif symm == U1Irrep
    Pspace = ℂ[U1Irrep](1//2 => 1, -1//2 => 1)
    Vspace = ℂ[U1Irrep](0 => Dbond ÷ 2, 1//2 => Dbond ÷ 4, -1//2 => Dbond ÷ 4)
    Espace = ℂ[U1Irrep](0 => χenv ÷ 2, 1//2 => χenv ÷ 4, -1//2 => χenv ÷ 4)
else
    error("Not implemented")
end

peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end
# Heisenberg model Hamiltonian
# (already only includes nearest neighbor terms)
ham = heisenberg_XYZ(ComplexF64, symm, InfiniteSquare(N1, N2); Jx=1.0, Jy=1.0, Jz=1.0)
# convert to real tensors
ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

# initialize CTMRG environment
peps_ = InfinitePEPS(peps)
envs = CTMRGEnv(rand, Float64, peps_, Espace)
trscheme_envs = truncerr(1e-10) & truncdim(χenv)
ctm_alg = SequentialCTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme_envs)
envs = leading_boundary(envs, peps_, ctm_alg)

# NTU
dts = [1e-2, 5e-3, 1e-3]
maxiter = 2000
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)
for (n, dt) in enumerate(dts)
    alg = NTUpdate(;
        dt,
        maxiter,
        tol=1e-8,
        bondenv_alg=NTUEnvNN(),
        opt_alg=FullEnvTruncation(; tol=1e-8, verbose=false, check_int=10, maxiter=50, trscheme=trscheme_peps),
        ctm_alg=SequentialCTMRG(; tol=1e-10, verbosity=2, maxiter=30, trscheme=trscheme_envs),
    )
    result = ntupdate(peps, envs, ham, alg, ctm_alg)
    global peps = result[1]
    global envs = result[2]
end

PEPSKit.display_weights(peps.weights)

# measure physical quantities
peps_ = InfinitePEPS(peps)
meas = measure_heis(peps_, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
# @test isapprox(meas["e_site"], -0.6675; atol=1e-3)
# @test isapprox(mean(meas["mag_norm"]), 0.3767; atol=1e-3)
