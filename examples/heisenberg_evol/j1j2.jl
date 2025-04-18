include("heis_tools.jl")

# benchmark data is from 
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/ea4140fbd7da0fc1b75fac2871f75bda125189a8/single-site_pg-C4v-A1/j20.5/state_1s_A1_j20.5_D2_chi_opt48.dat
Dbond, χenv, symm = 2, 32, U1Irrep
trscheme_env = truncerr(1e-10) & truncdim(χenv)
Nr, Nc = 2, 2
J1, J2 = 1.0, 0.5
ham = real(j1_j2(ComplexF64, symm, InfiniteSquare(Nr, Nc); J1, J2, sublattice=false))

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
if symm == Trivial
    Pspace = ℂ^2
    Vspace = ℂ^Dbond
    Espace = ℂ^χenv
elseif symm == U1Irrep
    Pspace = Vect[U1Irrep](1//2 => 1, -1//2 => 1)
    Vspace = Vect[U1Irrep](0 => 2, 1//2 => 1, -1//2 => 1)
    Espace = Vect[U1Irrep](0 => χenv ÷ 2, 1//2 => χenv ÷ 4, -1//2 => χenv ÷ 4)
else
    error("Not implemented")
end
Random.seed!(2025)
peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end

# simple update
dts = [1e-1, 1e-2, 1e-3, 1e-4]
tols = [1e-4, 1e-8, 1e-8, 1e-8]
maxiter = 10000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    Dcut = (n == 1) ? max(4, Dbond) : Dbond
    trscheme_peps = truncerr(1e-10) & truncdim(Dcut)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(peps, ham, alg)
    global peps = result[1]
end
# measure physical quantities with CTMRG
peps_ = InfinitePEPS(peps)
normalize!.(peps_.A, Inf)
Random.seed!(100)
env = CTMRGEnv(rand, Float64, peps_, Espace)
ctm_alg = SequentialCTMRG(; tol=1e-10, verbosity=3, trscheme=trscheme_env)
env, = leading_boundary(env, peps_, ctm_alg)
meas = measure_heis(peps_, ham, env)
display(meas)
energy, mag = meas["e_site"], mean(meas["mag_norm"])
@info @sprintf("Energy = %.8f\n", energy)
@info @sprintf("Staggered magnetization = %.8f\n", mag)
@test isapprox(energy, -0.470392432614689; atol=1e-3)
@test isapprox(mag, 0.30427239576842324; atol=1e-2)
