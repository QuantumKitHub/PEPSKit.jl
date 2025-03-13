include("heis_tools.jl")

# benchmark data is from Phys. Rev. B 94, 035133 (2016)
Dbond, χenv, symm = 4, 16, Trivial
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)
trscheme_env = truncerr(1e-10) & truncdim(χenv)
Nr, Nc = 2, 2
# Heisenberg model Hamiltonian
# (already only includes nearest neighbor terms)
ham = heisenberg_XYZ(ComplexF64, symm, InfiniteSquare(Nr, Nc); Jx=1.0, Jy=1.0, Jz=1.0)
# convert to real tensors
ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
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
Random.seed!(0)
peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end

# simple update
dts = [1e-2, 1e-3, 4e-4]
tols = [1e-6, 1e-8, 1e-8]
maxiter = 10000
for (dt, tol) in zip(dts, tols)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
    result = simpleupdate(peps, ham, alg; bipartite=true)
    global peps = result[1]
end
# measure physical quantities with CTMRG
peps_ = InfinitePEPS(peps)
env = CTMRGEnv(rand, Float64, peps_, Espace)
env, = leading_boundary(
    env,
    peps_;
    alg=:sequential,
    projector_alg=:fullinfinite,
    tol=1e-10,
    verbosity=2,
    trscheme=trscheme_env,
)
meas = measure_heis(peps_, ham, env)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.6675; atol=1e-3)
@test isapprox(mean(meas["mag_norm"]), 0.3767; atol=1e-3)
