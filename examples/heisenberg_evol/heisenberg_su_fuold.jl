include("heis_tools.jl")

# benchmark data is from Phys. Rev. B 94, 035133 (2016)

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dbond, χenv, symm = 4, 16, Trivial
N1, N2 = 2, 2
Random.seed!(0)
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

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 10000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncerr(1e-10) & truncdim(Dbond)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    result = simpleupdate(peps, ham, alg; bipartite=false)
    global peps = result[1]
end
# absort weight into site tensors
peps = InfinitePEPS(peps)
# CTMRG
envs = CTMRGEnv(rand, Float64, peps, Espace)
trscheme = truncerr(1e-10) & truncdim(χenv)
ctm_alg = SequentialCTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
meas = measure_heis(peps, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.6675; atol=1e-3)
@test isapprox(mean(meas["mag_norm"]), 0.3767; atol=1e-3)

# continue with full update
dts = [2e-2, 1e-2, 5e-3]
trscheme_peps = truncerr(1e-10) & truncdim(Dbond)
trscheme_envs = truncerr(1e-10) & truncdim(χenv)
colmove_alg = SequentialCTMRG(; verbosity=0, maxiter=1, trscheme=trscheme_envs)
CTMRGAlg = SequentialCTMRG
proj_alg = HalfInfiniteProjector
reconv_alg = CTMRGAlg(;
    tol=1e-6, maxiter=10, verbosity=2, trscheme=trscheme_envs, projector_alg=proj_alg
)
ctm_alg = CTMRGAlg(;
    tol=1e-10, maxiter=50, verbosity=2, trscheme=trscheme_envs, projector_alg=proj_alg
)
for dt in dts
    fu_alg = FullUpdate(;
        dt=dt,
        maxiter=1000,
        trscheme=trscheme_peps,
        colmove_alg=colmove_alg,
        reconv_alg=reconv_alg,
    )
    result = fullupdate(peps, envs, ham, fu_alg, ctm_alg)
    global peps = result[1]
    global envs = result[2]
end
# measure physical quantities
meas = measure_heis(peps, ham, envs)
display(meas)
@info @sprintf("Energy = %.8f\n", meas["e_site"])
@info @sprintf("Staggered magnetization = %.8f\n", mean(meas["mag_norm"]))
@test isapprox(meas["e_site"], -0.66875; atol=1e-4)
@test isapprox(mean(meas["mag_norm"]), 0.3510; atol=1e-3)
