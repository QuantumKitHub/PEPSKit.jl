using Test
using Random
using PEPSKit
using TensorKit

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut, symm = 8, Trivial
N1, N2 = 2, 2
Random.seed!(10)
if symm == Trivial
    Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    Vspace = Vect[fℤ₂](0 => Dcut / 2, 1 => Dcut / 2)
else
    error("Not implemented")
end
peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))

# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end

# Hubbard model Hamiltonian at half-filling
t, U = 1, 6
ham = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(N1, N2); t, U, mu=U / 2)

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 10000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncerr(1e-10) & truncdim(Dcut)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    peps, = simpleupdate(peps, ham, alg; bipartite=false)
end

# absorb weight into site tensors
peps = InfinitePEPS(peps)

# CTMRG
χenv0, χenv = 6, 20
Espace = Vect[fℤ₂](0 => χenv0 / 2, 1 => χenv0 / 2)
envs = CTMRGEnv(randn, Float64, peps, Espace)
for χ in [χenv0, χenv]
    ctm_alg = SequentialCTMRG(; maxiter=300, tol=1e-7)
    envs = leading_boundary(envs, peps, ctm_alg)
end

# Benchmark values of the ground state energy from
# Qin, M., Shi, H., & Zhang, S. (2016). Benchmark study of the two-dimensional Hubbard
# model with auxiliary-field quantum Monte Carlo method. Physical Review B, 94(8), 085103.
Es_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)
E_exact = Es_exact[U] - U / 2

# measure energy
E = costfun(peps, envs, ham) / (N1 * N2)
@info "Energy           = $E"
@info "Benchmark energy = $E_exact"
@test isapprox(E, E_exact; atol=5e-2)
