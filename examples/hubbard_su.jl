using Test
using Random
using PEPSKit
using TensorKit

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dbond, symm = 8, Trivial
Nr, Nc = 2, 2
Random.seed!(10)
if symm == Trivial
    Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    Vspace = Vect[fℤ₂](0 => Dbond / 2, 1 => Dbond / 2)
else
    error("Not implemented")
end
peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc))

# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end

# Hubbard model Hamiltonian at half-filling
t, U = 1, 6
ham = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(Nr, Nc); t, U, mu=U / 2)

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 20000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncerr(1e-10) & truncdim(Dbond)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    global peps, = simpleupdate(peps, ham, alg; bipartite=false)
end

# absorb weight into site tensors
peps = InfinitePEPS(peps)

# CTMRG
χenv0, χenv = 6, 20
Espace = Vect[fℤ₂](0 => χenv0 / 2, 1 => χenv0 / 2)
env = CTMRGEnv(randn, Float64, peps, Espace)
for χ in [χenv0, χenv]
    env, = leading_boundary(env, peps; alg=:sequential, maxiter=300, tol=1e-7)
end

# Benchmark values of the ground state energy from
# Qin, M., Shi, H., & Zhang, S. (2016). Benchmark study of the two-dimensional Hubbard
# model with auxiliary-field quantum Monte Carlo method. Physical Review B, 94(8), 085103.
Es_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)
E_exact = Es_exact[U] - U / 2

# measure energy
E = cost_function(peps, env, ham) / (Nr * Nc)
@info "Energy           = $E"
@info "Benchmark energy = $E_exact"
@test isapprox(E, E_exact; atol=5e-2)
