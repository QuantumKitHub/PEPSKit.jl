using Test
using Printf
using Random
using PEPSKit
using TensorKit
import Statistics: mean
import LinearAlgebra: I
include("utility/measure_heis.jl")
import .MeasureHeis: measure_heis

# random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
Dcut, χenv, symm = 8, 20, Trivial
N1, N2 = 2, 2
Random.seed!(0)
if symm == Trivial
    Pspace = Vect[fℤ₂]((0) => 2, (1) => 2)
    Vspace = Vect[fℤ₂]((0) => Dcut / 2, (1) => Dcut / 2)
    Espace = Vect[fℤ₂]((0) => χenv / 2, (1) => χenv / 2)
else
    error("Not implemented")
end

peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))
# normalize vertex tensors
for ind in CartesianIndices(peps.vertices)
    peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
end
# Hubbard model Hamiltonian
t = 1.0
U = 6.0
ham = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(N1, N2); t=t, U=U, mu=U / 2)
# convert to real tensors
ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

unit = TensorMap(Matrix{ComplexF64}(I, 4, 4), Pspace, Pspace)
one_site = [op for (ind, op) in ham.terms if length(ind) == 1][1]

# Convert to a Hamiltonian that only includes nearest-neighbour interactions
ham = LocalOperator(
    ham.lattice,
    Tuple(
        sites => op + (one_site ⊗ unit) / 2 for
        (sites, op) in ham.terms if length(sites) == 2
    )...,
)

# simple update
dts = [1e-2, 1e-3, 4e-4, 1e-4]
tols = [1e-6, 1e-8, 1e-8, 1e-8]
maxiter = 10000
for (n, (dt, tol)) in enumerate(zip(dts, tols))
    trscheme = truncerr(1e-10) & truncdim(Dcut)
    alg = SimpleUpdate(dt, tol, maxiter, trscheme)
    result = simpleupdate(peps, ham, alg; bipartite=false)
    global peps = result[1]
end

# absort weight into site tensors
peps = InfinitePEPS(peps)
# CTMRG
envs = CTMRGEnv(rand, ComplexF64, peps, Espace)
trscheme = truncerr(1e-9) & truncdim(χenv)
ctm_alg = CTMRG(; tol=1e-10, verbosity=2, trscheme=trscheme, ctmrgscheme=:sequential)
envs = leading_boundary(envs, peps, ctm_alg)
# measure physical quantities
E = expectation_value(peps, ham, envs)
@info @sprintf("Energy = %.8f\n", real(E / (N1 * N2)))

"""
Benchmark values of the ground state energy, based on https://www.osti.gov/servlets/purl/1565498 (A benchmark study of the two-dimensional Hubbard model
with auxiliary-field quantum Monte Carlo method)
"""
E_exact = Dict(0 => -1.62, 2 => -0.176, 4 => 0.8603, 6 => -0.6567, 8 => -0.5243)

@test isapprox(real(E / (N1 * N2)), E_exact[U] - U / 2; atol=1e-2)
