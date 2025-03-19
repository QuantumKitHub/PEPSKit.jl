using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

using MPSKit: add_physical_charge

# This example demonstrates the simulation of the two-dimensional Bose-Hubbard model using
# PEPSKit.jl. In particular, it showcases the use of internal symmetries and finite particle
# densities in PEPS simulations.

## Defining the model

# We will construct the Bose-Hubbard model Hamiltonian through the
# [`bose_hubbard_model` function from MPSKitModels.jl](https://quantumkithub.github.io/MPSKitModels.jl/dev/man/models/#MPSKitModels.bose_hubbard_model),
# as reexported by PEPSKit.jl. We'll simulate the model in its Mott-insulating phase
# where the ratio U/t is large, since in this phase we expect the ground state to be well
# approximated by a PEPS with a manifest global U(1) symmetry. Furthermore, we'll impose
# a cutoff at 2 bosons per site, set the chemical potential to zero and use a simple 1x1
# unit cell.

t = 1.0
U = 30.0
cutoff = 2
mu = 0.0
lattice = InfiniteSquare(1, 1)

# We'll impose an explicit global U(1) symmetry as well as a fixed particle number density
# in our simulations. We can do this by setting the `symmetry` keyword argument to `U1Irrep`
# and passing one as the particle number density keyword argument `n`.

symmetry = U1Irrep
n = 1

# We can then construct the Hamiltonian, and inspect the corresponding lattice of physical
# spaces.

H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff, t, U, n)
Pspaces = H.lattice

# Note that the physical space contains U(1) charges -1, 0 and +1. Indeed, imposing a
# particle number density of +1 corresponds to shifting the physical charges by -1 to
# 're-center' the physical charges around the desired density. When we do this with a cutoff
# of two bosons per site, i.e. starting from U(1) charges 0, 1 and 2 on the physical level,
# we indeed get the observed charges.

## Characterizing the virtual spaces

# When running PEPS simulations with explicit internal symmetries, specifying the structure
# of the virtual spaces of the PEPS and its environment becomes a bit more involved. For the
# environment, one could in principle allow the virtual space to be chosen dynamically
# during the boundary contraction using CTMRG by using a truncation scheme that allows for
# this (e.g. using alg=:truncdim or alg=:truncbelow to truncate to a fixed total bond
# dimension or singular value cutoff respectively). For the PEPS virtual space however, the
# structure has to be specified before the optimization.

# While there are a host of techniques to do this in an informed way (e.g. starting from
# a simple update result), here we just specify the virtual space manually. Since we're
# dealing with a model at unit filling our physical space only contains integer U(1) irreps.
# Therefore, we'll build our PEPS and environment spaces using integer U(1) irreps centered
# around the zero charge.

Vpeps = U1Space(0 => 2, 1 => 1, -1 => 1)
Venv = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)

## Finding the ground state

# Having defined our Hamiltonian and spaces, it is just a matter of pluggin this into the
# optimization framework in the usual way to find the ground state.

# specify algorithms and tolerances
boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, maxiter=10, alg=:eigsolver, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=200, ls_maxiter=2, ls_maxfg=2)
reuse_env = true

# initialize state
Nspaces = fill(Vpeps, size(lattice)...)
Espaces = fill(Vpeps, size(lattice)...)
Random.seed!(2928528935) # for reproducibility
ψ₀ = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
env₀ = CTMRGEnv(ψ₀, Venv)
env₀, = leading_boundary(env₀, ψ₀; boundary_alg...)

# optimize
ψ, env, E, info = fixedpoint(
    H, ψ₀, env₀; boundary_alg, gradient_alg, optimizer_alg, reuse_env
)

## Check the result

# We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
# using a cylinder circumference of Ly = 7 and a bond dimension of 446, which yields
# E = -0.273284888

E_ref = -0.273284888
@test E ≈ E_ref rtol = 1e-3
