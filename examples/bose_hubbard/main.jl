using Markdown #hide
md"""
# Optimizing the $U(1)$-symmetric Bose-Hubbard model

This example demonstrates the simulation of the two-dimensional Bose-Hubbard model. In
particular, the point will be to showcase the use of internal symmetries and finite
particle densities in PEPS ground state searches. As we will see, incorporating symmetries
into the simulation consists of initializing a symmetric Hamiltonian, PEPS state and CTM
environment - made possible through TensorKit.

But first let's seed the RNG and import the required modules:
"""

using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528935);

md"""
## Defining the model

We will construct the Bose-Hubbard model Hamiltonian through the
[`bose_hubbard_model`](https://quantumkithub.github.io/MPSKitModels.jl/dev/man/models/#MPSKitModels.bose_hubbard_model),
function from MPSKitModels as reexported by PEPSKit. We'll simulate the model in its
Mott-insulating phase where the ratio $U/t$ is large, since in this phase we expect the
ground state to be well approximated by a PEPS with a manifest global $U(1)$ symmetry.
Furthermore, we'll impose a cutoff at 2 bosons per site, set the chemical potential to zero
and use a simple $1 \times 1$ unit cell:
"""

t = 1.0
U = 30.0
cutoff = 2
mu = 0.0
lattice = InfiniteSquare(1, 1);

md"""
Next, we impose an explicit global $U(1)$ symmetry as well as a fixed particle number
density in our simulations. We can do this by setting the `symmetry` argument of the
Hamiltonian constructor to `U1Irrep` and passing one as the particle number density
keyword argument `n`:
"""

symmetry = U1Irrep
n = 1
H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff, t, U, n);

md"""
Before we continue, it might be interesting to inspect the corresponding lattice physical
spaces (which is here just a $1 \times 1$ matrix due to the single-site unit cell):
"""

physical_spaces = physicalspace(H)

md"""
Note that the physical space contains $U(1)$ charges -1, 0 and +1. Indeed, imposing a
particle number density of +1 corresponds to shifting the physical charges by -1 to
're-center' the physical charges around the desired density. When we do this with a cutoff
of two bosons per site, i.e. starting from $U(1)$ charges 0, 1 and 2 on the physical level,
we indeed get the observed charges.

## Characterizing the virtual spaces

When running PEPS simulations with explicit internal symmetries, specifying the structure of
the virtual spaces of the PEPS and its environment becomes a bit more involved. For the
environment, one could in principle allow the virtual space to be chosen dynamically during
the boundary contraction using CTMRG by using a truncation scheme that allows for this
(e.g. using `alg=:truncrank` or `alg=:trunctol` to truncate to a fixed total bond dimension
or singular value cutoff respectively). For the PEPS virtual space however, the structure
has to be specified before the optimization.

While there are a host of techniques to do this in an informed way (e.g. starting from a
simple update result), here we just specify the virtual space manually. Since we're dealing
with a model at unit filling our physical space only contains integer $U(1)$ irreps.
Therefore, we'll build our PEPS and environment spaces using integer $U(1)$ irreps centered
around the zero charge:
"""

V_peps = U1Space(0 => 2, 1 => 1, -1 => 1)
V_env = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2);

md"""
## Finding the ground state

Having defined our Hamiltonian and spaces, it is just a matter of plugging this into the
optimization framework in the usual way to find the ground state. So, we first specify all
algorithms and their tolerances:
"""

boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, trscheme = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, maxiter = 10, alg = :eigsolver, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, maxiter = 150, ls_maxiter = 2, ls_maxfg = 2);

md"""
!!! note
	Taking CTMRG gradients and optimizing symmetric tensors tends to be more problematic
    than with dense tensors. In particular, this means that one frequently needs to tweak
    the `boundary_alg`, `gradient_alg` and `optimizer_alg` settings. There rarely is a
    general-purpose set of settings which will always work, so instead one has to adjust
    the simulation settings for each specific application. For example, it might help to
    switch between the CTMRG flavors `alg=:simultaneous` and `alg=:sequential` to
    improve convergence. The evaluation of the CTMRG gradient can be instable, so there it
    is advised to try the different `iterscheme=:diffgauge` and `iterscheme=:fixed` schemes
    as well as different `alg` keywords. Of course the tolerances of the algorithms and
    their subalgorithms also have to be compatible. For more details on the available
    options, see the [`fixedpoint`](@ref) docstring.

Keep in mind that the PEPS is constructed from a unit cell of spaces, so we have to make a
matrix of `V_peps` spaces:
"""

virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);

md"""
And at last, we optimize (which might take a bit):
"""

peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity = 3
)
@show E;

md"""
We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:
"""

E_ref = -0.273284888
@show (E - E_ref) / E_ref;
