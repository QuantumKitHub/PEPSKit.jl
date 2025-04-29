using Markdown #hide
md"""
# Fermi-Hubbard model with $f\mathbb{Z}_2 \boxtimes U(1)$ symmetry, at large $U$ and half-filling

In this example, we will demonstrate how to handle fermionic PEPS tensors and how to
optimize them. To that end, we consider the two-dimensional Hubbard model

```math
H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- -
c_{i,\sigma}^- c_{j,\sigma}^+ \right) + U \sum_i n_{i,\uparrow}n_{i,\downarrow} - \mu \sum_i n_i
```

where $\sigma \in \{\uparrow,\downarrow\}$ and $n_{i,\sigma} = c_{i,\sigma}^+ c_{i,\sigma}^-$
is the fermionic number operator. As in previous examples, using fermionic degrees of freedom
is a matter of creating tensors with the right symmetry sectors - the rest of the simulation
workflow remains the same.

First though, we make the example deterministic by seeding the RNG, and we make our imports:
"""

using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528937);

md"""
## Defining the fermionic Hamiltonian

Let us start by fixing the parameters of the Hubbard model. We're going to use a hopping of
$t=1$ and a large $U=8$ on a $2 \times 2$ unit cell:
"""

t = 1.0
U = 8.0
lattice = InfiniteSquare(2, 2);

md"""
In order to create fermionic tensors, one needs to define symmetry sectors using TensorKit's
`FermionParity`. Not only do we want use fermion parity but we also want our
particles to exploit the global $U(1)$ symmetry. The combined product sector can be obtained
using the [Deligne product](https://jutho.github.io/TensorKit.jl/stable/lib/sectors/#TensorKitSectors.deligneproduct-Tuple{Sector,%20Sector}),
called through `⊠` which is obtained by typing `\boxtimes+TAB`. We will not impose any extra
spin symmetry, so we have:
"""

fermion = fℤ₂
particle_symmetry = U1Irrep
spin_symmetry = Trivial
S = fermion ⊠ particle_symmetry

md"""
The next step is defining graded virtual PEPS and environment spaces using `S`. Here we also
use the symmetry sector to impose half-filling. That is all we need to define the Hubbard
Hamiltonian:
"""

D, χ = 1, 1
V_peps = Vect[S]((0, 0) => 2 * D, (1, 1) => D, (1, -1) => D)
V_env = Vect[S](
    (0, 0) => 4 * χ, (1, -1) => 2 * χ, (1, 1) => 2 * χ, (0, 2) => χ, (0, -2) => χ
)
S_aux = S((1, -1))
H₀ = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, lattice; t, U)
H = add_physical_charge(H₀, fill(S_aux, size(H₀.lattice)...));

md"""
## Finding the ground state

Again, the procedure of ground state optimization is very similar to before. First, we
define all algorithmic parameters:
"""

boundary_alg = (; tol=1e-8, alg=:simultaneous, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, maxiter=80, ls_maxiter=3, ls_maxfg=3)

md"""
Second, we initialize a PEPS state and environment (which we converge) constructed from
symmetric physical and virtual spaces:
"""

physical_spaces = H.lattice
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);

md"""
And third, we start the ground state search (this does take quite long):
"""

peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;

md"""
Finally, let's compare the obtained energy against a reference energy from a QMC study by
[Qin et al.](@cite qin_benchmark_2016). With the parameters specified above, they obtain an
energy of $E_\text{ref} \approx 4 \times -0.5244140625 = -2.09765625$ (the factor 4 comes
from the $2 \times 2$ unit cell that we use here). Thus, we find:
"""

E_ref = -2.09765625
@show (E - E_ref) / E_ref;
