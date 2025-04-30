using Markdown #hide
md"""
# [Boundary MPS contractions of 2D networks] (@id e_boundary_mps)

Instead of using CTMRG to contract the network encoding the norm of an infinite PEPS, one
can also use so-called [boundary MPS methods](@cite haegeman_diagonalizing_2017) to contract
this network. In this example, we will demonstrate how to use [the VUMPS algorithm](@cite
vanderstraeten_tangentspace_2019) to do so.

Before we start, we'll fix the random seed for reproducability:
"""

using Random
Random.seed!(29384293742893);

md""" Besides `TensorKit` and `PEPSKit`, here we also need to load the
[`MPSKit.jl`](https://quantumkithub.github.io/MPSKit.jl/stable/) package which implements a
host of tools for working with 1D matrix product states (MPS), including the VUMPS
algorithm:
"""

using TensorKit, PEPSKit, MPSKit

md"""
## Computing a PEPS norm

We start by initializing a random infinite PEPS. Let us use uniformly distributed complex
entries using `rand` (which sometimes lead to better convergence than Gaussian distributed
`randn` elements):
"""

peps₀ = InfinitePEPS(rand, ComplexF64, ComplexSpace(2), ComplexSpace(2))

md"""
To compute its norm, usually we would construct a double-layer `InfiniteSquareNetwork` which
encodes the bra-ket PEPS overlap and then contract this infinite square network, for example
using CTMRG. Here however, we will use another approach. If we take out a single row of this
infinite norm network, we can interpret it as a 2D row-to-row transfer operator ``T``. Here,
this transfer operator consists of an effective local rank-4 tensor at every site of a 2D
square lattice, where the local effective tensor is given by the contraction of a bra and
ket [`PEPSKit.PEPSTensor`](@ref) across their physical leg. Since the network we want to
contract can be interpreted as the infinite power of ``T``, we can contract it by finding
its leading eigenvector as a 1D MPS, which we call the boundary MPS.

In PEPSKit.jl, we can directly contruct the transfer operator corresponding to a PEPS norm
network from a given infinite PEPS as an [`InfiniteTransferPEPS`](@ref) object.
"""

T = InfiniteTransferPEPS(peps₀, 1, 1)

md"""
Since we'll find the leading eigenvector of ``T`` as a boundary MPS, we first need to
initialize an initial guess to supply to our algorithm. We can do this using the
[`initialize_mps`](@ref) function, which constructs a random MPS with a specific virtual
space for a given transfer operator. Here, we'll build an initial guess for the boundary MPS
with a bond dimension of 20:
"""

mps₀ = initialize_mps(T, [ComplexSpace(20)])

md"""

Note that this will just construct a MPS with random Gaussian entries based on the physical
spaces of the supplied transfer operator. Of course, one might come up with a better initial
guess (leading to better convergence) depending on the application. To find the leading
boundary MPS fixed point, we call [`leading_boundary`](@ref) using the
[`MPSKit.VUMPS`](@extref) algorithm from MPSKit. Note that, by default, `leading_boundary`
uses CTMRG where the settings are supplied as keyword arguments, so in the present case we
need to supply the VUMPS algorithm struct explicitly:
"""

mps, env, ϵ = leading_boundary(mps₀, T, VUMPS(; tol=1e-6, verbosity=2));

md"""
The norm of the state per unit cell is then given by the expectation value
$\langle \psi_\text{MPS} | \mathbb{T} | \psi_\text{MPS} \rangle$:
"""

norm_vumps = abs(prod(expectation_value(mps, T)))

md"""
This can be compared to the result obtained using CTMRG, where we see that the results match:
"""

env_ctmrg, = leading_boundary(
    CTMRGEnv(peps₀, ComplexSpace(20)), peps₀; tol=1e-6, verbosity=2
)
norm_ctmrg = abs(norm(peps₀, env_ctmrg))
@show abs(norm_vumps - norm_ctmrg) / norm_vumps;

md"""
## Working with unit cells

For PEPS with non-trivial unit cells, the principle is exactly the same. The only difference
is that now the transfer operator of the PEPS norm partition function has multiple rows or
'lines', each of which can be represented by an [`InfiniteTransferPEPS`](@ref) object. Such
a multi-line transfer operator is represented by a `MultilineTransferPEPS` object. In this
case, the boundary MPS is an [`MultilineMPS`](@extref) object, which should be initialized
by specifying a virtual space for each site in the partition function unit cell.

First, we construct a PEPS with a $2 \times 2$ unit cell using the `unitcell` keyword
argument and then define the corresponding transfer operator:
"""

peps₀_2x2 = InfinitePEPS(
    rand, ComplexF64, ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2)
)
T_2x2 = PEPSKit.MultilineTransferPEPS(peps₀_2x2, 1);

md"""
Now, the procedure is the same as before: We compute the norm once using VUMPS, once using CTMRG and then compare.
"""

mps₀_2x2 = initialize_mps(T_2x2, fill(ComplexSpace(20), 2, 2))
mps_2x2, = leading_boundary(mps₀_2x2, T_2x2, VUMPS(; tol=1e-6, verbosity=2))
norm_2x2_vumps = abs(prod(expectation_value(mps_2x2, T_2x2)))

env_ctmrg_2x2, = leading_boundary(
    CTMRGEnv(peps₀_2x2, ComplexSpace(20)), peps₀_2x2; tol=1e-6, verbosity=2
)
norm_2x2_ctmrg = abs(norm(peps₀_2x2, env_ctmrg_2x2))

@show abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps;

md"""
Again, the results are compatible. Note that for larger unit cells and non-Hermitian PEPS
[the VUMPS algorithm may become unstable](@cite vanderstraeten_variational_2021), in which
case the CTMRG algorithm is recommended.

## Contracting PEPO overlaps

Using exactly the same machinery, we can contract 2D networks which encode the expectation
value of a PEPO for a given PEPS state. As an example, we can consider the overlap of the
PEPO correponding to the partition function of [3D classical Ising model](@ref e_3d_ising)
with our random PEPS from before and evaluate the overlap $\langle \psi_\text{PEPS} |
O_\text{PEPO} | \psi_\text{PEPS} \rangle$.

The classical Ising PEPO is defined as follows:
"""

function ising_pepo(β; unitcell=(1, 1, 1))
    t = ComplexF64[exp(β) exp(-β); exp(-β) exp(β)]
    q = sqrt(t)

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    return InfinitePEPO(O; unitcell)
end;

md"""
To evaluate the overlap, we instantiate the PEPO and the corresponding [`InfiniteTransferPEPO`](@ref)
in the right direction, on the right row of the partition function (trivial here):
"""

pepo = ising_pepo(1)
transfer_pepo = InfiniteTransferPEPO(peps₀, pepo, 1, 1)

md"""
As before, we converge the boundary MPS using VUMPS and then compute the expectation value:
"""

mps₀_pepo = initialize_mps(transfer_pepo, [ComplexSpace(20)])
mps_pepo, = leading_boundary(mps₀_pepo, transfer_pepo, VUMPS(; tol=1e-6, verbosity=2))
norm_pepo = abs(prod(expectation_value(mps_pepo, transfer_pepo)));
@show norm_pepo;

md"""
These objects and routines can be used to optimize PEPS fixed points of 3D partition
functions, see for example [Vanderstraeten et al.](@cite vanderstraeten_residual_2018)
"""
