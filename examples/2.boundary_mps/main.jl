using Markdown #hide
md"""
# Boundary MPS contractions using VUMPS and PEPOs

Instead of using CTMRG to contract an infinite PEPS, one can also use an boundary MPSs
ansatz to contract the infinite network. In particular, we will here use VUMPS to do so.

Before we start, we'll fix the random seed for reproducability:
"""

using Random
Random.seed!(29384293742893);

md"""
Besides `TensorKit` and `PEPSKit`, we here also need [`MPSKit`](https://quantumkithub.github.io/MPSKit.jl/stable/)
which implements the VUMPS algorithm as well as the required MPS operations:
"""

using TensorKit, PEPSKit, MPSKit

md"""
## Computing a PEPS norm

We start by initializing a random initial infinite PEPS:
"""

peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))

md"""
To compute its norm, we need to construct the transfer operator corresponding to
the partition function representing the overlap $\langle \psi_\text{PEPS} | \psi_\text{PEPS} \rangle$:
"""

transfer = InfiniteTransferPEPS(peps₀, 1, 1)

md"""
We then find its leading boundary MPS fixed point, where the corresponding eigenvalue
encodes the norm of the state. To that end, let us first we build an initial guess for the
boundary MPS, choosing a bond dimension of 20:
"""

mps₀ = initializeMPS(transfer, [ComplexSpace(20)])

md"""
Note that this will just construct a MPS with random Gaussian entries based on the virtual
spaces of the supplied transfer operator. Of course, one might come up with a better initial
guess (leading to better convergence) depending on the application. To find the leading
boundary MPS fixed point, we call [`leading_boundary`](@ref) using the [VUMPS](@extref)
algorithm from MPSKit. Note that, by default, `leading_boundary` uses CTMRG where the
settings are supplied as keyword arguments, so in the present case we need to supply the
VUMPS algorithm struct explicitly:
"""

mps, env, ϵ = leading_boundary(mps₀, transfer, VUMPS(; tol=1e-6, verbosity=2));

md"""
The norm of the state per unit cell is then given by the expectation value
$\langle \psi_\text{MPS} | \mathbb{T} | \psi_\text{MPS} \rangle$:
"""

norm_vumps = abs(prod(expectation_value(mps, transfer)))

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
is that now the transfer operator of the PEPS norm partition function has multiple lines,
each of which can be represented by an [`InfiniteTransferPEPS`](@ref) object. Such a
multi-line transfer operator is represented by a `MultilineTransferPEPS` object. In this
case, the boundary MPS is an [`MultilineMPS`](@extref) object, which should be initialized
by specifying a virtual space for each site in the partition function unit cell.

First, we construct a PEPS with a $2 \times 2$ unit cell using the `unitcell` keyword
argument and then define the corresponding transfer PEPS:
"""

peps₀_2x2 = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
transfer_2x2 = PEPSKit.MultilineTransferPEPS(peps₀_2x2, 1);

md"""
Now, the procedure is the same as before: We compute the norm once using VUMPS, once using CTMRG and then compare.
"""

mps₀_2x2 = initializeMPS(transfer_2x2, fill(ComplexSpace(20), 2, 2))
mps_2x2, = leading_boundary(mps₀_2x2, transfer_2x2, VUMPS(; tol=1e-6, verbosity=2))
norm_2x2_vumps = abs(prod(expectation_value(mps_2x2, transfer_2x2)))

env_ctmrg_2x2, = leading_boundary(
    CTMRGEnv(peps₀_2x2, ComplexSpace(20)), peps₀_2x2; tol=1e-6, verbosity=2
)
norm_2x2_ctmrg = abs(norm(peps₀_2x2, env_ctmrg_2x2))

@show abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps

md"""
Again, the results are compatible. Note that for larger unit cells and non-Hermitian PEPS
the VUMPS algorithm may become unstable, in which case the CTMRG algorithm is recommended.

## Contracting PEPO overlaps

Using exactly the same machinery, we can contract partition functions which encode the
expectation value of a PEPO for a given PEPS state. As an example, we can consider the
overlap of the PEPO correponding to the partition function of 3D classical Ising model with
our random PEPS from before and evaluate the overlap
$\langle \psi_\text{PEPS} | O_\text{PEPO} | \psi_\text{PEPS} \rangle$.

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

mps₀_pepo = initializeMPS(transfer_pepo, [ComplexSpace(20)])
mps_pepo, = leading_boundary(mps₀_pepo, transfer_pepo, VUMPS(; tol=1e-6, verbosity=2))
norm_pepo = abs(prod(expectation_value(mps_pepo, transfer_pepo)));
@show norm_pepo;

md"""
These objects and routines can be used to optimize PEPS fixed points of 3D partition
functions, see for example [Vanderstraeten et al.](@cite vanderstraeten_residual_2018)
"""
