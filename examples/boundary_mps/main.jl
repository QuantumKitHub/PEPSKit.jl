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

md"""
Besides `TensorKit` and `PEPSKit`, here we also need to load the
[`MPSKit.jl`](https://quantumkithub.github.io/MPSKit.jl/stable/) package which implements a
host of tools for working with 1D matrix product states (MPS), including the VUMPS
algorithm:
"""

using TensorKit, PEPSKit, MPSKit

md"""
## Computing a PEPS norm

We start by initializing a random infinite PEPS. Let us use normally distributed complex
entries using `randn`:
"""

ψ = InfinitePEPS(randn, ComplexF64, ComplexSpace(2), ComplexSpace(2))

md"""

To compute its norm, we have to contract a double-layer network which encodes the bra-ket
PEPS overlap ``\langle ψ | ψ \rangle``:

```@raw html
<center>
<img src="../../assets/figures/peps_norm_network.svg" alt="peps norm network" class="color-invertible" style="zoom: 180%"/>
</center>
```

In PEPSKit.jl, this structure is represented as an [`InfiniteSquareNetwork`](@ref) object,
whose effective local rank-4 constituent tensor is given by the contraction of a pair of bra
and ket [`PEPSKit.PEPSTensor`](@ref)s across their physical legs. Until now, we have always
contracted such a network using the CTMRG algorithm. Here however, we will use another
approach.

If we take out a single row of this infinite norm network, we can interpret it as a 1D
row-to-row transfer operator ``\mathbb{T}``,

```@raw html
<center>
<img src="../../assets/figures/peps_transfer_operator.svg" alt="peps transfer operator" class="color-invertible" style="zoom: 180%"/>
</center>
```

This transfer operator can be seen as an infinite chain of the effective local rank-4
tensors that make up the PEPS norm network. Since the network we want to contract can be
interpreted as the infinite power of ``\mathbb{T}``, we can contract it by finding its
leading eigenvector as a 1D MPS ``| \psi_{\text{MPS}} \rangle``, which we call the boundary
MPS. This boundary MPS should satisfy the eigenvalue equation
``\mathbb{T} | \psi_{\text{MPS}} \rangle \approx \Lambda | \psi_{\text{MPS}} \rangle``, or
diagrammatically:

```@raw html
<center>
<img src="../../assets/figures/peps_transfer_fixedpoint_equation.svg" alt="peps transfer fixedpoint equation" class="color-invertible" style="zoom: 180%"/>
</center>
```

Note that if ``\mathbb{T}`` is Hermitian, we can formulate this eigenvalue equation in terms of a
variational problem for the free energy,

```math
\begin{align} 
f &= \lim_{N \to ∞} - \frac{1}{N} \log \left( \frac{\langle \psi_{\text{MPS}} | \mathbb{T} | \psi_{\text{MPS}} \rangle}{\langle \psi_{\text{MPS}} | \psi_{\text{MPS}} \rangle} \right),
\\
&= -\log(\lambda)
\end{align}
```

where ``\lambda = \Lambda^{1/N}`` is the 'eigenvalue per site' of ``\mathbb{T}``, giving
``f`` the meaning of a free energy density.

Since the contraction of a PEPS norm network is in essence exactly the same problem as the
contraction of a 2D classical partition function, we can directly use boundary MPS
algorithms designed for 2D statistical mechanics models in this context. In particular,
we'll use the [the VUMPS algorithm](@cite vanderstraeten_tangentspace_2019) to perform the
boundary MPS contraction, and we'll call it through the [`leading_boundary`](@ref) method
from MPSKit.jl. This method precisely finds the MPS fixed point of a 1D transfer operator.

## Boundary MPS contractions with PEPSKit.jl

To use [`leading_boundary`](@ref), we first need to contruct the transfer operator
``\mathbb{T}`` as an [`MPSKit.InfiniteMPO`](@extref) object. In PEPSKit.jl, we can directly
construct the transfer operator corresponding to a PEPS norm network from a given infinite
PEPS as an [`InfiniteTransferPEPS`](@ref) object, which is a specific kind of
[`MPSKit.InfiniteMPO`](@extref).

To construct a 1D transfer operator from a 2D PEPS state, we need to specify which direction
should be facing north (`dir=1` corresponding to north, counting clockwise) and which row of
the network is selected from the north - but since we have a trivial unit cell there is only
one row here:
"""

dir = 1 ## does not rotate the partition function
row = 1
T = InfiniteTransferPEPS(ψ, dir, row)

md"""
Since we'll find the leading eigenvector of ``\mathbb{T}`` as a boundary MPS, we first need
to construct an initial guess to supply to our algorithm. We can do this using the
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
[`MPSKit.VUMPS`](@extref) algorithm:
"""

mps, env, ϵ = leading_boundary(mps₀, T, VUMPS(; tol = 1.0e-6, verbosity = 2));

md"""
The norm of the state per unit cell is then given by the expectation value
$\langle \psi_\text{MPS} | \mathbb{T} | \psi_\text{MPS} \rangle$ per site:
"""

norm_vumps = abs(prod(expectation_value(mps, T)))

md"""
This can be compared to the result obtained using CTMRG, where we see that the results
match:
"""

env_ctmrg, = leading_boundary(CTMRGEnv(ψ, ComplexSpace(20)), ψ; tol = 1.0e-6, verbosity = 2)
norm_ctmrg = abs(norm(ψ, env_ctmrg))
@show abs(norm_vumps - norm_ctmrg) / norm_vumps;

md"""
## Working with unit cells

For PEPS with non-trivial unit cells, the principle is exactly the same. The only difference
is that now the transfer operator of the PEPS norm partition function has multiple rows or
'lines', each of which can be represented by an [`InfiniteTransferPEPS`](@ref) object. Such
a multi-line transfer operator is represented by a [`PEPSKit.MultilineTransferPEPS`](@ref)
object. In this case, the boundary MPS is an [`MultilineMPS`](@extref) object, which should
be initialized by specifying a virtual space for each site in the partition function unit
cell.

First, we construct a PEPS with a $2 \times 2$ unit cell using the `unitcell` keyword
argument and then define the corresponding transfer operator, where we again specify the
direction which will be facing north:
"""

ψ_2x2 = InfinitePEPS(rand, ComplexF64, ComplexSpace(2), ComplexSpace(2); unitcell = (2, 2))
T_2x2 = PEPSKit.MultilineTransferPEPS(ψ_2x2, dir);

md"""
Now, the procedure is the same as before: We compute the norm once using VUMPS, once using CTMRG and then compare.
"""

mps₀_2x2 = initialize_mps(T_2x2, fill(ComplexSpace(20), 2, 2))
mps_2x2, = leading_boundary(mps₀_2x2, T_2x2, VUMPS(; tol = 1.0e-6, verbosity = 2))
norm_2x2_vumps = abs(prod(expectation_value(mps_2x2, T_2x2)))

env_ctmrg_2x2, = leading_boundary(
    CTMRGEnv(ψ_2x2, ComplexSpace(20)), ψ_2x2; tol = 1.0e-6, verbosity = 2
)
norm_2x2_ctmrg = abs(norm(ψ_2x2, env_ctmrg_2x2))

@show abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps;

md"""
Again, the results are compatible. Note that for larger unit cells and non-Hermitian PEPS
[the VUMPS algorithm may become unstable](@cite vanderstraeten_variational_2022), in which
case the CTMRG algorithm is recommended.

## Contracting PEPO overlaps

Using exactly the same machinery, we can contract 2D networks which encode the expectation
value of a PEPO for a given PEPS state. As an example, we can consider the overlap of the
PEPO correponding to the partition function of [3D classical Ising model](@ref e_3d_ising)
with our random PEPS from before and evaluate the overlap $\langle \psi |
T | \psi \rangle$.

The classical Ising PEPO is defined as follows:
"""

function ising_pepo(β; unitcell = (1, 1, 1))
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

T = ising_pepo(1)
transfer_pepo = InfiniteTransferPEPO(ψ, T, 1, 1)

md"""
As before, we converge the boundary MPS using VUMPS and then compute the expectation value:
"""

mps₀_pepo = initialize_mps(transfer_pepo, [ComplexSpace(20)])
mps_pepo, = leading_boundary(mps₀_pepo, transfer_pepo, VUMPS(; tol = 1.0e-6, verbosity = 2))
norm_pepo = abs(prod(expectation_value(mps_pepo, transfer_pepo)));
@show norm_pepo;

md"""
These objects and routines can be used to optimize PEPS fixed points of 3D partition
functions, see for example [Vanderstraeten et al.](@cite vanderstraeten_residual_2018)
"""
