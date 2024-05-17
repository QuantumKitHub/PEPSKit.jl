using Random
using PEPSKit
using TensorKit
using MPSKit
using LinearAlgebra

include("ising_pepo.jl")

# This example demonstrates some boundary-MPS methods for working with 2D projected
# entangled-pair states and operators.

## Computing a PEPS norm

# We start by initializing a random initial infinite PEPS
Random.seed!(29384293742893)
peps = InfinitePEPS(ComplexSpace(2), ComplexSpace(2))

# To compute its norm, we start by constructing the transfer operator corresponding to
# the partition function representing the overlap <peps|peps>
T = InfiniteTransferPEPS(peps, 1, 1)

# We then find its leading boundary MPS fixed point, where the corresponding eigenvalue
# encodes the norm of the state

# Fist we build an initial guess for the boundary MPS, choosing a bond dimension of 20
mps = PEPSKit.initializeMPS(T, [ComplexSpace(20)])

# We then find the leading boundary MPS fixed point using the VUMPS algorithm
mps, envs, ϵ = leading_boundary(mps, T, VUMPS())

# The norm of the state per unit cell is then given by the expectation value <mps|T|mps>
N = abs(prod(expectation_value(mps, T)))

# This can be compared to the result obtained using the CTMRG algorithm
ctm = leading_boundary(
    peps, CTMRG(; verbosity=1, fixedspace=true), CTMRGEnv(peps; Venv=ComplexSpace(20))
)
N´ = abs(norm(peps, ctm))

@show abs(N - N´) / N

## Working with unit cells

# For PEPS with non-trivial unit cells, the principle is exactly the same.
# The only difference is that now the transfer operator of the PEPS norm partition function
# has multiple lines, each of which can be represented by an `InfiniteTransferPEPS` object.
# Such a multi-line transfer operator is represented by a `TransferPEPSMultiline` object.
# In this case, the boundary MPS is an `MPSMultiline` object, which should be initialized
# by specifying a virtual space for each site in the partition function unit cell.

peps2 = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2))
T2 = PEPSKit.TransferPEPSMultiline(peps2, 1)

mps2 = PEPSKit.initializeMPS(T2, fill(ComplexSpace(20), 2, 2))
mps2, envs2, ϵ = leading_boundary(mps2, T2, VUMPS())
N2 = abs(prod(expectation_value(mps2, T2)))

ctm2 = leading_boundary(
    peps2, CTMRG(; verbosity=1, fixedspace=true), CTMRGEnv(peps2; Venv=ComplexSpace(20))
)
N2´ = abs(norm(peps2, ctm2))

@show abs(N2 - N2´) / N2

# Note that for larger unit cells and non-Hermitian PEPS the VUMPS algorithm may become
# unstable, in which case the CTMRG algorithm is recommended.

## Contracting PEPO overlaps

# Using exactly the same machinery, we can contract partition functions which encode the
# expectation value of a PEPO for a given PEPS state.
# As an example, we can consider the overlap of the PEPO correponding to the partition
# function of 3D classical ising model with our random PEPS from before and evaluate
# <peps|O|peps>.

pepo = ising_pepo(1)
T3 = InfiniteTransferPEPO(peps, pepo, 1, 1)

mps3 = PEPSKit.initializeMPS(T3, [ComplexSpace(20)])
mps3, envs3, ϵ = leading_boundary(mps3, T3, VUMPS())
@show N3 = abs(prod(expectation_value(mps3, T3)))

# These objects and routines can be used to optimize PEPS fixed points of 3D partition
# functions, see for example https://arxiv.org/abs/1805.10598

nothing
