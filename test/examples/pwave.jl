using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# Initialize parameters
unitcell = (2, 2)
H = pwave_superconductor(InfiniteSquare(unitcell...))
Dbond = 2
χenv = 16

# initialize states
Pspace = Vect[FermionParity](0 => 1, 1 => 1)
Vspace = Vect[FermionParity](0 => Dbond ÷ 2, 1 => Dbond ÷ 2)
Envspace = Vect[FermionParity](0 => χenv ÷ 2, 1 => χenv ÷ 2)
Random.seed!(91283219347)
peps₀ = InfinitePEPS(Pspace, Vspace, Vspace; unitcell)
env₀, = leading_boundary(CTMRGEnv(peps₀, Envspace), peps₀)

# find fixedpoint
_, _, E, = fixedpoint(H, peps₀, env₀; tol = 1.0e-3, optimizer_alg = (; maxiter = 10))

# comparison with Gaussian PEPS minimum at D=2 on 1000x1000 square lattice with aPBC
@test E / prod(size(peps₀)) ≈ -2.6241 atol = 5.0e-2
