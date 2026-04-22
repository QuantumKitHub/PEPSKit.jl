```@meta
EditURL = "../../../../examples/xxz/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/xxz/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/xxz/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/xxz)


# Néel order in the $U(1)$-symmetric XXZ model

Here, we want to look at a special case of the Heisenberg model, where the $x$ and $y$
couplings are equal, called the XXZ model

```math
H_0 = J \big(\sum_{\langle i, j \rangle} S_i^x S_j^x + S_i^y S_j^y + \Delta S_i^z S_j^z \big) .
```

For appropriate $\Delta$, the model enters an antiferromagnetic phase (Néel order) which we
will force by adding staggered magnetic charges to $H_0$. Furthermore, since the XXZ
Hamiltonian obeys a $U(1)$ symmetry, we will make use of that and work with $U(1)$-symmetric
PEPS and CTMRG environments. For simplicity, we will consider spin-$1/2$ operators.

But first, let's make this example deterministic and import the required packages:

````julia
using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528935);
````

## Constructing the model

Let us define the $U(1)$-symmetric XXZ Hamiltonian on a $2 \times 2$ unit cell with the
parameters:

````julia
J = 1.0
Delta = 1.0
spin = 1 // 2
symmetry = U1Irrep
lattice = InfiniteSquare(2, 2)
H₀ = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin);
````

This ensures that our PEPS ansatz can support the bipartite Néel order. As discussed above,
we encode the Néel order directly in the ansatz by adding staggered auxiliary physical
charges:

````julia
S_aux = [
    U1Irrep(-1 // 2) U1Irrep(1 // 2)
    U1Irrep(1 // 2) U1Irrep(-1 // 2)
]
H = add_physical_charge(H₀, S_aux);
````

## Specifying the symmetric virtual spaces

Before we create an initial PEPS and CTM environment, we need to think about which
symmetric spaces we need to construct. Since we want to exploit the global $U(1)$ symmetry
of the model, we will use TensorKit's `U1Space`s where we specify dimensions for each
symmetry sector. From the virtual spaces, we will need to construct a unit cell (a matrix)
of spaces which will be supplied to the PEPS constructor. The same is true for the physical
spaces, which can be extracted directly from the Hamiltonian `LocalOperator`:

````julia
V_peps = U1Space(0 => 2, 1 => 1, -1 => 1)
V_env = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2)
virtual_spaces = fill(V_peps, size(lattice)...)
physical_spaces = physicalspace(H)
````

````
2×2 Matrix{TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}}:
 Rep[U₁](…) of dim 2   Rep[U₁](…) of dim 2
 Rep[U₁](…) of dim 2  Rep[U₁](…) of dim 2
````

## Ground state search

From this point onwards it's business as usual: Create an initial PEPS and environment
(using the symmetric spaces), specify the algorithmic parameters and optimize:

````julia
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, trunc = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, alg = :eigsolver, maxiter = 10, iterscheme = :fixed)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, maxiter = 85, ls_maxiter = 3, ls_maxfg = 3)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -2.356413456811e+03 +3.307968169629e+02im	err = 1.0000e+00
[ Info: CTMRG conv 30:	obj = +6.245129734283e+03 -4.008506948594e-08im	err = 5.3638614844e-09	time = 14.24 sec

````

Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian and check the
resulting ground state energy per site using our $(2 \times 2)$ unit cell. Note that the
optimization might take a while since precompilation of symmetric AD code takes longer and
because symmetric tensors do create a bit of overhead (which does pay off at larger bond and
environment dimensions):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity = 3
)
@show E / prod(size(lattice));
````

````
[ Info: LBFGS: initializing with f = -1.385136095079e-01, ‖∇f‖ = 1.2184e+00
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -2.44e-02, ϕ - ϕ₀ = -4.56e-01
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/linesearches.jl:151
[ Info: LBFGS: iter    1, Δt 45.37 s: f = -5.947088553357e-01, ‖∇f‖ = 3.7329e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -7.72e-03, ϕ - ϕ₀ = -1.52e+00
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/linesearches.jl:151
[ Info: LBFGS: iter    2, Δt 45.67 s: f = -2.114273976569e+00, ‖∇f‖ = 2.9121e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, Δt  9.49 s: f = -2.218657558447e+00, ‖∇f‖ = 1.4788e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, Δt 28.19 s: f = -2.473597365661e+00, ‖∇f‖ = 1.2506e+00, α = 3.17e+00, m = 2, nfg = 3
[ Info: LBFGS: iter    5, Δt  9.28 s: f = -2.546159342811e+00, ‖∇f‖ = 1.4463e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, Δt  7.74 s: f = -2.614645567632e+00, ‖∇f‖ = 4.0554e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, Δt  8.85 s: f = -2.622673934023e+00, ‖∇f‖ = 1.8054e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, Δt  7.07 s: f = -2.626310260611e+00, ‖∇f‖ = 1.7749e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, Δt  8.59 s: f = -2.632769137184e+00, ‖∇f‖ = 1.8586e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, Δt  6.53 s: f = -2.639694621494e+00, ‖∇f‖ = 2.2500e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, Δt  8.00 s: f = -2.644827934020e+00, ‖∇f‖ = 1.2801e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, Δt  5.91 s: f = -2.646459705819e+00, ‖∇f‖ = 6.7575e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, Δt  6.22 s: f = -2.647499600831e+00, ‖∇f‖ = 6.0731e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, Δt  7.75 s: f = -2.648703045894e+00, ‖∇f‖ = 7.1313e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, Δt  6.29 s: f = -2.650602127388e+00, ‖∇f‖ = 9.3675e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, Δt  7.53 s: f = -2.652309117542e+00, ‖∇f‖ = 8.3679e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, Δt  5.86 s: f = -2.654182949224e+00, ‖∇f‖ = 9.5661e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, Δt  7.49 s: f = -2.655830713358e+00, ‖∇f‖ = 1.4282e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt  5.62 s: f = -2.658506508894e+00, ‖∇f‖ = 8.6260e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, Δt  5.98 s: f = -2.660101929403e+00, ‖∇f‖ = 5.5569e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, Δt  7.27 s: f = -2.660655802769e+00, ‖∇f‖ = 5.0089e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, Δt  5.69 s: f = -2.661713752636e+00, ‖∇f‖ = 6.6020e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt  6.23 s: f = -2.663782967628e+00, ‖∇f‖ = 1.4168e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt  7.11 s: f = -2.664843906404e+00, ‖∇f‖ = 1.3559e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt  6.05 s: f = -2.666211885495e+00, ‖∇f‖ = 6.7533e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt  7.18 s: f = -2.666722965867e+00, ‖∇f‖ = 5.1877e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt  6.02 s: f = -2.667030607084e+00, ‖∇f‖ = 4.7362e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt  7.70 s: f = -2.668170313888e+00, ‖∇f‖ = 5.6311e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt  6.04 s: f = -2.668423708832e+00, ‖∇f‖ = 1.1943e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt  7.68 s: f = -2.669339639442e+00, ‖∇f‖ = 4.0859e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt  5.65 s: f = -2.669607092068e+00, ‖∇f‖ = 3.0582e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt  6.03 s: f = -2.669888674448e+00, ‖∇f‖ = 3.6474e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt  7.56 s: f = -2.670409260559e+00, ‖∇f‖ = 5.7241e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt  6.35 s: f = -2.670955670621e+00, ‖∇f‖ = 6.0848e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt  7.45 s: f = -2.671400740936e+00, ‖∇f‖ = 4.4890e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt  6.16 s: f = -2.671654819742e+00, ‖∇f‖ = 2.3662e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt  7.47 s: f = -2.671805688410e+00, ‖∇f‖ = 2.3809e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt  6.15 s: f = -2.672069418094e+00, ‖∇f‖ = 3.7660e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt  7.50 s: f = -2.672391998692e+00, ‖∇f‖ = 4.6082e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt  6.15 s: f = -2.672631813229e+00, ‖∇f‖ = 2.8972e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt  7.60 s: f = -2.672757646725e+00, ‖∇f‖ = 2.0266e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt  5.76 s: f = -2.672874968531e+00, ‖∇f‖ = 2.3891e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt  6.09 s: f = -2.673085935987e+00, ‖∇f‖ = 3.1467e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt  7.17 s: f = -2.673264955042e+00, ‖∇f‖ = 5.1067e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt  6.11 s: f = -2.673441653001e+00, ‖∇f‖ = 2.2050e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt  7.64 s: f = -2.673518614777e+00, ‖∇f‖ = 1.6760e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt  6.03 s: f = -2.673610642660e+00, ‖∇f‖ = 2.1356e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt  7.60 s: f = -2.673749855167e+00, ‖∇f‖ = 3.0804e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt  6.08 s: f = -2.673964481832e+00, ‖∇f‖ = 2.8038e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt  6.24 s: f = -2.674085336827e+00, ‖∇f‖ = 3.7211e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt  7.20 s: f = -2.674190542900e+00, ‖∇f‖ = 1.7211e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt  6.09 s: f = -2.674244307002e+00, ‖∇f‖ = 1.4385e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt  7.45 s: f = -2.674308652203e+00, ‖∇f‖ = 1.8132e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt  6.28 s: f = -2.674434242130e+00, ‖∇f‖ = 2.0442e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 13.84 s: f = -2.674482156825e+00, ‖∇f‖ = 2.0921e-02, α = 3.35e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   56, Δt  7.94 s: f = -2.674544199355e+00, ‖∇f‖ = 1.1684e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt  5.86 s: f = -2.674594731099e+00, ‖∇f‖ = 1.2135e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt  6.35 s: f = -2.674646300923e+00, ‖∇f‖ = 1.7065e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt  7.54 s: f = -2.674708781785e+00, ‖∇f‖ = 1.4159e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt  6.33 s: f = -2.674769125064e+00, ‖∇f‖ = 1.4517e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt  7.79 s: f = -2.674820428656e+00, ‖∇f‖ = 1.9999e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt  6.10 s: f = -2.674864524466e+00, ‖∇f‖ = 1.5512e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt  8.24 s: f = -2.674936594389e+00, ‖∇f‖ = 1.4904e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt  6.26 s: f = -2.674955282643e+00, ‖∇f‖ = 1.9377e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt  7.47 s: f = -2.674989475402e+00, ‖∇f‖ = 1.0749e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt  5.94 s: f = -2.675008560457e+00, ‖∇f‖ = 9.3873e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt  7.63 s: f = -2.675035121673e+00, ‖∇f‖ = 1.0893e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt  5.88 s: f = -2.675093424988e+00, ‖∇f‖ = 1.5948e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 13.66 s: f = -2.675123092585e+00, ‖∇f‖ = 1.8297e-02, α = 5.07e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   70, Δt  6.11 s: f = -2.675158145862e+00, ‖∇f‖ = 1.0411e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt  8.91 s: f = -2.675184397098e+00, ‖∇f‖ = 7.5551e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt  6.70 s: f = -2.675202127278e+00, ‖∇f‖ = 1.0326e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt  9.29 s: f = -2.675232074331e+00, ‖∇f‖ = 1.0276e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 15.73 s: f = -2.675249906405e+00, ‖∇f‖ = 1.7745e-02, α = 3.56e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   75, Δt  6.61 s: f = -2.675277747740e+00, ‖∇f‖ = 8.6877e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt  6.90 s: f = -2.675295012134e+00, ‖∇f‖ = 5.9675e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt  9.00 s: f = -2.675309457155e+00, ‖∇f‖ = 8.1916e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt  6.50 s: f = -2.675327390804e+00, ‖∇f‖ = 1.1679e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt  8.11 s: f = -2.675346033915e+00, ‖∇f‖ = 7.3995e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, Δt  6.22 s: f = -2.675361840744e+00, ‖∇f‖ = 6.2470e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, Δt  7.71 s: f = -2.675372633878e+00, ‖∇f‖ = 1.0378e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, Δt  6.16 s: f = -2.675385864468e+00, ‖∇f‖ = 7.9203e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, Δt  8.23 s: f = -2.675408682962e+00, ‖∇f‖ = 5.6544e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, Δt  6.10 s: f = -2.675422272234e+00, ‖∇f‖ = 7.8184e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 18.43 m: f = -2.675439882383e+00, ‖∇f‖ = 7.8348e-03
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/lbfgs.jl:199
E / prod(size(lattice)) = -0.6688599705957462

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian
as in the [Heisenberg example](@ref examples_heisenberg). In that example, with a
non-symmetric $D=2$ PEPS simulation, we reached a ground-state energy per site of around
$E_\text{D=2} = -0.6625\dots$. Again comparing against [Sandvik's](@cite
sandvik_computational_2011) accurate QMC estimate ``E_{\text{ref}}=−0.6694421``, we see that
we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

