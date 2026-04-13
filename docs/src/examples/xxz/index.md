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
gradient_alg = (; tol = 1.0e-6, alg = :eigsolver, maxiter = 10, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, maxiter = 85, ls_maxiter = 3, ls_maxfg = 3)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -2.356413456811e+03 +3.307968169629e+02im	err = 1.0000e+00
[ Info: CTMRG conv 30:	obj = +6.245129734283e+03 -4.009962140117e-08im	err = 5.3638613065e-09	time = 1.07 sec

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
[ Info: LBFGS: iter    1, Δt 49.56 s: f = -5.947088553802e-01, ‖∇f‖ = 3.7329e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -7.72e-03, ϕ - ϕ₀ = -1.52e+00
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/linesearches.jl:151
[ Info: LBFGS: iter    2, Δt 45.67 s: f = -2.114273976232e+00, ‖∇f‖ = 2.9121e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, Δt  8.71 s: f = -2.218657556737e+00, ‖∇f‖ = 1.4788e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, Δt 27.86 s: f = -2.473597362493e+00, ‖∇f‖ = 1.2506e+00, α = 3.17e+00, m = 2, nfg = 3
[ Info: LBFGS: iter    5, Δt  9.06 s: f = -2.546159337642e+00, ‖∇f‖ = 1.4463e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, Δt  8.67 s: f = -2.614645566780e+00, ‖∇f‖ = 4.0554e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, Δt  9.03 s: f = -2.622673933972e+00, ‖∇f‖ = 1.8054e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, Δt  8.40 s: f = -2.626310260618e+00, ‖∇f‖ = 1.7749e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, Δt  7.93 s: f = -2.632769136711e+00, ‖∇f‖ = 1.8586e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, Δt  7.61 s: f = -2.639694621229e+00, ‖∇f‖ = 2.2500e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, Δt  7.16 s: f = -2.644827933828e+00, ‖∇f‖ = 1.2801e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, Δt  7.50 s: f = -2.646459705942e+00, ‖∇f‖ = 6.7575e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, Δt  6.99 s: f = -2.647499600848e+00, ‖∇f‖ = 6.0731e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, Δt  7.50 s: f = -2.648703045941e+00, ‖∇f‖ = 7.1313e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, Δt  7.11 s: f = -2.650602127531e+00, ‖∇f‖ = 9.3675e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, Δt  6.94 s: f = -2.652309117887e+00, ‖∇f‖ = 8.3679e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, Δt  7.16 s: f = -2.654182949559e+00, ‖∇f‖ = 9.5661e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, Δt  7.49 s: f = -2.655830713827e+00, ‖∇f‖ = 1.4282e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt  7.23 s: f = -2.658506509688e+00, ‖∇f‖ = 8.6259e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, Δt  7.64 s: f = -2.660101929784e+00, ‖∇f‖ = 5.5569e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, Δt  6.90 s: f = -2.660655804151e+00, ‖∇f‖ = 5.0089e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, Δt  6.77 s: f = -2.661713763966e+00, ‖∇f‖ = 6.6021e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt  7.52 s: f = -2.663782980193e+00, ‖∇f‖ = 1.4168e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt  7.16 s: f = -2.664843902331e+00, ‖∇f‖ = 1.3559e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt  7.53 s: f = -2.666211884109e+00, ‖∇f‖ = 6.7533e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt  7.02 s: f = -2.666722962130e+00, ‖∇f‖ = 5.1877e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt  7.53 s: f = -2.667030602502e+00, ‖∇f‖ = 4.7362e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt  7.18 s: f = -2.668170280191e+00, ‖∇f‖ = 5.6312e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt  7.54 s: f = -2.668423712729e+00, ‖∇f‖ = 1.1943e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt  7.17 s: f = -2.669339626497e+00, ‖∇f‖ = 4.0858e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt  7.49 s: f = -2.669607082478e+00, ‖∇f‖ = 3.0582e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt  6.99 s: f = -2.669888660598e+00, ‖∇f‖ = 3.6474e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt  7.62 s: f = -2.670409252201e+00, ‖∇f‖ = 5.7241e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt  7.24 s: f = -2.670955657881e+00, ‖∇f‖ = 6.0849e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt  7.57 s: f = -2.671400731193e+00, ‖∇f‖ = 4.4891e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt  7.01 s: f = -2.671654809276e+00, ‖∇f‖ = 2.3662e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt  7.38 s: f = -2.671805678430e+00, ‖∇f‖ = 2.3809e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt  6.97 s: f = -2.672069404251e+00, ‖∇f‖ = 3.7660e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt  7.29 s: f = -2.672392002437e+00, ‖∇f‖ = 4.6077e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt  6.99 s: f = -2.672631813806e+00, ‖∇f‖ = 2.8973e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt  7.42 s: f = -2.672757659661e+00, ‖∇f‖ = 2.0266e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt  6.29 s: f = -2.672874991777e+00, ‖∇f‖ = 2.3891e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt  6.72 s: f = -2.673085962228e+00, ‖∇f‖ = 3.1468e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt  6.38 s: f = -2.673264939913e+00, ‖∇f‖ = 5.1073e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt  6.77 s: f = -2.673441648495e+00, ‖∇f‖ = 2.2047e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt  6.31 s: f = -2.673518600682e+00, ‖∇f‖ = 1.6760e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt  6.72 s: f = -2.673610627656e+00, ‖∇f‖ = 2.1357e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt  6.49 s: f = -2.673749851382e+00, ‖∇f‖ = 3.0805e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt  6.83 s: f = -2.673964407099e+00, ‖∇f‖ = 2.8044e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt  6.54 s: f = -2.674085306498e+00, ‖∇f‖ = 3.7187e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt  6.70 s: f = -2.674190416395e+00, ‖∇f‖ = 1.7204e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt  6.90 s: f = -2.674244147958e+00, ‖∇f‖ = 1.4388e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt  6.69 s: f = -2.674308492367e+00, ‖∇f‖ = 1.8135e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt  6.99 s: f = -2.674434142909e+00, ‖∇f‖ = 2.0460e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 13.65 s: f = -2.674482027661e+00, ‖∇f‖ = 2.0923e-02, α = 3.35e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   56, Δt  6.68 s: f = -2.674544061262e+00, ‖∇f‖ = 1.1687e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt  7.07 s: f = -2.674594606079e+00, ‖∇f‖ = 1.2128e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt  6.64 s: f = -2.674646184030e+00, ‖∇f‖ = 1.7080e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt  7.11 s: f = -2.674708616316e+00, ‖∇f‖ = 1.4174e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt  6.76 s: f = -2.674768771588e+00, ‖∇f‖ = 1.4598e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt  7.12 s: f = -2.674820230487e+00, ‖∇f‖ = 1.9700e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt  6.68 s: f = -2.674864015912e+00, ‖∇f‖ = 1.5491e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt  6.87 s: f = -2.674936674188e+00, ‖∇f‖ = 1.4360e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt  6.57 s: f = -2.674957688553e+00, ‖∇f‖ = 2.0196e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt  6.80 s: f = -2.674990714506e+00, ‖∇f‖ = 1.0037e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt  6.30 s: f = -2.675007817715e+00, ‖∇f‖ = 9.4268e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt  6.97 s: f = -2.675032496794e+00, ‖∇f‖ = 1.1461e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt  6.68 s: f = -2.675089463603e+00, ‖∇f‖ = 1.4806e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt  7.01 s: f = -2.675108881011e+00, ‖∇f‖ = 2.8465e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt  6.60 s: f = -2.675166443058e+00, ‖∇f‖ = 1.1529e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, Δt  7.03 s: f = -2.675186479546e+00, ‖∇f‖ = 6.7512e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt  6.69 s: f = -2.675204431516e+00, ‖∇f‖ = 8.4356e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt  6.98 s: f = -2.675227128219e+00, ‖∇f‖ = 1.1948e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt  6.74 s: f = -2.675257941898e+00, ‖∇f‖ = 1.3696e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt  7.10 s: f = -2.675283294818e+00, ‖∇f‖ = 9.3807e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt  6.73 s: f = -2.675300545094e+00, ‖∇f‖ = 6.3181e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt  7.08 s: f = -2.675312515675e+00, ‖∇f‖ = 8.9126e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt  6.76 s: f = -2.675328270454e+00, ‖∇f‖ = 7.2766e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt  7.09 s: f = -2.675354289574e+00, ‖∇f‖ = 7.6916e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, Δt 13.87 s: f = -2.675364316717e+00, ‖∇f‖ = 9.2305e-03, α = 4.61e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   81, Δt  6.72 s: f = -2.675376292963e+00, ‖∇f‖ = 6.5369e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, Δt  7.11 s: f = -2.675389682288e+00, ‖∇f‖ = 7.1072e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, Δt  6.63 s: f = -2.675405538777e+00, ‖∇f‖ = 9.7469e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, Δt  7.06 s: f = -2.675421118041e+00, ‖∇f‖ = 7.2757e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 12.42 m: f = -2.675438005792e+00, ‖∇f‖ = 6.4678e-03
└ @ OptimKit ~/.julia/packages/OptimKit/OEwMx/src/lbfgs.jl:199
E / prod(size(lattice)) = -0.6688595014480129

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian
as in the [Heisenberg example](@ref examples_heisenberg). In that example, with a
non-symmetric $D=2$ PEPS simulation, we reached a ground-state energy per site of around
$E_\text{D=2} = -0.6625\dots$. Again comparing against [Sandvik's](@cite
sandvik_computational_2011) accurate QMC estimate ``E_{\text{ref}}=−0.6694421``, we see that
we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

