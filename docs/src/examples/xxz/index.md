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
[ Info: CTMRG conv 30:	obj = +6.245129734283e+03 -4.009325493826e-08im	err = 5.3638614449e-09	time = 7.21 sec

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
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/linesearches.jl:148
[ Info: LBFGS: iter    1, Δt  1.47 m: f = -5.947088555354e-01, ‖∇f‖ = 3.7329e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -7.72e-03, ϕ - ϕ₀ = -1.52e+00
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/linesearches.jl:148
[ Info: LBFGS: iter    2, Δt  1.37 m: f = -2.114273975713e+00, ‖∇f‖ = 2.9121e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, Δt 16.27 s: f = -2.218657557832e+00, ‖∇f‖ = 1.4788e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, Δt 50.77 s: f = -2.473597362695e+00, ‖∇f‖ = 1.2506e+00, α = 3.17e+00, m = 2, nfg = 3
[ Info: LBFGS: iter    5, Δt 15.40 s: f = -2.546159338872e+00, ‖∇f‖ = 1.4463e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, Δt 16.81 s: f = -2.614645567157e+00, ‖∇f‖ = 4.0554e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, Δt 15.00 s: f = -2.622673933783e+00, ‖∇f‖ = 1.8054e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, Δt 15.49 s: f = -2.626310260551e+00, ‖∇f‖ = 1.7749e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, Δt 13.94 s: f = -2.632769138215e+00, ‖∇f‖ = 1.8586e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, Δt 14.24 s: f = -2.639694625673e+00, ‖∇f‖ = 2.2500e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, Δt 12.62 s: f = -2.644827933644e+00, ‖∇f‖ = 1.2801e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, Δt 13.99 s: f = -2.646459706216e+00, ‖∇f‖ = 6.7575e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, Δt 12.48 s: f = -2.647499601247e+00, ‖∇f‖ = 6.0731e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, Δt 13.90 s: f = -2.648703044472e+00, ‖∇f‖ = 7.1312e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, Δt 12.38 s: f = -2.650602130567e+00, ‖∇f‖ = 9.3675e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, Δt 12.72 s: f = -2.652309127838e+00, ‖∇f‖ = 8.3679e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, Δt 12.14 s: f = -2.654182955360e+00, ‖∇f‖ = 9.5661e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, Δt 12.51 s: f = -2.655830722048e+00, ‖∇f‖ = 1.4282e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, Δt 13.91 s: f = -2.658506524808e+00, ‖∇f‖ = 8.6259e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, Δt 12.28 s: f = -2.660101934378e+00, ‖∇f‖ = 5.5568e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, Δt 13.67 s: f = -2.660655823922e+00, ‖∇f‖ = 5.0087e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, Δt 12.20 s: f = -2.661713913904e+00, ‖∇f‖ = 6.6024e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, Δt 13.88 s: f = -2.663783161449e+00, ‖∇f‖ = 1.4168e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, Δt 12.33 s: f = -2.664843824225e+00, ‖∇f‖ = 1.3560e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, Δt 13.71 s: f = -2.666211864482e+00, ‖∇f‖ = 6.7535e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, Δt 12.34 s: f = -2.666722906773e+00, ‖∇f‖ = 5.1877e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, Δt 12.35 s: f = -2.667030535551e+00, ‖∇f‖ = 4.7362e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, Δt 13.93 s: f = -2.668169807778e+00, ‖∇f‖ = 5.6321e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, Δt 12.64 s: f = -2.668423674818e+00, ‖∇f‖ = 1.1940e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, Δt 13.82 s: f = -2.669339425071e+00, ‖∇f‖ = 4.0856e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, Δt 12.36 s: f = -2.669606925028e+00, ‖∇f‖ = 3.0584e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, Δt 13.86 s: f = -2.669888443527e+00, ‖∇f‖ = 3.6473e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, Δt 12.59 s: f = -2.670409100956e+00, ‖∇f‖ = 5.7239e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, Δt 14.06 s: f = -2.670955476785e+00, ‖∇f‖ = 6.0862e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, Δt 12.54 s: f = -2.671400581183e+00, ‖∇f‖ = 4.4907e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, Δt 13.84 s: f = -2.671654670301e+00, ‖∇f‖ = 2.3660e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, Δt 12.53 s: f = -2.671805543674e+00, ‖∇f‖ = 2.3806e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, Δt 12.62 s: f = -2.672069196257e+00, ‖∇f‖ = 3.7666e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, Δt 13.93 s: f = -2.672392041467e+00, ‖∇f‖ = 4.6014e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, Δt 12.58 s: f = -2.672631814576e+00, ‖∇f‖ = 2.8983e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, Δt 13.94 s: f = -2.672757830427e+00, ‖∇f‖ = 2.0269e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, Δt 11.54 s: f = -2.672875298674e+00, ‖∇f‖ = 2.3893e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, Δt 12.91 s: f = -2.673086282043e+00, ‖∇f‖ = 3.1487e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, Δt 11.56 s: f = -2.673264734617e+00, ‖∇f‖ = 5.1144e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, Δt 11.70 s: f = -2.673441586270e+00, ‖∇f‖ = 2.2014e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, Δt 13.22 s: f = -2.673518413423e+00, ‖∇f‖ = 1.6755e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, Δt 11.73 s: f = -2.673610437186e+00, ‖∇f‖ = 2.1374e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, Δt 12.99 s: f = -2.673749787831e+00, ‖∇f‖ = 3.0825e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, Δt 11.81 s: f = -2.673963455728e+00, ‖∇f‖ = 2.8112e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, Δt 13.16 s: f = -2.674085248803e+00, ‖∇f‖ = 3.6768e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, Δt 11.39 s: f = -2.674188984088e+00, ‖∇f‖ = 1.7117e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, Δt 11.56 s: f = -2.674242447315e+00, ‖∇f‖ = 1.4444e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, Δt 12.95 s: f = -2.674306699476e+00, ‖∇f‖ = 1.8187e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, Δt 11.87 s: f = -2.674433434449e+00, ‖∇f‖ = 2.0657e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, Δt 24.94 s: f = -2.674481180296e+00, ‖∇f‖ = 2.0935e-02, α = 3.31e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   56, Δt 13.27 s: f = -2.674543091778e+00, ‖∇f‖ = 1.1697e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, Δt 11.73 s: f = -2.674593597475e+00, ‖∇f‖ = 1.2064e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, Δt 13.98 s: f = -2.674645033379e+00, ‖∇f‖ = 1.7233e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, Δt 13.19 s: f = -2.674707076560e+00, ‖∇f‖ = 1.4282e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, Δt 12.30 s: f = -2.674765993748e+00, ‖∇f‖ = 1.5331e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, Δt 13.45 s: f = -2.674818411605e+00, ‖∇f‖ = 1.7528e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, Δt 11.79 s: f = -2.674860141812e+00, ‖∇f‖ = 1.5281e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, Δt 13.29 s: f = -2.674937524252e+00, ‖∇f‖ = 1.3781e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, Δt 11.59 s: f = -2.674948199372e+00, ‖∇f‖ = 2.8631e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, Δt 22.94 s: f = -2.674990650090e+00, ‖∇f‖ = 9.4163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, Δt 11.50 s: f = -2.675004596824e+00, ‖∇f‖ = 7.9770e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, Δt 12.75 s: f = -2.675026772162e+00, ‖∇f‖ = 1.1890e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, Δt 11.54 s: f = -2.675068849496e+00, ‖∇f‖ = 1.5839e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, Δt 12.83 s: f = -2.675131833485e+00, ‖∇f‖ = 1.9865e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, Δt 23.26 s: f = -2.675161486689e+00, ‖∇f‖ = 1.6149e-02, α = 3.77e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   71, Δt 12.81 s: f = -2.675191940653e+00, ‖∇f‖ = 7.4164e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, Δt 11.69 s: f = -2.675210048264e+00, ‖∇f‖ = 8.1107e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, Δt 12.81 s: f = -2.675226236810e+00, ‖∇f‖ = 1.0563e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, Δt 11.71 s: f = -2.675255865322e+00, ‖∇f‖ = 1.5000e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, Δt 12.83 s: f = -2.675284908686e+00, ‖∇f‖ = 9.5271e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, Δt 11.47 s: f = -2.675303880609e+00, ‖∇f‖ = 6.1567e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, Δt 13.03 s: f = -2.675316351383e+00, ‖∇f‖ = 8.2404e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, Δt 11.71 s: f = -2.675331006484e+00, ‖∇f‖ = 8.6196e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, Δt 11.73 s: f = -2.675352594041e+00, ‖∇f‖ = 1.1186e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, Δt 13.08 s: f = -2.675368505391e+00, ‖∇f‖ = 1.0487e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, Δt 11.58 s: f = -2.675379342123e+00, ‖∇f‖ = 5.6587e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, Δt 12.54 s: f = -2.675386556147e+00, ‖∇f‖ = 5.4612e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, Δt 11.64 s: f = -2.675400703567e+00, ‖∇f‖ = 7.6013e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, Δt 13.11 s: f = -2.675419793189e+00, ‖∇f‖ = 1.4146e-02, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 26.03 m: f = -2.675438660313e+00, ‖∇f‖ = 7.9074e-03
└ @ OptimKit ~/.julia/packages/OptimKit/dRsBo/src/lbfgs.jl:199
E / prod(size(lattice)) = -0.6688596650783208

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian
as in the [Heisenberg example](@ref examples_heisenberg). In that example, with a
non-symmetric $D=2$ PEPS simulation, we reached a ground-state energy per site of around
$E_\text{D=2} = -0.6625\dots$. Again comparing against [Sandvik's](@cite
sandvik_computational_2011) accurate QMC estimate ``E_{\text{ref}}=−0.6694421``, we see that
we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

