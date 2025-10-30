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
 Rep[TensorKitSectors.U₁](0=>1, 1=>1)   Rep[TensorKitSectors.U₁](0=>1, -1=>1)
 Rep[TensorKitSectors.U₁](0=>1, -1=>1)  Rep[TensorKitSectors.U₁](0=>1, 1=>1)
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
[ Info: CTMRG conv 30:	obj = +6.245129734283e+03 -4.008688847534e-08im	err = 5.3638617378e-09	time = 13.26 sec

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
[ Info: LBFGS: initializing with f = -0.138513609508, ‖∇f‖ = 1.2196e+00
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -2.40e-02, ϕ - ϕ₀ = -4.51e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  398.63 s: f = -0.589257888877, ‖∇f‖ = 3.6887e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -7.97e-03, ϕ - ϕ₀ = -1.52e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  471.79 s: f = -2.110619430506, ‖∇f‖ = 3.0001e+00, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  489.29 s: f = -2.211007715638, ‖∇f‖ = 1.4524e+00, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  540.15 s: f = -2.468824623210, ‖∇f‖ = 1.2356e+00, α = 3.20e+00, m = 2, nfg = 3
[ Info: LBFGS: iter    5, time  557.26 s: f = -2.531277644141, ‖∇f‖ = 1.6029e+00, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  573.35 s: f = -2.612553859919, ‖∇f‖ = 4.2867e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  589.11 s: f = -2.622279660579, ‖∇f‖ = 1.9145e-01, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  603.99 s: f = -2.626484058783, ‖∇f‖ = 1.7329e-01, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  614.25 s: f = -2.631986947540, ‖∇f‖ = 1.8511e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  623.59 s: f = -2.639349412387, ‖∇f‖ = 1.9572e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  632.73 s: f = -2.644232912048, ‖∇f‖ = 1.6746e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, time  641.13 s: f = -2.646535544453, ‖∇f‖ = 6.8267e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, time  649.63 s: f = -2.647359319818, ‖∇f‖ = 5.8837e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  658.45 s: f = -2.648538593263, ‖∇f‖ = 6.8261e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  667.16 s: f = -2.650130908354, ‖∇f‖ = 7.0878e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  675.47 s: f = -2.651987522211, ‖∇f‖ = 8.2170e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time  682.87 s: f = -2.653758642111, ‖∇f‖ = 1.0420e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time  691.17 s: f = -2.655421985606, ‖∇f‖ = 1.3031e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  699.68 s: f = -2.657803102660, ‖∇f‖ = 9.4755e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time  708.34 s: f = -2.659668183898, ‖∇f‖ = 7.3133e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time  716.68 s: f = -2.660316016122, ‖∇f‖ = 4.9501e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time  724.73 s: f = -2.661149682031, ‖∇f‖ = 6.8936e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  732.96 s: f = -2.662868297230, ‖∇f‖ = 1.1740e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  741.23 s: f = -2.664663245986, ‖∇f‖ = 1.3179e-01, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  757.59 s: f = -2.665627671142, ‖∇f‖ = 1.0486e-01, α = 4.29e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   26, time  765.97 s: f = -2.666450510119, ‖∇f‖ = 4.5156e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  774.07 s: f = -2.666835911121, ‖∇f‖ = 4.8567e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  782.21 s: f = -2.667478452332, ‖∇f‖ = 7.2290e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  790.47 s: f = -2.668521250487, ‖∇f‖ = 8.5697e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  807.83 s: f = -2.669003677417, ‖∇f‖ = 9.9343e-02, α = 4.16e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   31, time  816.07 s: f = -2.669690963961, ‖∇f‖ = 4.0493e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  824.28 s: f = -2.669998638777, ‖∇f‖ = 3.1202e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  831.93 s: f = -2.670307400955, ‖∇f‖ = 4.9386e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  840.36 s: f = -2.670842812036, ‖∇f‖ = 6.0221e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  857.54 s: f = -2.671142652515, ‖∇f‖ = 7.1490e-02, α = 5.55e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   36, time  866.09 s: f = -2.671543536411, ‖∇f‖ = 3.6309e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  875.07 s: f = -2.671766796068, ‖∇f‖ = 2.8048e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  883.35 s: f = -2.671899292949, ‖∇f‖ = 3.5366e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  891.73 s: f = -2.672194986508, ‖∇f‖ = 4.5375e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  900.39 s: f = -2.672493817829, ‖∇f‖ = 5.5658e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  908.89 s: f = -2.672753505736, ‖∇f‖ = 2.0687e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  922.83 s: f = -2.672850238780, ‖∇f‖ = 2.0479e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  931.21 s: f = -2.672980643257, ‖∇f‖ = 2.3947e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  939.63 s: f = -2.673286900540, ‖∇f‖ = 3.0422e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  956.20 s: f = -2.673398343775, ‖∇f‖ = 3.5067e-02, α = 3.21e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   46, time  964.48 s: f = -2.673579922383, ‖∇f‖ = 2.0123e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  972.18 s: f = -2.673698834773, ‖∇f‖ = 1.8405e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  979.81 s: f = -2.673818459445, ‖∇f‖ = 2.4347e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  987.48 s: f = -2.673933525165, ‖∇f‖ = 1.8911e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  995.50 s: f = -2.674089997515, ‖∇f‖ = 2.3738e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1010.77 s: f = -2.674158043027, ‖∇f‖ = 2.8417e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1018.66 s: f = -2.674219511522, ‖∇f‖ = 1.3645e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1032.56 s: f = -2.674261951986, ‖∇f‖ = 1.2275e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1040.39 s: f = -2.674326093056, ‖∇f‖ = 1.7139e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1048.26 s: f = -2.674419581548, ‖∇f‖ = 2.5006e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1056.18 s: f = -2.674486565423, ‖∇f‖ = 2.2734e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1064.48 s: f = -2.674541533348, ‖∇f‖ = 1.1543e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1072.30 s: f = -2.674588014769, ‖∇f‖ = 1.2125e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1086.30 s: f = -2.674627004117, ‖∇f‖ = 1.5452e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1094.42 s: f = -2.674692190363, ‖∇f‖ = 2.3761e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1108.21 s: f = -2.674747139489, ‖∇f‖ = 1.3175e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1116.11 s: f = -2.674775151717, ‖∇f‖ = 1.0912e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1130.04 s: f = -2.674811464235, ‖∇f‖ = 1.2577e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1138.01 s: f = -2.674856898301, ‖∇f‖ = 1.9201e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1146.50 s: f = -2.674909465713, ‖∇f‖ = 1.8198e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1154.50 s: f = -2.674952274100, ‖∇f‖ = 1.0659e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1162.87 s: f = -2.674984765700, ‖∇f‖ = 9.6416e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1170.77 s: f = -2.675005125019, ‖∇f‖ = 1.2764e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1179.63 s: f = -2.675038973771, ‖∇f‖ = 1.6162e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1194.16 s: f = -2.675081913358, ‖∇f‖ = 1.6581e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1223.54 s: f = -2.675119056919, ‖∇f‖ = 1.9355e-02, α = 5.42e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   72, time 1237.63 s: f = -2.675164711729, ‖∇f‖ = 8.9296e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1252.02 s: f = -2.675185025614, ‖∇f‖ = 9.8394e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1266.62 s: f = -2.675208703900, ‖∇f‖ = 8.9414e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1281.08 s: f = -2.675245782533, ‖∇f‖ = 1.2526e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1295.89 s: f = -2.675262164632, ‖∇f‖ = 1.7156e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1310.31 s: f = -2.675283818451, ‖∇f‖ = 7.9273e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1324.56 s: f = -2.675298865471, ‖∇f‖ = 6.1320e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1339.42 s: f = -2.675312616160, ‖∇f‖ = 6.9124e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1354.70 s: f = -2.675331281367, ‖∇f‖ = 1.0125e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1369.67 s: f = -2.675340850961, ‖∇f‖ = 1.7155e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1384.07 s: f = -2.675364134565, ‖∇f‖ = 6.2780e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 1398.77 s: f = -2.675373081794, ‖∇f‖ = 5.3437e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1412.79 s: f = -2.675385605957, ‖∇f‖ = 7.6234e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 1427.99 s: f = -2.675398713824, ‖∇f‖ = 1.1289e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E / prod(size(lattice)) = -0.66884967845596

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian
as in the [Heisenberg example](@ref examples_heisenberg). In that example, with a
non-symmetric $D=2$ PEPS simulation, we reached a ground-state energy per site of around
$E_\text{D=2} = -0.6625\dots$. Again comparing against [Sandvik's](@cite
sandvik_computational_2011) accurate QMC estimate ``E_{\text{ref}}=−0.6694421``, we see that
we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

