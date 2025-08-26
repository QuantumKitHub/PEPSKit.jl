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
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, trscheme = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, alg = :eigsolver, maxiter = 10, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, maxiter = 85, ls_maxiter = 3, ls_maxfg = 3)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -2.356413456811e+03 +3.307968169629e+02im	err = 1.0000e+00
[ Info: CTMRG conv 30:	obj = +6.245129734283e+03 -4.009098120150e-08im	err = 5.3638613840e-09	time = 7.27 sec

````

Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian. Note that the
optimization might take a while since precompilation of symmetric AD code takes longer and
because symmetric tensors do create a bit of overhead (which does pay off at larger bond
and environment dimensions):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity = 3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = -0.034628402377, ‖∇f‖ = 3.0490e-01
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -6.00e-03, ϕ - ϕ₀ = -1.13e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  889.03 s: f = -0.147314472246, ‖∇f‖ = 9.2219e-01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.99e-03, ϕ - ϕ₀ = -3.80e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  995.15 s: f = -0.527654857567, ‖∇f‖ = 7.5002e-01, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time 1022.24 s: f = -0.552751928887, ‖∇f‖ = 3.6310e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time 1100.86 s: f = -0.617206129929, ‖∇f‖ = 3.0891e-01, α = 3.20e+00, m = 2, nfg = 3
[ Info: LBFGS: iter    5, time 1126.38 s: f = -0.632819487273, ‖∇f‖ = 4.0073e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time 1148.89 s: f = -0.653138513637, ‖∇f‖ = 1.0717e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time 1164.51 s: f = -0.655569926077, ‖∇f‖ = 4.7861e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time 1180.01 s: f = -0.656621018321, ‖∇f‖ = 4.3322e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time 1193.45 s: f = -0.657996751278, ‖∇f‖ = 4.6277e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time 1207.53 s: f = -0.659837382098, ‖∇f‖ = 4.8930e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time 1219.82 s: f = -0.661058228215, ‖∇f‖ = 4.1864e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, time 1233.30 s: f = -0.661633890101, ‖∇f‖ = 1.7067e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, time 1245.63 s: f = -0.661839831812, ‖∇f‖ = 1.4709e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time 1258.34 s: f = -0.662134636143, ‖∇f‖ = 1.7065e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time 1269.63 s: f = -0.662532716659, ‖∇f‖ = 1.7719e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time 1282.52 s: f = -0.662996909612, ‖∇f‖ = 2.0543e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time 1293.42 s: f = -0.663439633406, ‖∇f‖ = 2.6049e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time 1305.81 s: f = -0.663855589525, ‖∇f‖ = 3.2567e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time 1317.68 s: f = -0.664450880841, ‖∇f‖ = 2.3691e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time 1330.61 s: f = -0.664917088518, ‖∇f‖ = 1.8281e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time 1341.93 s: f = -0.665079077676, ‖∇f‖ = 1.2380e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time 1354.60 s: f = -0.665287463007, ‖∇f‖ = 1.7237e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time 1365.42 s: f = -0.665717355832, ‖∇f‖ = 2.9356e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time 1378.72 s: f = -0.666166129442, ‖∇f‖ = 3.2946e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time 1404.97 s: f = -0.666407108755, ‖∇f‖ = 2.6211e-02, α = 4.29e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   26, time 1416.89 s: f = -0.666612715340, ‖∇f‖ = 1.1288e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time 1431.81 s: f = -0.666709056953, ‖∇f‖ = 1.2143e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time 1443.40 s: f = -0.666869683924, ‖∇f‖ = 1.8077e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time 1457.84 s: f = -0.667130274911, ‖∇f‖ = 2.1430e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1485.22 s: f = -0.667250996806, ‖∇f‖ = 2.4838e-02, α = 4.17e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   31, time 1497.10 s: f = -0.667422880816, ‖∇f‖ = 1.0123e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1511.10 s: f = -0.667499816879, ‖∇f‖ = 7.8017e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1522.96 s: f = -0.667576990274, ‖∇f‖ = 1.2346e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1536.10 s: f = -0.667710803542, ‖∇f‖ = 1.5052e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1560.08 s: f = -0.667785707694, ‖∇f‖ = 1.7863e-02, α = 5.54e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   36, time 1571.82 s: f = -0.667885903685, ‖∇f‖ = 9.0771e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1585.43 s: f = -0.667941746884, ‖∇f‖ = 7.0177e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1596.86 s: f = -0.667974884585, ‖∇f‖ = 8.8422e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1609.77 s: f = -0.668048817152, ‖∇f‖ = 1.1324e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1623.19 s: f = -0.668123278247, ‖∇f‖ = 1.3967e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1636.73 s: f = -0.668188420240, ‖∇f‖ = 5.1732e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1648.09 s: f = -0.668212525320, ‖∇f‖ = 5.1150e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1661.89 s: f = -0.668245117146, ‖∇f‖ = 5.9778e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1673.68 s: f = -0.668321415567, ‖∇f‖ = 7.7043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1698.15 s: f = -0.668349419497, ‖∇f‖ = 8.7093e-03, α = 3.23e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   46, time 1711.17 s: f = -0.668394232921, ‖∇f‖ = 4.9490e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1723.67 s: f = -0.668423902730, ‖∇f‖ = 4.4315e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1736.82 s: f = -0.668453519415, ‖∇f‖ = 6.6590e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1749.64 s: f = -0.668482512628, ‖∇f‖ = 4.7810e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1763.19 s: f = -0.668522432274, ‖∇f‖ = 4.7205e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1783.90 s: f = -0.668536269465, ‖∇f‖ = 8.2957e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1796.84 s: f = -0.668554701366, ‖∇f‖ = 3.4938e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1808.20 s: f = -0.668564624859, ‖∇f‖ = 3.0996e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1821.15 s: f = -0.668580402014, ‖∇f‖ = 4.3886e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1833.30 s: f = -0.668603907002, ‖∇f‖ = 5.3988e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time 1857.93 s: f = -0.668617246171, ‖∇f‖ = 5.8694e-03, α = 4.99e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   57, time 1871.12 s: f = -0.668632724021, ‖∇f‖ = 3.2358e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1883.00 s: f = -0.668645126494, ‖∇f‖ = 3.1763e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1895.59 s: f = -0.668655398082, ‖∇f‖ = 4.2974e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time 1908.32 s: f = -0.668669899756, ‖∇f‖ = 4.4860e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1921.43 s: f = -0.668685368560, ‖∇f‖ = 3.1222e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1933.39 s: f = -0.668694965358, ‖∇f‖ = 4.0039e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1947.37 s: f = -0.668703207325, ‖∇f‖ = 3.3668e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1959.17 s: f = -0.668712392399, ‖∇f‖ = 3.4980e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1972.01 s: f = -0.668728646296, ‖∇f‖ = 6.7104e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1984.55 s: f = -0.668739676349, ‖∇f‖ = 3.7164e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1997.58 s: f = -0.668745430605, ‖∇f‖ = 2.1495e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 2010.22 s: f = -0.668751516193, ‖∇f‖ = 2.0809e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 2022.61 s: f = -0.668760790859, ‖∇f‖ = 2.5898e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 2034.67 s: f = -0.668768119558, ‖∇f‖ = 7.2678e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 2048.26 s: f = -0.668784264610, ‖∇f‖ = 2.6530e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 2061.44 s: f = -0.668789170500, ‖∇f‖ = 1.8137e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 2074.23 s: f = -0.668795881528, ‖∇f‖ = 2.0901e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 2087.76 s: f = -0.668799083020, ‖∇f‖ = 6.5951e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 2101.93 s: f = -0.668808063685, ‖∇f‖ = 2.8612e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 2114.35 s: f = -0.668814365126, ‖∇f‖ = 1.5759e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 2127.06 s: f = -0.668817947423, ‖∇f‖ = 1.9177e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 2141.19 s: f = -0.668824279559, ‖∇f‖ = 2.2614e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 2153.98 s: f = -0.668828130138, ‖∇f‖ = 4.6422e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 2166.74 s: f = -0.668835067927, ‖∇f‖ = 1.5939e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 2179.82 s: f = -0.668837219514, ‖∇f‖ = 2.3215e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 2192.00 s: f = -0.668837920006, ‖∇f‖ = 2.8050e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 2205.41 s: f = -0.668842793389, ‖∇f‖ = 1.8653e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 2218.47 s: f = -0.668846752110, ‖∇f‖ = 1.3953e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 2231.79 s: f = -0.668849702901, ‖∇f‖ = 2.0146e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.6688497029008129

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian as
in the [Heisenberg example](@ref examples_heisenberg). In that example, with a non-symmetric
$D=2$ PEPS simulation, we reached a ground-state energy of around $E_\text{D=2} = -0.6625\dots$.
Again comparing against [Sandvik's](@cite sandvik_computational_2011) accurate QMC estimate
``E_{\text{ref}}=−0.6694421``, we see that we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

