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
spin = 1//2
symmetry = U1Irrep
lattice = InfiniteSquare(2, 2)
H₀ = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin);
````

This ensures that our PEPS ansatz can support the bipartite Néel order. As discussed above,
we encode the Néel order directly in the ansatz by adding staggered auxiliary physical
charges:

````julia
S_aux = [
    U1Irrep(-1//2) U1Irrep(1//2)
    U1Irrep(1//2) U1Irrep(-1//2)
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
boundary_alg = (; tol=1e-8, alg=:simultaneous, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, maxiter=85, ls_maxiter=3, ls_maxfg=3)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -2.356413456811e+03 +3.307968169629e+02im	err = 1.0000e+00
[ Info: CTMRG conv 30:	obj = +6.245129734283e+03 -4.010098564322e-08im	err = 5.3638617277e-09	time = 0.92 sec

````

Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian. Note that the
optimization might take a while since precompilation of symmetric AD code takes longer and
because symmetric tensors do create a bit of overhead (which does pay off at larger bond
and environment dimensions):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = -0.034628402377, ‖∇f‖ = 3.0490e-01
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -6.00e-03, ϕ - ϕ₀ = -1.13e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  200.41 s: f = -0.147314472246, ‖∇f‖ = 9.2219e-01, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.99e-03, ϕ - ϕ₀ = -3.80e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  269.67 s: f = -0.527654857567, ‖∇f‖ = 7.5002e-01, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  286.74 s: f = -0.552751928887, ‖∇f‖ = 3.6310e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  337.91 s: f = -0.617206129929, ‖∇f‖ = 3.0891e-01, α = 3.20e+00, m = 2, nfg = 3
[ Info: LBFGS: iter    5, time  354.97 s: f = -0.632819487274, ‖∇f‖ = 4.0073e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  370.30 s: f = -0.653138513637, ‖∇f‖ = 1.0717e-01, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  381.55 s: f = -0.655569926077, ‖∇f‖ = 4.7861e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  392.14 s: f = -0.656621018321, ‖∇f‖ = 4.3322e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  402.41 s: f = -0.657996751278, ‖∇f‖ = 4.6277e-02, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  412.44 s: f = -0.659837382098, ‖∇f‖ = 4.8930e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  421.91 s: f = -0.661058228215, ‖∇f‖ = 4.1864e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, time  431.77 s: f = -0.661633890101, ‖∇f‖ = 1.7067e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, time  440.37 s: f = -0.661839831812, ‖∇f‖ = 1.4709e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  449.27 s: f = -0.662134636143, ‖∇f‖ = 1.7065e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  458.85 s: f = -0.662532716659, ‖∇f‖ = 1.7719e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  467.98 s: f = -0.662996909612, ‖∇f‖ = 2.0543e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time  477.51 s: f = -0.663439633406, ‖∇f‖ = 2.6049e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time  486.45 s: f = -0.663855589525, ‖∇f‖ = 3.2567e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  495.46 s: f = -0.664450880841, ‖∇f‖ = 2.3691e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time  505.64 s: f = -0.664917088518, ‖∇f‖ = 1.8281e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time  514.58 s: f = -0.665079077676, ‖∇f‖ = 1.2380e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time  523.59 s: f = -0.665287463007, ‖∇f‖ = 1.7237e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  533.23 s: f = -0.665717355832, ‖∇f‖ = 2.9356e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time  543.00 s: f = -0.666166129443, ‖∇f‖ = 3.2946e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  562.76 s: f = -0.666407108756, ‖∇f‖ = 2.6211e-02, α = 4.29e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   26, time  573.09 s: f = -0.666612715340, ‖∇f‖ = 1.1288e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  582.76 s: f = -0.666709056953, ‖∇f‖ = 1.2143e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  592.05 s: f = -0.666869683924, ‖∇f‖ = 1.8077e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  602.27 s: f = -0.667130274911, ‖∇f‖ = 2.1430e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  622.23 s: f = -0.667250996806, ‖∇f‖ = 2.4838e-02, α = 4.17e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   31, time  631.86 s: f = -0.667422880816, ‖∇f‖ = 1.0123e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  641.47 s: f = -0.667499816879, ‖∇f‖ = 7.8017e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  651.74 s: f = -0.667576990274, ‖∇f‖ = 1.2346e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  661.42 s: f = -0.667710803542, ‖∇f‖ = 1.5052e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  680.14 s: f = -0.667785707694, ‖∇f‖ = 1.7863e-02, α = 5.54e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   36, time  689.37 s: f = -0.667885903685, ‖∇f‖ = 9.0771e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  698.32 s: f = -0.667941746884, ‖∇f‖ = 7.0177e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  708.15 s: f = -0.667974884585, ‖∇f‖ = 8.8422e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  716.81 s: f = -0.668048817152, ‖∇f‖ = 1.1324e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  725.46 s: f = -0.668123278247, ‖∇f‖ = 1.3967e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  735.36 s: f = -0.668188420240, ‖∇f‖ = 5.1732e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  744.40 s: f = -0.668212525320, ‖∇f‖ = 5.1150e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  754.13 s: f = -0.668245117146, ‖∇f‖ = 5.9778e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  763.24 s: f = -0.668321415567, ‖∇f‖ = 7.7043e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  781.85 s: f = -0.668349419497, ‖∇f‖ = 8.7093e-03, α = 3.23e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   46, time  790.85 s: f = -0.668394232920, ‖∇f‖ = 4.9490e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  799.85 s: f = -0.668423902729, ‖∇f‖ = 4.4315e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  809.56 s: f = -0.668453519413, ‖∇f‖ = 6.6590e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time  818.73 s: f = -0.668482512627, ‖∇f‖ = 4.7810e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  827.88 s: f = -0.668522432273, ‖∇f‖ = 4.7205e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  842.27 s: f = -0.668536269464, ‖∇f‖ = 8.2957e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  851.35 s: f = -0.668554701367, ‖∇f‖ = 3.4938e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  860.86 s: f = -0.668564624860, ‖∇f‖ = 3.0996e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  870.08 s: f = -0.668580402016, ‖∇f‖ = 4.3886e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  879.06 s: f = -0.668603907003, ‖∇f‖ = 5.3988e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  897.98 s: f = -0.668617246170, ‖∇f‖ = 5.8694e-03, α = 4.99e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   57, time  907.72 s: f = -0.668632724020, ‖∇f‖ = 3.2358e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  917.01 s: f = -0.668645126493, ‖∇f‖ = 3.1763e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  926.33 s: f = -0.668655398082, ‖∇f‖ = 4.2974e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  936.00 s: f = -0.668669899756, ‖∇f‖ = 4.4860e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  945.03 s: f = -0.668685368558, ‖∇f‖ = 3.1222e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  954.24 s: f = -0.668694965356, ‖∇f‖ = 4.0039e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  963.47 s: f = -0.668703207321, ‖∇f‖ = 3.3668e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  972.78 s: f = -0.668712392397, ‖∇f‖ = 3.4980e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  981.96 s: f = -0.668728646271, ‖∇f‖ = 6.7104e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  991.59 s: f = -0.668739676361, ‖∇f‖ = 3.7165e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1000.90 s: f = -0.668745430589, ‖∇f‖ = 2.1495e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1010.57 s: f = -0.668751516160, ‖∇f‖ = 2.0809e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1019.38 s: f = -0.668760790802, ‖∇f‖ = 2.5898e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1028.31 s: f = -0.668768119147, ‖∇f‖ = 7.2679e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1038.11 s: f = -0.668784264566, ‖∇f‖ = 2.6529e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1047.35 s: f = -0.668789170375, ‖∇f‖ = 1.8137e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1056.72 s: f = -0.668795881546, ‖∇f‖ = 2.0901e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1066.35 s: f = -0.668799081816, ‖∇f‖ = 6.5956e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1075.63 s: f = -0.668808063181, ‖∇f‖ = 2.8612e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1085.30 s: f = -0.668814364081, ‖∇f‖ = 1.5758e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1094.40 s: f = -0.668817946829, ‖∇f‖ = 1.9178e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1103.72 s: f = -0.668824278385, ‖∇f‖ = 2.2613e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1113.67 s: f = -0.668828126813, ‖∇f‖ = 4.6442e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1123.07 s: f = -0.668835065826, ‖∇f‖ = 1.5939e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1132.41 s: f = -0.668837215645, ‖∇f‖ = 2.3238e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1142.13 s: f = -0.668837926114, ‖∇f‖ = 2.8015e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 1151.39 s: f = -0.668842788932, ‖∇f‖ = 1.8642e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1160.72 s: f = -0.668846748527, ‖∇f‖ = 1.3959e-03, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 1170.53 s: f = -0.668849701358, ‖∇f‖ = 2.0170e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.6688497013575809

````

Note that for the specified parameters $J = \Delta = 1$, we simulated the same Hamiltonian as
in the [Heisenberg example](@ref examples_heisenberg). In that example, with a non-symmetric
$D=2$ PEPS simulation, we reached a ground-state energy of around $E_\text{D=2} = -0.6625\dots$.
Again comparing against [Sandvik's](@cite sandvik_computational_2011) accurate QMC estimate
``E_{\text{ref}}=−0.6694421``, we see that we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

