```@meta
EditURL = "../../../../examples/4.xxz/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//4.xxz/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//4.xxz/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//4.xxz)


# Néel order in the $U(1)$-symmetric XXZ model

Here, we want to look at a special case of the Heisenberg model, where the $x$ and $y$
couplings are equal, called the XXZ model

```math
H_0 = J \big(\sum_{\langle i, j \rangle} S_i^x S_j^x + S_i^y S_j^y + \Delta S_i^z S_j^z \big) .
```

For appropriate $\Delta$, the model enters an antiferromagnetic phase (Néel order) which we
will force by adding staggered magnetic charges to ``H_0``. Furthermore, since the XXZ
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
physical_spaces = H.lattice
````

````
2×2 Matrix{GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}}:
 Rep[U₁](0=>1, -1=>1)  Rep[U₁](0=>1, 1=>1)
 Rep[U₁](0=>1, 1=>1)   Rep[U₁](0=>1, -1=>1)
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
[ Info: CTMRG init:	obj = -1.121020187593e+04 -6.991066478500e+03im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +6.369731502336e+03 -8.500319381710e-08im	err = 7.5599921139e-09	time = 2.09 sec

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
[ Info: LBFGS: initializing with f = -0.033045967451, ‖∇f‖ = 3.2973e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -9.83e-03, ϕ - ϕ₀ = -1.52e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time   81.96 s: f = -0.185441286972, ‖∇f‖ = 1.8487e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.83e-03, ϕ - ϕ₀ = -3.94e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  139.01 s: f = -0.579296056587, ‖∇f‖ = 5.7534e-01, α = 2.50e+01, m = 0, nfg = 4
[ Info: LBFGS: iter    3, time  147.91 s: f = -0.613492668655, ‖∇f‖ = 3.3905e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: LBFGS: iter    4, time  155.93 s: f = -0.638698278101, ‖∇f‖ = 2.2127e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    5, time  164.77 s: f = -0.650278206280, ‖∇f‖ = 1.9661e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: LBFGS: iter    6, time  173.00 s: f = -0.654875781442, ‖∇f‖ = 7.1166e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: LBFGS: iter    7, time  181.88 s: f = -0.656067729644, ‖∇f‖ = 5.1946e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: LBFGS: iter    8, time  190.01 s: f = -0.659056547972, ‖∇f‖ = 5.4114e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: LBFGS: iter    9, time  198.16 s: f = -0.660492187403, ‖∇f‖ = 1.0001e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter   10, time  205.43 s: f = -0.662131899972, ‖∇f‖ = 3.0720e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   11, time  213.51 s: f = -0.662491636970, ‖∇f‖ = 2.1327e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   12, time  220.80 s: f = -0.662815874554, ‖∇f‖ = 2.0939e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   13, time  228.87 s: f = -0.663206295533, ‖∇f‖ = 2.3137e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   14, time  236.16 s: f = -0.663617412480, ‖∇f‖ = 3.2148e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   15, time  244.23 s: f = -0.663981399612, ‖∇f‖ = 2.1162e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   16, time  251.62 s: f = -0.664500170184, ‖∇f‖ = 3.0047e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   17, time  259.80 s: f = -0.665018204774, ‖∇f‖ = 3.4052e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   18, time  267.91 s: f = -0.665352132477, ‖∇f‖ = 4.0506e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   19, time  276.74 s: f = -0.665708949595, ‖∇f‖ = 1.8115e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   20, time  284.74 s: f = -0.665851319983, ‖∇f‖ = 1.7887e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   21, time  293.66 s: f = -0.666077676932, ‖∇f‖ = 2.1504e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   22, time  301.95 s: f = -0.666415750625, ‖∇f‖ = 2.1742e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time  319.57 s: f = -0.666535417444, ‖∇f‖ = 2.1053e-02, α = 3.37e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   24, time  328.83 s: f = -0.666677914685, ‖∇f‖ = 1.4107e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time  337.42 s: f = -0.666880717597, ‖∇f‖ = 1.5936e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time  346.84 s: f = -0.667020492218, ‖∇f‖ = 2.1705e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time  355.47 s: f = -0.667174861341, ‖∇f‖ = 1.3538e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time  364.80 s: f = -0.667242799138, ‖∇f‖ = 1.3749e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time  373.39 s: f = -0.667289201830, ‖∇f‖ = 9.6270e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time  382.70 s: f = -0.667382804207, ‖∇f‖ = 1.1004e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time  391.35 s: f = -0.667514031651, ‖∇f‖ = 1.5020e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time  400.73 s: f = -0.667654012398, ‖∇f‖ = 1.6237e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time  409.44 s: f = -0.667695159156, ‖∇f‖ = 2.0278e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time  418.69 s: f = -0.667791548479, ‖∇f‖ = 7.2359e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time  427.10 s: f = -0.667831622740, ‖∇f‖ = 7.0221e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   36, time  436.35 s: f = -0.667897197556, ‖∇f‖ = 1.0611e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time  444.91 s: f = -0.667974329902, ‖∇f‖ = 1.3520e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time  454.41 s: f = -0.668044978527, ‖∇f‖ = 8.1835e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time  463.08 s: f = -0.668096641090, ‖∇f‖ = 5.8523e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time  474.36 s: f = -0.668140633648, ‖∇f‖ = 8.7357e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time  485.42 s: f = -0.668191141610, ‖∇f‖ = 1.0519e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time  497.24 s: f = -0.668251028839, ‖∇f‖ = 9.9323e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time  508.60 s: f = -0.668287406221, ‖∇f‖ = 8.1822e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time  520.51 s: f = -0.668312220264, ‖∇f‖ = 4.9144e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time  531.38 s: f = -0.668335763877, ‖∇f‖ = 5.4414e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time  540.97 s: f = -0.668372098383, ‖∇f‖ = 6.8595e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time  549.63 s: f = -0.668431605944, ‖∇f‖ = 8.8807e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time  567.86 s: f = -0.668464682116, ‖∇f‖ = 8.2571e-03, α = 5.24e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   49, time  577.42 s: f = -0.668492867254, ‖∇f‖ = 4.3005e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time  586.10 s: f = -0.668513727270, ‖∇f‖ = 3.9910e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time  595.51 s: f = -0.668532155485, ‖∇f‖ = 4.9090e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time  604.13 s: f = -0.668564505110, ‖∇f‖ = 7.4817e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time  613.65 s: f = -0.668593093779, ‖∇f‖ = 5.7300e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time  622.26 s: f = -0.668613013637, ‖∇f‖ = 4.0964e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time  631.67 s: f = -0.668634613224, ‖∇f‖ = 4.2489e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   56, time  640.31 s: f = -0.668649481004, ‖∇f‖ = 4.5912e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time  650.03 s: f = -0.668670198485, ‖∇f‖ = 3.8195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time  658.82 s: f = -0.668688448995, ‖∇f‖ = 3.4508e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time  668.38 s: f = -0.668696405392, ‖∇f‖ = 5.8372e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   60, time  677.09 s: f = -0.668706977023, ‖∇f‖ = 3.5019e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time  686.54 s: f = -0.668721369553, ‖∇f‖ = 2.8608e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time  695.13 s: f = -0.668732770821, ‖∇f‖ = 3.0068e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time  704.62 s: f = -0.668738498959, ‖∇f‖ = 7.3212e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time  713.61 s: f = -0.668752996535, ‖∇f‖ = 3.0434e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time  723.01 s: f = -0.668762694601, ‖∇f‖ = 1.9787e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time  731.88 s: f = -0.668773371271, ‖∇f‖ = 2.8034e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time  741.54 s: f = -0.668783730169, ‖∇f‖ = 4.4007e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time  750.34 s: f = -0.668794471804, ‖∇f‖ = 2.4210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time  759.94 s: f = -0.668800925048, ‖∇f‖ = 1.7114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time  768.74 s: f = -0.668808484765, ‖∇f‖ = 2.5442e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time  778.43 s: f = -0.668813882136, ‖∇f‖ = 3.7919e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time  787.17 s: f = -0.668819791141, ‖∇f‖ = 2.1652e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time  796.73 s: f = -0.668826145765, ‖∇f‖ = 1.9093e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time  805.39 s: f = -0.668830491901, ‖∇f‖ = 2.4506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time  814.87 s: f = -0.668836964982, ‖∇f‖ = 2.6465e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time  833.41 s: f = -0.668840404782, ‖∇f‖ = 3.1398e-03, α = 3.87e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   77, time  842.18 s: f = -0.668843872314, ‖∇f‖ = 3.1246e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time  851.69 s: f = -0.668847211574, ‖∇f‖ = 2.4091e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time  860.43 s: f = -0.668850327797, ‖∇f‖ = 2.0185e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time  870.17 s: f = -0.668852132421, ‖∇f‖ = 4.1279e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time  897.81 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 4.20e-02, m = 20, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   82, time  926.31 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   83, time  954.24 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   84, time  983.38 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
┌ Warning: LBFGS: not converged to requested tol after 85 iterations and time 1011.99 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.6688553292829834

````

Note that for the specified parameters $J=\Delta=1$, we simulated the same Hamiltonian as
in the [Heisenberg example](@ref examples_heisenberg). In that example, with a non-symmetric
$D=2$ PEPS simulation, we reached a ground-state energy of around $E_\text{D=2} = -0.6625\dots$.
Again comparing against [Sandvik's](@cite sandvik_computational_2011) accurate QMC estimate
``E_{\text{ref}}=−0.6694421``, we see that we already got closer to the reference energy.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

