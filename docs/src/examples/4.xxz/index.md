```@meta
EditURL = "../../../../examples/4.xxz/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//4.xxz/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//4.xxz/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//4.xxz)

````julia
using Markdown
````

# Néel order in the U(1)-symmetric XXZ model

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
2×2 Matrix{TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}}:
 Rep[TensorKitSectors.U₁](0=>1, -1=>1)  Rep[TensorKitSectors.U₁](0=>1, 1=>1)
 Rep[TensorKitSectors.U₁](0=>1, 1=>1)   Rep[TensorKitSectors.U₁](0=>1, -1=>1)
````

## Ground state search

From this point onwards it's business as usual: Create an initial PEPS and environment
(using the symmetric spaces), specify the algorithmic parameters and optimize:

````julia
boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=100, ls_maxiter=2, ls_maxfg=2)

peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = -1.121020187593e+04 -6.991066478500e+03im	err = 1.0000e+00
[ Info: CTMRG conv 26:	obj = +6.369731502336e+03 -8.500319381710e-08im	err = 7.5599921139e-09	time = 0.97 sec

````

Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian. Note that the
optimization might take a while since precompilation of symmetric AD code takes longer and
because symmetric tensors do create a bit of overhead (which does pay off at larger bond
and environment dimensions):

````julia
peps, env, E, info = fixedpoint(H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg)
@show E;
````

````
[ Info: CTMRG init:	obj = +2.434421528081e-05	err = 1.0000e+00
[ Info: CTMRG conv 4:	obj = +2.434421528088e-05	err = 4.5912943585e-10	time = 0.18 sec
[ Info: LBFGS: initializing with f = -0.033045967451, ‖∇f‖ = 3.2973e-01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: CTMRG init:	obj = +2.453223895159e-05	err = 1.0000e+00
[ Info: CTMRG conv 19:	obj = +2.453720354328e-05	err = 5.3853880385e-09	time = 0.70 sec
[ Info: CTMRG init:	obj = +2.536991372204e-05	err = 1.0000e+00
[ Info: CTMRG conv 21:	obj = +2.551399135525e-05	err = 5.7602811253e-09	time = 1.01 sec
[ Info: CTMRG init:	obj = +3.190392580620e-05	err = 1.0000e+00
[ Info: CTMRG conv 23:	obj = +3.871282989002e-05	err = 2.3099743648e-09	time = 0.80 sec
[ Info: CTMRG init:	obj = +1.997307247185e-04 +1.744054390924e-14im	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +2.397713225168e-03	err = 8.6119116463e-09	time = 0.59 sec
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -9.83e-03, ϕ - ϕ₀ = -1.52e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    1, time  173.47 s: f = -0.185441286972, ‖∇f‖ = 1.8487e+00, α = 2.50e+01, m = 0, nfg = 4
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: CTMRG init:	obj = +4.060460706144e-05	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +4.073280032714e-05	err = 8.7858583593e-09	time = 0.72 sec
[ Info: CTMRG init:	obj = +4.950177698074e-05	err = 1.0000e+00
[ Info: CTMRG conv 24:	obj = +5.405071831250e-05	err = 4.1362491224e-09	time = 0.87 sec
[ Info: CTMRG init:	obj = +1.459089235128e-04	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +3.628610875102e-04	err = 7.1545714198e-09	time = 0.50 sec
[ Info: CTMRG init:	obj = +1.067948187035e-02	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +3.300787404705e-02	err = 2.7551053256e-09	time = 0.38 sec
┌ Warning: Linesearch not converged after 1 iterations and 4 function evaluations:
│ α = 2.50e+01, dϕ = -1.83e-03, ϕ - ϕ₀ = -3.94e-01
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    2, time  219.43 s: f = -0.579296056587, ‖∇f‖ = 5.7534e-01, α = 2.50e+01, m = 0, nfg = 4
[ Info: CTMRG init:	obj = +5.216577372581e-04	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +5.262193467872e-04	err = 2.3063817877e-09	time = 0.43 sec
[ Info: LBFGS: iter    3, time  225.57 s: f = -0.613492668655, ‖∇f‖ = 3.3905e-01, α = 1.00e+00, m = 1, nfg = 1
[ Info: CTMRG init:	obj = +5.712313609937e-04	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +5.762503340314e-04	err = 1.2560009849e-09	time = 0.44 sec
[ Info: LBFGS: iter    4, time  231.66 s: f = -0.638698278101, ‖∇f‖ = 2.2127e-01, α = 1.00e+00, m = 2, nfg = 1
[ Info: CTMRG init:	obj = +6.503993965962e-04	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +6.672622988324e-04	err = 1.2420966253e-09	time = 0.42 sec
[ Info: LBFGS: iter    5, time  237.33 s: f = -0.650278206280, ‖∇f‖ = 1.9661e-01, α = 1.00e+00, m = 3, nfg = 1
[ Info: CTMRG init:	obj = +7.095032316279e-04	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +7.107783954017e-04	err = 6.2073791690e-09	time = 0.40 sec
[ Info: LBFGS: iter    6, time  243.08 s: f = -0.654875781442, ‖∇f‖ = 7.1166e-02, α = 1.00e+00, m = 4, nfg = 1
[ Info: CTMRG init:	obj = +7.678769240767e-04	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +7.677598689487e-04	err = 2.8083589443e-09	time = 0.42 sec
[ Info: LBFGS: iter    7, time  248.84 s: f = -0.656067729644, ‖∇f‖ = 5.1946e-02, α = 1.00e+00, m = 5, nfg = 1
[ Info: CTMRG init:	obj = +1.047392070849e-03	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +1.045750152055e-03	err = 3.4449671849e-09	time = 0.42 sec
[ Info: LBFGS: iter    8, time  254.59 s: f = -0.659056547972, ‖∇f‖ = 5.4114e-02, α = 1.00e+00, m = 6, nfg = 1
[ Info: CTMRG init:	obj = +1.532812857985e-03	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +1.525097438199e-03	err = 2.4326768610e-09	time = 0.38 sec
[ Info: LBFGS: iter    9, time  259.75 s: f = -0.660492187403, ‖∇f‖ = 1.0001e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: CTMRG init:	obj = +1.795787577321e-03	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +1.794727275308e-03	err = 2.9197329251e-09	time = 0.36 sec
[ Info: LBFGS: iter   10, time  265.12 s: f = -0.662131899972, ‖∇f‖ = 3.0720e-02, α = 1.00e+00, m = 8, nfg = 1
[ Info: CTMRG init:	obj = +1.996305356333e-03	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +1.995481173538e-03	err = 1.5914652201e-09	time = 0.37 sec
[ Info: LBFGS: iter   11, time  270.19 s: f = -0.662491636970, ‖∇f‖ = 2.1327e-02, α = 1.00e+00, m = 9, nfg = 1
[ Info: CTMRG init:	obj = +2.243383437860e-03	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +2.242179147938e-03	err = 1.3840632855e-09	time = 0.37 sec
[ Info: LBFGS: iter   12, time  275.22 s: f = -0.662815874554, ‖∇f‖ = 2.0939e-02, α = 1.00e+00, m = 10, nfg = 1
[ Info: CTMRG init:	obj = +2.529321906303e-03	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +2.527958437132e-03	err = 1.3943320526e-09	time = 0.38 sec
[ Info: LBFGS: iter   13, time  280.65 s: f = -0.663206295533, ‖∇f‖ = 2.3137e-02, α = 1.00e+00, m = 11, nfg = 1
[ Info: CTMRG init:	obj = +2.905364340563e-03	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +2.902988231101e-03	err = 2.0819309387e-09	time = 0.39 sec
[ Info: LBFGS: iter   14, time  286.03 s: f = -0.663617412480, ‖∇f‖ = 3.2148e-02, α = 1.00e+00, m = 12, nfg = 1
[ Info: CTMRG init:	obj = +2.904272464292e-03	err = 1.0000e+00
[ Info: CTMRG conv 10:	obj = +2.905132091807e-03	err = 1.9716972635e-09	time = 0.39 sec
[ Info: LBFGS: iter   15, time  291.13 s: f = -0.663981399612, ‖∇f‖ = 2.1162e-02, α = 1.00e+00, m = 13, nfg = 1
[ Info: CTMRG init:	obj = +2.742432815679e-03	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +2.745820290756e-03	err = 1.5265448213e-09	time = 0.38 sec
[ Info: LBFGS: iter   16, time  296.57 s: f = -0.664500170184, ‖∇f‖ = 3.0047e-02, α = 1.00e+00, m = 14, nfg = 1
[ Info: CTMRG init:	obj = +2.401849189969e-03	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +2.405656778850e-03	err = 2.5894583460e-09	time = 0.46 sec
[ Info: LBFGS: iter   17, time  301.86 s: f = -0.665018204774, ‖∇f‖ = 3.4052e-02, α = 1.00e+00, m = 15, nfg = 1
[ Info: CTMRG init:	obj = +2.122203005349e-03	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +2.121928393701e-03	err = 9.5870670882e-09	time = 0.44 sec
[ Info: LBFGS: iter   18, time  307.57 s: f = -0.665352132477, ‖∇f‖ = 4.0506e-02, α = 1.00e+00, m = 16, nfg = 1
[ Info: CTMRG init:	obj = +2.072289083756e-03	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +2.072594745594e-03	err = 4.5258473845e-09	time = 0.43 sec
[ Info: LBFGS: iter   19, time  313.22 s: f = -0.665708949595, ‖∇f‖ = 1.8115e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: CTMRG init:	obj = +2.162982716617e-03	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +2.163052122686e-03	err = 5.9164202930e-09	time = 0.38 sec
[ Info: LBFGS: iter   20, time  319.05 s: f = -0.665851319983, ‖∇f‖ = 1.7887e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: CTMRG init:	obj = +2.220648043164e-03	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +2.221583997885e-03	err = 5.4700111805e-09	time = 0.43 sec
[ Info: LBFGS: iter   21, time  324.88 s: f = -0.666077676932, ‖∇f‖ = 2.1504e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: CTMRG init:	obj = +2.452606638527e-03	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +2.457433179381e-03	err = 7.8365492645e-09	time = 0.49 sec
[ Info: LBFGS: iter   22, time  330.78 s: f = -0.666415750625, ‖∇f‖ = 2.1742e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.618995877418e-03	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +2.627247025253e-03	err = 5.4628097961e-09	time = 0.43 sec
[ Info: CTMRG init:	obj = +2.517366578177e-03	err = 1.0000e+00
[ Info: CTMRG conv 11:	obj = +2.518273673784e-03	err = 8.0918301167e-09	time = 0.39 sec
[ Info: LBFGS: iter   23, time  342.43 s: f = -0.666535417444, ‖∇f‖ = 2.1053e-02, α = 3.37e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +2.575130612824e-03	err = 1.0000e+00
[ Info: CTMRG conv 12:	obj = +2.575296959997e-03	err = 5.8864311994e-09	time = 0.43 sec
[ Info: LBFGS: iter   24, time  348.22 s: f = -0.666677914685, ‖∇f‖ = 1.4107e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.640725585692e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +2.643147349459e-03	err = 2.1654459528e-09	time = 0.51 sec
[ Info: LBFGS: iter   25, time  354.02 s: f = -0.666880717597, ‖∇f‖ = 1.5936e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.677813236881e-03	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +2.677409654697e-03	err = 7.1910371417e-09	time = 0.48 sec
[ Info: LBFGS: iter   26, time  359.89 s: f = -0.667020492218, ‖∇f‖ = 2.1705e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.714827500016e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +2.715087100998e-03	err = 3.4660002553e-09	time = 0.52 sec
[ Info: LBFGS: iter   27, time  365.73 s: f = -0.667174861341, ‖∇f‖ = 1.3538e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.870936188843e-03	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +2.871152272950e-03	err = 4.4859828582e-09	time = 0.49 sec
[ Info: LBFGS: iter   28, time  371.84 s: f = -0.667242799138, ‖∇f‖ = 1.3749e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.894454117569e-03	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +2.894497874679e-03	err = 2.8815190399e-09	time = 0.51 sec
[ Info: LBFGS: iter   29, time  377.75 s: f = -0.667289201830, ‖∇f‖ = 9.6270e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +2.992675491930e-03	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +2.992923140174e-03	err = 5.1513256196e-09	time = 0.48 sec
[ Info: LBFGS: iter   30, time  383.74 s: f = -0.667382804207, ‖∇f‖ = 1.1004e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.166823421024e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +3.168045231564e-03	err = 1.9426667949e-09	time = 0.52 sec
[ Info: LBFGS: iter   31, time  389.50 s: f = -0.667514031651, ‖∇f‖ = 1.5020e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.391726443493e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +3.393587165917e-03	err = 4.1266089130e-09	time = 0.49 sec
[ Info: LBFGS: iter   32, time  395.39 s: f = -0.667654012398, ‖∇f‖ = 1.6237e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.608251780050e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +3.612137047877e-03	err = 5.5508459619e-09	time = 0.55 sec
[ Info: LBFGS: iter   33, time  401.25 s: f = -0.667695159156, ‖∇f‖ = 2.0278e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.567945490801e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +3.568212239514e-03	err = 4.2558976934e-09	time = 0.51 sec
[ Info: LBFGS: iter   34, time  407.49 s: f = -0.667791548479, ‖∇f‖ = 7.2359e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.560786709647e-03	err = 1.0000e+00
[ Info: CTMRG conv 13:	obj = +3.560744298718e-03	err = 9.7700409998e-09	time = 0.49 sec
[ Info: LBFGS: iter   35, time  413.61 s: f = -0.667831622740, ‖∇f‖ = 7.0221e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.612058119712e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +3.612096455450e-03	err = 9.4956185538e-09	time = 0.50 sec
[ Info: LBFGS: iter   36, time  419.79 s: f = -0.667897197556, ‖∇f‖ = 1.0611e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.748694446268e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +3.749410253030e-03	err = 8.1837906634e-09	time = 0.53 sec
[ Info: LBFGS: iter   37, time  426.04 s: f = -0.667974329902, ‖∇f‖ = 1.3520e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +3.932731964018e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +3.934152642199e-03	err = 4.4759136527e-09	time = 0.58 sec
[ Info: LBFGS: iter   38, time  432.14 s: f = -0.668044978527, ‖∇f‖ = 8.1835e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.075036525220e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.075095185707e-03	err = 4.9388492576e-09	time = 0.54 sec
[ Info: LBFGS: iter   39, time  438.40 s: f = -0.668096641090, ‖∇f‖ = 5.8523e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.249636465901e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +4.250516433969e-03	err = 9.0111357740e-09	time = 0.55 sec
[ Info: LBFGS: iter   40, time  444.50 s: f = -0.668140633648, ‖∇f‖ = 8.7357e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.408266209899e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.408238818952e-03	err = 9.3231498401e-09	time = 0.54 sec
[ Info: LBFGS: iter   41, time  450.71 s: f = -0.668191141610, ‖∇f‖ = 1.0519e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.597894504232e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.598301067090e-03	err = 8.3891599548e-09	time = 0.60 sec
[ Info: LBFGS: iter   42, time  456.88 s: f = -0.668251028839, ‖∇f‖ = 9.9323e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.696678230485e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.697823025374e-03	err = 8.0248890527e-09	time = 0.57 sec
[ Info: LBFGS: iter   43, time  462.72 s: f = -0.668287406221, ‖∇f‖ = 8.1822e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.650711531624e-03	err = 1.0000e+00
[ Info: CTMRG conv 14:	obj = +4.650746519732e-03	err = 6.7135782228e-09	time = 0.47 sec
[ Info: LBFGS: iter   44, time  468.42 s: f = -0.668312220264, ‖∇f‖ = 4.9144e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.625470238204e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.625513939090e-03	err = 4.9755202754e-09	time = 0.54 sec
[ Info: LBFGS: iter   45, time  474.22 s: f = -0.668335763877, ‖∇f‖ = 5.4414e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.669165659738e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.669545010290e-03	err = 2.7865418046e-09	time = 0.56 sec
[ Info: LBFGS: iter   46, time  480.47 s: f = -0.668372098383, ‖∇f‖ = 6.8595e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +4.882117118241e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +4.884335131166e-03	err = 8.3333297223e-09	time = 0.59 sec
[ Info: LBFGS: iter   47, time  486.78 s: f = -0.668431605944, ‖∇f‖ = 8.8807e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.309290623370e-03	err = 1.0000e+00
[ Info: CTMRG conv 17:	obj = +5.316323973446e-03	err = 8.0212919448e-09	time = 0.62 sec
[ Info: CTMRG init:	obj = +5.119887535680e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +5.121775455646e-03	err = 6.2134661508e-09	time = 0.57 sec
[ Info: LBFGS: iter   48, time  498.87 s: f = -0.668464682116, ‖∇f‖ = 8.2571e-03, α = 5.24e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +5.273402134894e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +5.273723968011e-03	err = 3.4080188496e-09	time = 0.55 sec
[ Info: LBFGS: iter   49, time  504.67 s: f = -0.668492867254, ‖∇f‖ = 4.3005e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.434996342170e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +5.435269932189e-03	err = 9.7185194815e-09	time = 0.52 sec
[ Info: LBFGS: iter   50, time  510.48 s: f = -0.668513727270, ‖∇f‖ = 3.9910e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.535741796887e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +5.535812124729e-03	err = 7.5597802907e-09	time = 0.55 sec
[ Info: LBFGS: iter   51, time  516.41 s: f = -0.668532155485, ‖∇f‖ = 4.9090e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.721601939015e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +5.722494303974e-03	err = 9.0525500416e-09	time = 0.56 sec
[ Info: LBFGS: iter   52, time  522.37 s: f = -0.668564505110, ‖∇f‖ = 7.4817e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.854170952025e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +5.855570255032e-03	err = 6.0643662779e-09	time = 0.58 sec
[ Info: LBFGS: iter   53, time  528.39 s: f = -0.668593093779, ‖∇f‖ = 5.7300e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.807157999050e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +5.807180207336e-03	err = 6.8718778031e-09	time = 0.57 sec
[ Info: LBFGS: iter   54, time  534.30 s: f = -0.668613013637, ‖∇f‖ = 4.0964e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.777684965760e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +5.778327844031e-03	err = 6.8696167870e-09	time = 0.53 sec
[ Info: LBFGS: iter   55, time  540.15 s: f = -0.668634613224, ‖∇f‖ = 4.2489e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.776704820114e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +5.777004562734e-03	err = 7.3944931153e-09	time = 0.56 sec
[ Info: LBFGS: iter   56, time  546.08 s: f = -0.668649481004, ‖∇f‖ = 4.5912e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.856675141670e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +5.857261938207e-03	err = 5.2072352127e-09	time = 0.58 sec
[ Info: LBFGS: iter   57, time  552.10 s: f = -0.668670198485, ‖∇f‖ = 3.8195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +5.995689940197e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +5.996264576639e-03	err = 6.8920420466e-09	time = 0.59 sec
[ Info: LBFGS: iter   58, time  558.08 s: f = -0.668688448995, ‖∇f‖ = 3.4508e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.099733436549e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.099798555351e-03	err = 4.5817148903e-09	time = 0.56 sec
[ Info: LBFGS: iter   59, time  564.00 s: f = -0.668696405392, ‖∇f‖ = 5.8372e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.115965920909e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +6.116008207532e-03	err = 4.0996420107e-09	time = 0.56 sec
[ Info: LBFGS: iter   60, time  570.39 s: f = -0.668706977023, ‖∇f‖ = 3.5019e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.154049968178e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +6.154040328233e-03	err = 4.3393809493e-09	time = 0.55 sec
[ Info: LBFGS: iter   61, time  576.75 s: f = -0.668721369553, ‖∇f‖ = 2.8608e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.180409862959e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +6.180481649061e-03	err = 6.7759235085e-09	time = 0.56 sec
[ Info: LBFGS: iter   62, time  583.14 s: f = -0.668732770821, ‖∇f‖ = 3.0068e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.216422621938e-03	err = 1.0000e+00
[ Info: CTMRG conv 17:	obj = +6.216968359833e-03	err = 3.5461853049e-09	time = 0.65 sec
[ Info: LBFGS: iter   63, time  589.47 s: f = -0.668738498959, ‖∇f‖ = 7.3212e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.221199582857e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.221285221433e-03	err = 4.3169649076e-09	time = 0.58 sec
[ Info: LBFGS: iter   64, time  595.63 s: f = -0.668752996535, ‖∇f‖ = 3.0434e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.242137977852e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.242318533556e-03	err = 4.6010616214e-09	time = 0.62 sec
[ Info: LBFGS: iter   65, time  602.00 s: f = -0.668762694601, ‖∇f‖ = 1.9787e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.289082932438e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.289343235396e-03	err = 5.8175591078e-09	time = 0.63 sec
[ Info: LBFGS: iter   66, time  608.31 s: f = -0.668773371271, ‖∇f‖ = 2.8034e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.398675130868e-03	err = 1.0000e+00
[ Info: CTMRG conv 17:	obj = +6.398833095649e-03	err = 4.5926394812e-09	time = 0.63 sec
[ Info: LBFGS: iter   67, time  614.61 s: f = -0.668783730169, ‖∇f‖ = 4.4007e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.490371007286e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.490358021783e-03	err = 6.5723120435e-09	time = 0.59 sec
[ Info: LBFGS: iter   68, time  620.49 s: f = -0.668794471804, ‖∇f‖ = 2.4210e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.535126515606e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +6.535148940005e-03	err = 7.7202755813e-09	time = 0.51 sec
[ Info: LBFGS: iter   69, time  626.32 s: f = -0.668800925048, ‖∇f‖ = 1.7114e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.595058098376e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.595020144173e-03	err = 3.8463595010e-09	time = 0.57 sec
[ Info: LBFGS: iter   70, time  632.22 s: f = -0.668808484765, ‖∇f‖ = 2.5442e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.629079062355e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.629041212706e-03	err = 4.2289764196e-09	time = 0.56 sec
[ Info: LBFGS: iter   71, time  638.44 s: f = -0.668813882136, ‖∇f‖ = 3.7919e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.639171252265e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.639210497400e-03	err = 3.7366899617e-09	time = 0.59 sec
[ Info: LBFGS: iter   72, time  644.68 s: f = -0.668819791141, ‖∇f‖ = 2.1652e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.654973412141e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.655339377376e-03	err = 7.5374087983e-09	time = 0.61 sec
[ Info: LBFGS: iter   73, time  650.69 s: f = -0.668826145765, ‖∇f‖ = 1.9093e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.662806120898e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.662814498802e-03	err = 3.3848722633e-09	time = 0.60 sec
[ Info: LBFGS: iter   74, time  656.72 s: f = -0.668830491901, ‖∇f‖ = 2.4506e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.702717314521e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +6.702719872350e-03	err = 9.3358252520e-09	time = 0.56 sec
[ Info: LBFGS: iter   75, time  662.62 s: f = -0.668836964982, ‖∇f‖ = 2.6465e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.755435653801e-03	err = 1.0000e+00
[ Info: CTMRG conv 17:	obj = +6.755466254550e-03	err = 4.1759687247e-09	time = 0.63 sec
[ Info: CTMRG init:	obj = +6.724498871765e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.724503911311e-03	err = 5.4391737468e-09	time = 0.63 sec
[ Info: LBFGS: iter   76, time  675.39 s: f = -0.668840404782, ‖∇f‖ = 3.1398e-03, α = 3.87e-01, m = 20, nfg = 2
[ Info: CTMRG init:	obj = +6.769517301888e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.769508414086e-03	err = 5.1735891627e-09	time = 0.59 sec
[ Info: LBFGS: iter   77, time  681.45 s: f = -0.668843872314, ‖∇f‖ = 3.1246e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.802028757627e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.802028595034e-03	err = 6.3599636500e-09	time = 0.60 sec
[ Info: LBFGS: iter   78, time  687.52 s: f = -0.668847211574, ‖∇f‖ = 2.4091e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.816904509594e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.816904040071e-03	err = 3.2048624166e-09	time = 0.60 sec
[ Info: LBFGS: iter   79, time  693.54 s: f = -0.668850327797, ‖∇f‖ = 2.0185e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.900461182865e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.900570345503e-03	err = 4.2595997369e-09	time = 0.65 sec
[ Info: LBFGS: iter   80, time  699.67 s: f = -0.668852132421, ‖∇f‖ = 4.1279e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: CTMRG init:	obj = +6.912967976917e-03	err = 1.0000e+00
[ Info: CTMRG conv 17:	obj = +6.913007896384e-03	err = 6.0275386485e-09	time = 0.62 sec
[ Info: CTMRG init:	obj = +6.910412300869e-03	err = 1.0000e+00
[ Info: CTMRG conv 17:	obj = +6.910433755449e-03	err = 4.3218398488e-09	time = 0.62 sec
[ Info: CTMRG init:	obj = +6.901253491862e-03	err = 1.0000e+00
[ Info: CTMRG conv 15:	obj = +6.901253321410e-03	err = 2.0427380336e-09	time = 0.52 sec
[ Info: LBFGS: iter   81, time  718.23 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 4.20e-02, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.72 sec
[ Info: CTMRG init:	obj = +6.903715176268e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777121e-03	err = 4.0082178383e-09	time = 0.66 sec
[ Info: CTMRG init:	obj = +6.904105837467e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486670e-03	err = 5.2863114192e-09	time = 0.59 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   82, time  737.05 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.76 sec
[ Info: CTMRG init:	obj = +6.903715176279e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777131e-03	err = 4.0082185801e-09	time = 0.68 sec
[ Info: CTMRG init:	obj = +6.904105837342e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486544e-03	err = 5.2863118416e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   83, time  756.63 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.71 sec
[ Info: CTMRG init:	obj = +6.903715176288e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777140e-03	err = 4.0082195479e-09	time = 0.66 sec
[ Info: CTMRG init:	obj = +6.904105837343e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486545e-03	err = 5.2863124904e-09	time = 0.61 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   84, time  776.38 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.74 sec
[ Info: CTMRG init:	obj = +6.903715176298e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777150e-03	err = 4.0082209060e-09	time = 0.68 sec
[ Info: CTMRG init:	obj = +6.904105837280e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486481e-03	err = 5.2863111416e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   85, time  796.20 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.74 sec
[ Info: CTMRG init:	obj = +6.903715176296e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777149e-03	err = 4.0082189483e-09	time = 0.66 sec
[ Info: CTMRG init:	obj = +6.904105837102e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486302e-03	err = 5.2863118722e-09	time = 0.60 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   86, time  816.05 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.74 sec
[ Info: CTMRG init:	obj = +6.903715176290e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777143e-03	err = 4.0082193245e-09	time = 0.66 sec
[ Info: CTMRG init:	obj = +6.904105837270e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486472e-03	err = 5.2863100503e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   87, time  835.56 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.71 sec
[ Info: CTMRG init:	obj = +6.903715176296e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777148e-03	err = 4.0082200300e-09	time = 0.65 sec
[ Info: CTMRG init:	obj = +6.904105837224e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486425e-03	err = 5.2863111525e-09	time = 0.57 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   88, time  854.74 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.72 sec
[ Info: CTMRG init:	obj = +6.903715176315e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777166e-03	err = 4.0082194363e-09	time = 0.65 sec
[ Info: CTMRG init:	obj = +6.904105837312e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486514e-03	err = 5.2863100478e-09	time = 0.57 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   89, time  874.42 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.69 sec
[ Info: CTMRG init:	obj = +6.903715176263e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777116e-03	err = 4.0082172169e-09	time = 0.62 sec
[ Info: CTMRG init:	obj = +6.904105837338e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486540e-03	err = 5.2863119129e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   90, time  893.28 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.75 sec
[ Info: CTMRG init:	obj = +6.903715176276e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777129e-03	err = 4.0082175126e-09	time = 0.65 sec
[ Info: CTMRG init:	obj = +6.904105837251e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486452e-03	err = 5.2863111569e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   91, time  913.14 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.77 sec
[ Info: CTMRG init:	obj = +6.903715176272e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777125e-03	err = 4.0082181074e-09	time = 0.68 sec
[ Info: CTMRG init:	obj = +6.904105837232e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486434e-03	err = 5.2863134906e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   92, time  933.21 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.73 sec
[ Info: CTMRG init:	obj = +6.903715176283e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777136e-03	err = 4.0082201639e-09	time = 0.68 sec
[ Info: CTMRG init:	obj = +6.904105837346e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486548e-03	err = 5.2863126347e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   93, time  952.50 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.72 sec
[ Info: CTMRG init:	obj = +6.903715176277e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777130e-03	err = 4.0082177395e-09	time = 0.64 sec
[ Info: CTMRG init:	obj = +6.904105837321e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486523e-03	err = 5.2863123948e-09	time = 0.60 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   94, time  972.36 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.78 sec
[ Info: CTMRG init:	obj = +6.903715176301e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777153e-03	err = 4.0082202712e-09	time = 0.66 sec
[ Info: CTMRG init:	obj = +6.904105837315e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486517e-03	err = 5.2863110108e-09	time = 0.60 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   95, time  992.67 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.73 sec
[ Info: CTMRG init:	obj = +6.903715176310e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777162e-03	err = 4.0082201146e-09	time = 0.65 sec
[ Info: CTMRG init:	obj = +6.904105837264e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486466e-03	err = 5.2863115583e-09	time = 0.55 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   96, time 1012.35 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.74 sec
[ Info: CTMRG init:	obj = +6.903715176248e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777101e-03	err = 4.0082187020e-09	time = 0.67 sec
[ Info: CTMRG init:	obj = +6.904105837262e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486464e-03	err = 5.2863108791e-09	time = 0.56 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   97, time 1032.65 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.79 sec
[ Info: CTMRG init:	obj = +6.903715176342e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777193e-03	err = 4.0082195840e-09	time = 0.68 sec
[ Info: CTMRG init:	obj = +6.904105837350e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486552e-03	err = 5.2863133708e-09	time = 0.58 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   98, time 1052.51 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.77 sec
[ Info: CTMRG init:	obj = +6.903715176276e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777128e-03	err = 4.0082194381e-09	time = 0.64 sec
[ Info: CTMRG init:	obj = +6.904105837352e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486554e-03	err = 5.2863122797e-09	time = 0.56 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter   99, time 1072.16 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02, α = 0.00e+00, m = 20, nfg = 3
[ Info: CTMRG init:	obj = +6.520833675208e-03	err = 1.0000e+00
[ Info: CTMRG conv 20:	obj = +6.527453546061e-03	err = 9.6395155728e-09	time = 0.78 sec
[ Info: CTMRG init:	obj = +6.903715176277e-03	err = 1.0000e+00
[ Info: CTMRG conv 18:	obj = +6.903870777129e-03	err = 4.0082200112e-09	time = 0.70 sec
[ Info: CTMRG init:	obj = +6.904105837289e-03	err = 1.0000e+00
[ Info: CTMRG conv 16:	obj = +6.904113486491e-03	err = 5.2863126525e-09	time = 0.60 sec
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 0.00e+00, dϕ = -8.64e-04, ϕ - ϕ₀ = 0.00e+00
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
┌ Warning: LBFGS: not converged to requested tol after 100 iterations and time 1092.27 s: f = -0.668855329283, ‖∇f‖ = 3.7864e-02
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

