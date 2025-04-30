```@meta
EditURL = "../../../../examples/bose_hubbard/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuantumKitHub/PEPSKit.jl/gh-pages?filepath=dev/examples/.//bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/QuantumKitHub/PEPSKit.jl/blob/gh-pages/dev/examples/.//bose_hubbard/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/PEPSKit.jl/examples/tree/gh-pages/dev/examples/.//bose_hubbard)


# Optimizing the $U(1)$-symmetric Bose-Hubbard model

This example demonstrates the simulation of the two-dimensional Bose-Hubbard model. In
particular, the point will be to showcase the use of internal symmetries and finite
particle densities in PEPS ground state searches. As we will see, incorporating symmetries
into the simulation consists of initializing a symmetric Hamiltonian, PEPS state and CTM
environment - made possible through TensorKit.

But first let's seed the RNG and import the required modules:

````julia
using Random
using TensorKit, PEPSKit
using MPSKit: add_physical_charge
Random.seed!(2928528935);
````

## Defining the model

We will construct the Bose-Hubbard model Hamiltonian through the
[`bose_hubbard_model`](https://quantumkithub.github.io/MPSKitModels.jl/dev/man/models/#MPSKitModels.bose_hubbard_model),
function from MPSKitModels as reexported by PEPSKit. We'll simulate the model in its
Mott-insulating phase where the ratio $U/t$ is large, since in this phase we expect the
ground state to be well approximated by a PEPS with a manifest global $U(1)$ symmetry.
Furthermore, we'll impose a cutoff at 2 bosons per site, set the chemical potential to zero
and use a simple $1 \times 1$ unit cell:

````julia
t = 1.0
U = 30.0
cutoff = 2
mu = 0.0
lattice = InfiniteSquare(1, 1);
````

Next, we impose an explicit global $U(1)$ symmetry as well as a fixed particle number
density in our simulations. We can do this by setting the `symmetry` argument of the
Hamiltonian constructor to `U1Irrep` and passing one as the particle number density
keyword argument `n`:

````julia
symmetry = U1Irrep
n = 1
H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff, t, U, n);
````

Before we continue, it might be interesting to inspect the corresponding lattice physical
spaces (which is here just a $1 \times 1$ matrix due to the single-site unit cell):

````julia
physical_spaces = H.lattice
````

````
1×1 Matrix{TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}}:
 Rep[TensorKitSectors.U₁](0=>1, 1=>1, -1=>1)
````

Note that the physical space contains $U(1)$ charges -1, 0 and +1. Indeed, imposing a
particle number density of +1 corresponds to shifting the physical charges by -1 to
're-center' the physical charges around the desired density. When we do this with a cutoff
of two bosons per site, i.e. starting from $U(1)$ charges 0, 1 and 2 on the physical level,
we indeed get the observed charges.

## Characterizing the virtual spaces

When running PEPS simulations with explicit internal symmetries, specifying the structure of
the virtual spaces of the PEPS and its environment becomes a bit more involved. For the
environment, one could in principle allow the virtual space to be chosen dynamically during
the boundary contraction using CTMRG by using a truncation scheme that allows for this
(e.g. using `alg=:truncdim` or `alg=:truncbelow` to truncate to a fixed total bond dimension
or singular value cutoff respectively). For the PEPS virtual space however, the structure
has to be specified before the optimization.

While there are a host of techniques to do this in an informed way (e.g. starting from a
simple update result), here we just specify the virtual space manually. Since we're dealing
with a model at unit filling our physical space only contains integer $U(1)$ irreps.
Therefore, we'll build our PEPS and environment spaces using integer $U(1)$ irreps centered
around the zero charge:

````julia
V_peps = U1Space(0 => 2, 1 => 1, -1 => 1)
V_env = U1Space(0 => 6, 1 => 4, -1 => 4, 2 => 2, -2 => 2);
````

## Finding the ground state

Having defined our Hamiltonian and spaces, it is just a matter of plugging this into the
optimization framework in the usual way to find the ground state. So, we first specify all
algorithms and their tolerances:

````julia
boundary_alg = (; tol=1e-8, alg=:simultaneous, trscheme=(; alg=:fixedspace))
gradient_alg = (; tol=1e-6, maxiter=10, alg=:eigsolver, iterscheme=:diffgauge)
optimizer_alg = (; tol=1e-4, alg=:lbfgs, maxiter=150, ls_maxiter=2, ls_maxfg=2);
````

!!! note
	Taking CTMRG gradients and optimizing symmetric tensors tends to be more problematic
    than with dense tensors. In particular, this means that one frequently needs to tweak
    the `boundary_alg`, `gradient_alg` and `optimizer_alg` settings. There rarely is a
    general-purpose set of settings which will always work, so instead one has to adjust
    the simulation settings for each specific application. For example, it might help to
    switch between the CTMRG flavors `alg=:simultaneous` and `alg=:sequential` to
    improve convergence. The evaluation of the CTMRG gradient can be instable, so there it
    is advised to try the different `iterscheme=:diffgauge` and `iterscheme=:fixed` schemes
    as well as different `alg` keywords. Of course the tolerances of the algorithms and
    their subalgorithms also have to be compatible. For more details on the available
    options, see the [`fixedpoint`](@ref) docstring.

Keep in mind that the PEPS is constructed from a unit cell of spaces, so we have to make a
matrix of `V_peps` spaces:

````julia
virtual_spaces = fill(V_peps, size(lattice)...)
peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces)
env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);
````

````
[ Info: CTMRG init:	obj = +1.696011117279e+00 +7.895649499440e-02im	err = 1.0000e+00
[ Info: CTMRG conv 19:	obj = +1.181834754305e+01 -1.525384139600e-11im	err = 3.7197306424e-09	time = 1.27 sec

````

And at last, we optimize (which might take a bit):

````julia
peps, env, E, info = fixedpoint(
    H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg, verbosity=3
)
@show E;
````

````
[ Info: LBFGS: initializing with f = 9.360531870688, ‖∇f‖ = 1.6957e+01
┌ Warning: The function `scale!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Float64}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:91
[ Info: LBFGS: iter    1, time 1425.57 s: f = 0.112865963396, ‖∇f‖ = 5.9876e+00, α = 1.56e+02, m = 0, nfg = 7
┌ Warning: The function `add!!` is not implemented for (values of) type `Tuple{InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, InfinitePEPS{TensorKit.TensorMap{ComplexF64, TensorKit.GradedSpace{TensorKitSectors.U1Irrep, TensorKit.SortedVectorDict{TensorKitSectors.U1Irrep, Int64}}, 1, 4, Vector{ComplexF64}}}, Int64, VectorInterface.One}`;
│ this fallback will disappear in future versions of VectorInterface.jl
└ @ VectorInterface ~/.julia/packages/VectorInterface/J6qCR/src/fallbacks.jl:163
[ Info: LBFGS: iter    2, time 1462.43 s: f = 0.031010339650, ‖∇f‖ = 4.7933e+00, α = 5.55e-01, m = 1, nfg = 2
[ Info: LBFGS: iter    3, time 1468.51 s: f = -0.073336754293, ‖∇f‖ = 1.4989e+00, α = 1.00e+00, m = 2, nfg = 1
[ Info: LBFGS: iter    4, time 1476.82 s: f = -0.113156779559, ‖∇f‖ = 1.4101e+00, α = 1.00e+00, m = 3, nfg = 1
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 4.53e-02, dϕ = -5.09e-01, ϕ - ϕ₀ = -2.42e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    5, time 1496.49 s: f = -0.137358553717, ‖∇f‖ = 1.3312e+00, α = 4.53e-02, m = 4, nfg = 3
┌ Warning: Linesearch not converged after 2 iterations and 3 function evaluations:
│ α = 4.20e-02, dϕ = -3.57e-01, ϕ - ϕ₀ = -1.56e-02
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/linesearches.jl:148
[ Info: LBFGS: iter    6, time 1514.64 s: f = -0.152937922237, ‖∇f‖ = 1.2509e+00, α = 4.20e-02, m = 5, nfg = 3
[ Info: LBFGS: iter    7, time 1538.19 s: f = -0.167923544087, ‖∇f‖ = 3.0315e+00, α = 3.97e-01, m = 6, nfg = 4
[ Info: LBFGS: iter    8, time 1544.27 s: f = -0.200656150027, ‖∇f‖ = 8.4469e-01, α = 1.00e+00, m = 7, nfg = 1
[ Info: LBFGS: iter    9, time 1550.09 s: f = -0.214889985053, ‖∇f‖ = 5.6025e-01, α = 1.00e+00, m = 8, nfg = 1
[ Info: LBFGS: iter   10, time 1555.20 s: f = -0.222943109801, ‖∇f‖ = 9.7822e-01, α = 1.00e+00, m = 9, nfg = 1
[ Info: LBFGS: iter   11, time 1560.12 s: f = -0.230726654816, ‖∇f‖ = 4.2339e-01, α = 1.00e+00, m = 10, nfg = 1
[ Info: LBFGS: iter   12, time 1564.37 s: f = -0.238162325681, ‖∇f‖ = 2.5749e-01, α = 1.00e+00, m = 11, nfg = 1
[ Info: LBFGS: iter   13, time 1568.87 s: f = -0.247343408520, ‖∇f‖ = 3.2371e-01, α = 1.00e+00, m = 12, nfg = 1
[ Info: LBFGS: iter   14, time 1572.04 s: f = -0.253866737463, ‖∇f‖ = 2.4010e-01, α = 1.00e+00, m = 13, nfg = 1
[ Info: LBFGS: iter   15, time 1574.25 s: f = -0.261289395453, ‖∇f‖ = 3.4176e-01, α = 1.00e+00, m = 14, nfg = 1
[ Info: LBFGS: iter   16, time 1576.55 s: f = -0.267194174286, ‖∇f‖ = 2.0409e-01, α = 1.00e+00, m = 15, nfg = 1
[ Info: LBFGS: iter   17, time 1579.49 s: f = -0.269427456764, ‖∇f‖ = 1.4197e-01, α = 1.00e+00, m = 16, nfg = 1
[ Info: LBFGS: iter   18, time 1582.43 s: f = -0.270261160270, ‖∇f‖ = 7.8704e-02, α = 1.00e+00, m = 17, nfg = 1
[ Info: LBFGS: iter   19, time 1585.67 s: f = -0.270680108511, ‖∇f‖ = 6.2948e-02, α = 1.00e+00, m = 18, nfg = 1
[ Info: LBFGS: iter   20, time 1587.48 s: f = -0.271217770832, ‖∇f‖ = 9.0237e-02, α = 1.00e+00, m = 19, nfg = 1
[ Info: LBFGS: iter   21, time 1589.30 s: f = -0.271547867503, ‖∇f‖ = 4.3177e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   22, time 1591.06 s: f = -0.271652426217, ‖∇f‖ = 3.3993e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   23, time 1592.87 s: f = -0.271919640568, ‖∇f‖ = 3.6522e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   24, time 1595.28 s: f = -0.272079486038, ‖∇f‖ = 4.0652e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   25, time 1597.07 s: f = -0.272236289070, ‖∇f‖ = 2.8052e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   26, time 1598.85 s: f = -0.272374999395, ‖∇f‖ = 2.3005e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   27, time 1600.57 s: f = -0.272451064713, ‖∇f‖ = 2.8468e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   28, time 1602.31 s: f = -0.272553269910, ‖∇f‖ = 2.2120e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   29, time 1604.70 s: f = -0.272753621227, ‖∇f‖ = 2.5836e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   30, time 1606.49 s: f = -0.272791434353, ‖∇f‖ = 3.8796e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   31, time 1608.64 s: f = -0.272830578966, ‖∇f‖ = 1.7177e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   32, time 1610.36 s: f = -0.272860065057, ‖∇f‖ = 1.0765e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   33, time 1612.11 s: f = -0.272917839871, ‖∇f‖ = 1.7521e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   34, time 1614.49 s: f = -0.272987230891, ‖∇f‖ = 2.0873e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   35, time 1618.14 s: f = -0.273018580221, ‖∇f‖ = 2.6969e-02, α = 4.93e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   36, time 1619.92 s: f = -0.273067382640, ‖∇f‖ = 1.0394e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   37, time 1621.67 s: f = -0.273081112333, ‖∇f‖ = 6.3470e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   38, time 1624.62 s: f = -0.273087245911, ‖∇f‖ = 9.2635e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   39, time 1627.04 s: f = -0.273091551053, ‖∇f‖ = 5.4087e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   40, time 1628.83 s: f = -0.273095914939, ‖∇f‖ = 4.9572e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   41, time 1630.65 s: f = -0.273099955057, ‖∇f‖ = 6.8773e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   42, time 1632.49 s: f = -0.273103689409, ‖∇f‖ = 5.7556e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   43, time 1635.42 s: f = -0.273109236879, ‖∇f‖ = 6.0202e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   44, time 1637.40 s: f = -0.273117603189, ‖∇f‖ = 1.3781e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   45, time 1639.26 s: f = -0.273127462131, ‖∇f‖ = 8.3496e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   46, time 1641.14 s: f = -0.273135749952, ‖∇f‖ = 7.4036e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   47, time 1643.02 s: f = -0.273146331612, ‖∇f‖ = 8.4981e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   48, time 1646.14 s: f = -0.273156439074, ‖∇f‖ = 1.0154e-02, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   49, time 1648.15 s: f = -0.273161861296, ‖∇f‖ = 9.8380e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   50, time 1650.64 s: f = -0.273166241637, ‖∇f‖ = 4.6402e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   51, time 1652.47 s: f = -0.273167844104, ‖∇f‖ = 2.8874e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   52, time 1654.33 s: f = -0.273169393956, ‖∇f‖ = 3.8294e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   53, time 1656.30 s: f = -0.273172294889, ‖∇f‖ = 5.1406e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   54, time 1658.20 s: f = -0.273176654891, ‖∇f‖ = 5.3195e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   55, time 1662.84 s: f = -0.273178415582, ‖∇f‖ = 6.2372e-03, α = 2.39e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   56, time 1664.87 s: f = -0.273182983220, ‖∇f‖ = 3.8458e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   57, time 1667.76 s: f = -0.273187408223, ‖∇f‖ = 4.3379e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   58, time 1670.67 s: f = -0.273193071706, ‖∇f‖ = 4.4117e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   59, time 1677.19 s: f = -0.273196045493, ‖∇f‖ = 7.7711e-03, α = 5.06e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   60, time 1680.08 s: f = -0.273199424947, ‖∇f‖ = 4.9415e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   61, time 1683.00 s: f = -0.273202267976, ‖∇f‖ = 3.1402e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   62, time 1685.95 s: f = -0.273203233848, ‖∇f‖ = 2.8294e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   63, time 1688.99 s: f = -0.273204600698, ‖∇f‖ = 2.5656e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   64, time 1690.74 s: f = -0.273204925101, ‖∇f‖ = 5.2974e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   65, time 1692.46 s: f = -0.273205966860, ‖∇f‖ = 2.2643e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   66, time 1694.15 s: f = -0.273206609644, ‖∇f‖ = 2.0621e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   67, time 1696.07 s: f = -0.273207765008, ‖∇f‖ = 3.3333e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   68, time 1698.43 s: f = -0.273209290000, ‖∇f‖ = 3.6314e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   69, time 1700.21 s: f = -0.273210336569, ‖∇f‖ = 6.9802e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   70, time 1702.00 s: f = -0.273212030217, ‖∇f‖ = 1.7588e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   71, time 1703.84 s: f = -0.273212480151, ‖∇f‖ = 1.4240e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   72, time 1705.62 s: f = -0.273213078023, ‖∇f‖ = 2.0765e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   73, time 1707.64 s: f = -0.273213880125, ‖∇f‖ = 3.5718e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   74, time 1709.94 s: f = -0.273215045145, ‖∇f‖ = 2.5061e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   75, time 1711.78 s: f = -0.273216916331, ‖∇f‖ = 1.9910e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   76, time 1714.09 s: f = -0.273218689353, ‖∇f‖ = 4.3844e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   77, time 1715.84 s: f = -0.273220202940, ‖∇f‖ = 4.1041e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   78, time 1718.07 s: f = -0.273221806934, ‖∇f‖ = 3.5949e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   79, time 1720.70 s: f = -0.273222485563, ‖∇f‖ = 6.0992e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   80, time 1722.96 s: f = -0.273223791530, ‖∇f‖ = 1.9925e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   81, time 1725.87 s: f = -0.273224262076, ‖∇f‖ = 1.5944e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   82, time 1727.62 s: f = -0.273224852979, ‖∇f‖ = 2.0007e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   83, time 1729.37 s: f = -0.273225773728, ‖∇f‖ = 2.1761e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   84, time 1733.14 s: f = -0.273226388165, ‖∇f‖ = 2.9882e-03, α = 5.20e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   85, time 1735.79 s: f = -0.273227173575, ‖∇f‖ = 1.8724e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   86, time 1737.55 s: f = -0.273227751786, ‖∇f‖ = 1.7098e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   87, time 1739.32 s: f = -0.273228422942, ‖∇f‖ = 1.5751e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   88, time 1742.26 s: f = -0.273229395977, ‖∇f‖ = 1.8136e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   89, time 1747.19 s: f = -0.273229821385, ‖∇f‖ = 2.4985e-03, α = 3.36e-01, m = 20, nfg = 2
[ Info: LBFGS: iter   90, time 1749.35 s: f = -0.273230528832, ‖∇f‖ = 1.6082e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   91, time 1751.23 s: f = -0.273231112443, ‖∇f‖ = 1.5216e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   92, time 1753.03 s: f = -0.273231613563, ‖∇f‖ = 2.6699e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   93, time 1754.77 s: f = -0.273232215697, ‖∇f‖ = 1.4347e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   94, time 1757.16 s: f = -0.273233125782, ‖∇f‖ = 1.7764e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   95, time 1758.93 s: f = -0.273233450052, ‖∇f‖ = 2.7567e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   96, time 1760.68 s: f = -0.273233803468, ‖∇f‖ = 1.2666e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   97, time 1762.45 s: f = -0.273234086238, ‖∇f‖ = 1.0932e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   98, time 1764.25 s: f = -0.273234487308, ‖∇f‖ = 1.5990e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter   99, time 1766.67 s: f = -0.273234975900, ‖∇f‖ = 1.6599e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  100, time 1770.23 s: f = -0.273235076855, ‖∇f‖ = 1.8653e-03, α = 1.62e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  101, time 1772.10 s: f = -0.273235308827, ‖∇f‖ = 9.8269e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  102, time 1773.99 s: f = -0.273235461843, ‖∇f‖ = 8.3532e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  103, time 1776.37 s: f = -0.273235703016, ‖∇f‖ = 1.3504e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  104, time 1778.12 s: f = -0.273236030970, ‖∇f‖ = 1.7152e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  105, time 1779.89 s: f = -0.273236666544, ‖∇f‖ = 1.9163e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  106, time 1781.63 s: f = -0.273237201871, ‖∇f‖ = 2.0606e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  107, time 1783.47 s: f = -0.273237605262, ‖∇f‖ = 1.2100e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  108, time 1785.84 s: f = -0.273237923146, ‖∇f‖ = 1.1911e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  109, time 1787.60 s: f = -0.273238474676, ‖∇f‖ = 1.5922e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  110, time 1789.38 s: f = -0.273239020165, ‖∇f‖ = 3.5400e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  111, time 1791.17 s: f = -0.273239721493, ‖∇f‖ = 1.5841e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  112, time 1792.95 s: f = -0.273240056185, ‖∇f‖ = 8.1807e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  113, time 1795.44 s: f = -0.273240176875, ‖∇f‖ = 8.2119e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  114, time 1797.22 s: f = -0.273240388391, ‖∇f‖ = 9.1300e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  115, time 1798.98 s: f = -0.273240719099, ‖∇f‖ = 1.3486e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  116, time 1800.75 s: f = -0.273241124067, ‖∇f‖ = 1.8644e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  117, time 1802.52 s: f = -0.273241441776, ‖∇f‖ = 1.1376e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  118, time 1805.03 s: f = -0.273241674928, ‖∇f‖ = 1.0744e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  119, time 1807.43 s: f = -0.273242054196, ‖∇f‖ = 1.3968e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  120, time 1809.19 s: f = -0.273242373511, ‖∇f‖ = 3.4093e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  121, time 1810.97 s: f = -0.273242838731, ‖∇f‖ = 1.6035e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  122, time 1813.28 s: f = -0.273243220149, ‖∇f‖ = 1.0189e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  123, time 1815.64 s: f = -0.273243410574, ‖∇f‖ = 1.0299e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  124, time 1818.49 s: f = -0.273243873822, ‖∇f‖ = 1.5353e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  125, time 1822.08 s: f = -0.273244136936, ‖∇f‖ = 1.3969e-03, α = 4.63e-01, m = 20, nfg = 2
[ Info: LBFGS: iter  126, time 1824.72 s: f = -0.273244383827, ‖∇f‖ = 6.9496e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  127, time 1826.72 s: f = -0.273244536785, ‖∇f‖ = 6.2695e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  128, time 1828.67 s: f = -0.273244685913, ‖∇f‖ = 1.3081e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  129, time 1830.62 s: f = -0.273244940432, ‖∇f‖ = 9.2142e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  130, time 1832.67 s: f = -0.273245349898, ‖∇f‖ = 9.5066e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  131, time 1835.35 s: f = -0.273245602436, ‖∇f‖ = 2.1118e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  132, time 1837.11 s: f = -0.273245856212, ‖∇f‖ = 9.4425e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  133, time 1838.84 s: f = -0.273245974747, ‖∇f‖ = 7.2475e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  134, time 1840.59 s: f = -0.273246213451, ‖∇f‖ = 1.0622e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  135, time 1842.36 s: f = -0.273246353289, ‖∇f‖ = 2.1630e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  136, time 1844.66 s: f = -0.273246531186, ‖∇f‖ = 1.0798e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  137, time 1846.38 s: f = -0.273246646636, ‖∇f‖ = 5.4376e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  138, time 1848.11 s: f = -0.273246688623, ‖∇f‖ = 5.9696e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  139, time 1849.88 s: f = -0.273246849584, ‖∇f‖ = 1.0371e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  140, time 1851.66 s: f = -0.273246989170, ‖∇f‖ = 2.0820e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  141, time 1853.95 s: f = -0.273247216059, ‖∇f‖ = 1.2002e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  142, time 1855.75 s: f = -0.273247413465, ‖∇f‖ = 7.8638e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  143, time 1857.53 s: f = -0.273247523052, ‖∇f‖ = 7.4363e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  144, time 1859.32 s: f = -0.273247723357, ‖∇f‖ = 9.7224e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  145, time 1861.14 s: f = -0.273247872194, ‖∇f‖ = 2.6987e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  146, time 1863.03 s: f = -0.273248167442, ‖∇f‖ = 1.3795e-03, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  147, time 1865.26 s: f = -0.273248387392, ‖∇f‖ = 8.2588e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  148, time 1867.08 s: f = -0.273248518043, ‖∇f‖ = 8.5059e-04, α = 1.00e+00, m = 20, nfg = 1
[ Info: LBFGS: iter  149, time 1868.86 s: f = -0.273248701939, ‖∇f‖ = 9.1018e-04, α = 1.00e+00, m = 20, nfg = 1
┌ Warning: LBFGS: not converged to requested tol after 150 iterations and time 1871.22 s: f = -0.273248873410, ‖∇f‖ = 1.7740e-03
└ @ OptimKit ~/.julia/packages/OptimKit/G6i79/src/lbfgs.jl:197
E = -0.2732488734104027

````

We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation
using a cylinder circumference of $L_y = 7$ and a bond dimension of 446, which yields
$E = -0.273284888$:

````julia
E_ref = -0.273284888
@show (E - E_ref) / E_ref;
````

````
(E - E_ref) / E_ref = -0.00013178405092523278

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

