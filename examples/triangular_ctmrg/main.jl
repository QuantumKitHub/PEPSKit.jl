using Markdown #hide
md"""
# [Triangular CTMRG and optimization](@id examples_triangular_ctmrg)

This example contracts and optimizes an antiferromagnetic spin-1/2 Heisenberg PEPS
on the triangular lattice using [`InfinitePEPSTriangular`](@ref),
[`CTMRGEnvTriangular`](@ref), and the native [`fixedpoint`](@ref) interface.
A single-site tensor and fixed 120° sublattice rotations represent a three-sublattice
state. We start from a seeded random real tensor with `D = 2` and a complex
environment with `χ = 32`. Check convergence in `χ` before drawing quantitative
conclusions from the energy.

From the repository root, prepare the example environment once with
`julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`,
then run `julia --project=examples --threads=2 examples/triangular_ctmrg/main.jl`.
"""

using Random
using TensorKit, PEPSKit
using TensorKitTensors.SpinOperators
Random.seed!(123456789);

md"""
## Unit cell and nearest neighbors

PEPSKit indexes the unit cell as a Julia matrix: `[r, c]` means row `r`, column `c`,
starting at 1. For `[A B C; D E F; G H I]`, `[2, 1]` is `D` and `[1, 2]` is `B`.
The six virtual legs are ordered `NW, NE, E, SE, SW, W`, with offsets
`(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)` respectively.

To count every nearest-neighbor bond once, we use the following three offsets:

| Bond family | Offset `(Δr, Δc)` | Virtual leg | Relative rotation |
|:------------|:------------------|:------------|:------------------|
| 0°          | `(0, 1)`          | E           | `2π/3`            |
| 60°         | `(1, -1)`         | SW          | `-4π/3`           |
| 120°        | `(1, 0)`          | SE          | `-2π/3`           |

The bond-family angles are unoriented; the relative rotations refer to the directed
offsets in the table. With `R(θ) = exp(-im * θ * S_y)`, the physical tensor at
`[r, c]` is `R((r - 1)q[1] + (c - 1)q[2]) * A`, where `q = (-2π/3, 2π/3)`.
This gives `[A B C; C A B; B C A]`, with `B = R(2π/3) * A` and
`C = R(4π/3) * A`, up to irrelevant spin-1/2 tensor phases. It is the PEPSKit
version of the variPEPS pattern `[A B C; B C A; C A B]` with `q = (2π/3, 2π/3)`.

We optimize `A` in a single-site unit cell. All three sublattices share its
variational parameters, and the wavevector stays fixed during optimization.
"""

D = 2
χ = 32
P, V = ℂ^2, ℂ^D
peps₀ = InfinitePEPSTriangular(randn, Float64, P, V; unitcell = (1, 1));

md"""
## Hamiltonian and CTMRG

The Heisenberg interaction is invariant under a common rotation of both spins, so
only their relative rotation enters each bond. We absorb it into the gate on the
second site: `hδ = (id(P) ⊗ R(q⋅δ))' * h * (id(P) ⊗ R(q⋅δ))`. The diagonal
angle `-4π/3` gives the same gate as `2π/3`. This lets the existing
[`LocalOperator`](@ref) and `expectation_value` interfaces evaluate the physical
three-sublattice energy using one tensor and three bond gates.

We use antiferromagnetic coupling `J = 1`. Keep the neighbor coordinates unwrapped,
including those outside the unit cell: the contractions apply periodicity while
preserving which bond is intended. Triangular expectation values support onsite and nearest-neighbor
terms in all three bond families.
"""

h = S_x_S_x() + S_y_S_y() + S_z_S_z()
offsets = (CartesianIndex(0, 1), CartesianIndex(1, -1), CartesianIndex(1, 0))
q = (-2π / 3, 2π / 3)
rotation(θ) = TensorMap([cos(θ / 2) -sin(θ / 2); sin(θ / 2) cos(θ / 2)], P, P)
gates = map(offsets) do δ
    U = id(P) ⊗ rotation(q[1] * δ[1] + q[2] * δ[2])
    return U' * h * U
end
H = LocalOperator(
    physicalspace(peps₀),
    [(I, I + δ) => gate for I in CartesianIndices(size(peps₀)) for (δ, gate) in zip(offsets, gates)],
);

md"""
The environment constructor takes the doubled virtual space `V ⊗ V'` of the PEPS
norm network and an environment space `ℂ^χ`. We initialize it with `ones`.
The PEPS tensor and the rotated Heisenberg gates are real, but the current gauge
fixing for fixed-point differentiation requires a complex environment.

[`leading_boundary`](@ref) selects triangular CTMRG from the environment; here we
also name it explicitly, so the same settings can be passed to the optimizer.
We use the default `projector_alg = :full` for the corner construction.
The alternative `:twothirds` uses two thirds of the environment. `conditioning = true` (the default) conditions the direct
Ta/Tb edge split. It does not toggle corner conditioning.
"""

env_initial = CTMRGEnvTriangular(ones, ComplexF64, V ⊗ V', ℂ^χ; unitcell = size(peps₀))
boundary_alg = (;
    alg = :SimultaneousCTMRGTriangular, tol = 1.0e-7, maxiter = 100,
    projector_alg = :full, trunctype = :FixedSpaceTruncation,
)
env₀, info_ctmrg = leading_boundary(env_initial, peps₀; boundary_alg...);
@show info_ctmrg.converged info_ctmrg.convergence_error info_ctmrg.contraction_metrics;

md"""
Convergence compares the corner and edge singular-value spectra. The truncation
metrics describe the local SVD errors, not the error in the energy. Check CTMRG
convergence and increase `χ` when assessing the accuracy of an entangled calculation.

`expectation_value` sums all terms in `H`, so this Hamiltonian gives the energy per
unit cell. Divide by the number of sites to obtain the energy per site:
"""

E₀ = real(expectation_value(peps₀, H, env₀))
@show E₀ / length(peps₀);

md"""
## Native optimization

We use L-BFGS with [`FixedPointGradient`](@ref) and a GMRES linear solver to
differentiate the CTMRG fixed point. Triangular optimization currently requires
`precondition_alg = nothing`
and does not support symmetrization. Alternatively, `gradient_alg = nothing`
differentiates through the CTMRG iterations.

PEPSKit adapts the CTMRG and gradient tolerances during optimization, using looser
solves when the gradient is large. Check the final gradient norm and increase the
optimization budget if necessary.
Convergence of CTMRG at a fixed `χ` does not establish convergence in `χ`; repeat
the calculation at larger environment dimensions to assess the energy.
"""

peps, env, E, info_opt = fixedpoint(
    H, peps₀, env₀;
    boundary_alg,
    gradient_alg = (; alg = :FixedPointGradient, solver_alg = (; alg = :GMRES)),
    precondition_alg = nothing,
    optimizer_alg = (; alg = :LBFGS, tol = 1.0e-6, maxiter = 100), verbosity = 3,
);
@show E / length(peps) info_opt.gradnorms[end];

md"""
The `D = 2` ansatz allows entanglement beyond the 120° product state, whose
energy is `-3/8` per site. The returned `E` is the energy per unit cell, which here
contains a single site. Increase both `D` and `χ` to study the quantum ground state.
"""
