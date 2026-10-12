# Environments

!!! note
    This section is still under construction.

## Split CTMRG

[`SplitCTMRGEnv`](@ref) stores separate ket and bra factors for each CTMRG edge, connected by an interlayer bond.
[`SplitCTMRG`](@ref) absorbs and projects the two layers sequentially using the full projectors of [Naumann et al., Phys. Rev. B 111, 235116 (2025)](https://doi.org/10.1103/PhysRevB.111.235116).
Its default [`SplitProjector`](@ref) constructs the four projector types directly from their split environment halves.
The outer and interlayer spaces can be chosen independently:

```julia
env₀ = SplitCTMRGEnv(ψ, ComplexSpace(32); interlayer_space = ComplexSpace(24))
alg = SplitCTMRG(; tol = 1.0e-8, maxiter = 100)
env, info = leading_boundary(env₀, ψ, alg)
energy = expectation_value(ψ, H, CTMRGEnv(env))
```

By default, both spaces remain fixed during iteration.
Use `trunc` for the outer projectors and `interlayer_trunc` for the interlayer projectors to select other truncation strategies.
When omitted, `interlayer_trunc` inherits `trunc`; the default `FixedSpaceTruncation()` preserves the existing spaces of each bond independently.
`SplitCTMRGEnv(env::CTMRGEnv)` factorizes an existing environment exactly, while its `trunc` keyword can reduce the interlayer spaces.
`CTMRGEnv(env::SplitCTMRGEnv)` contracts the interlayer bonds so the existing expectation-value routines can be reused.
This version supports PEPS overlap networks, full projectors, and automatic differentiation through finite sweeps or [`FixedPointGradient`](@ref).
For comparisons with variPEPS, select its `Projector_Method.FULL` option; its default `FISHMAN` construction reweights the environment halves and uses different projectors.

Fixed-point differentiation uses [`ScramblingEnvGauge`](@ref) on the alternating ket and bra boundary factors to align both outer and interlayer gauges.
The ordinary and split environments use the same boundary-MPS gauge method, with each split edge contributing two MPS sites instead of one.
Both sets of bond spaces are held fixed in the differentiated iteration, including when the forward run uses other truncation strategies.
Use a complex environment for fixed-point gradients, as for conventional asymmetric CTMRG.
The first version contracts split edges into `CTMRGEnv` for energy evaluation and local preconditioning, preserving the derivative through both factors:

```julia
env₀ = SplitCTMRGEnv(randn, ComplexF64, ψ, ComplexSpace(32); interlayer_space = ComplexSpace(24))
opt_alg = PEPSOptimize(;
    boundary_alg = SplitCTMRG(; tol = 1.0e-8),
    gradient_alg = FixedPointGradient(; solver_alg = (; alg = :GMRES)),
)
ψ_opt, env_opt, energy, info = fixedpoint(H, ψ, env₀, opt_alg)
```

`ImplicitGradient` is not supported for split CTMRG.
