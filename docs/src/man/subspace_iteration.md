# [Subspace-iteration SVD](@id subspace_iteration)

`SubspaceIteration` is a warm-started SVD algorithm for the existing CTMRG projectors.
For example:

```julia
alg = SimultaneousCTMRG(;
    projector_alg = :FullInfiniteProjector,
    decomposition_alg = (; alg = :SubspaceIteration, subspace = 24),
)
```

The same `decomposition_alg` works with `HalfInfiniteProjector` and `SequentialCTMRG`.
The retained space is selected by the projector's usual `trunc` option; `subspace` controls the larger trial space.
`C4vEighProjector` requires a Hermitian eigensolver and does not use this SVD algorithm.

## Alternating iteration

Let $M$ be the half- or full-infinite environment and let $X$ embed a $k$-dimensional trial space into its domain.
Each decomposition starts from the previous right trial basis, or an orthogonal Gaussian sample on the first call.
With $q = \mathtt{n\_subspace\_iter}$, the updates are

```math
\begin{equation}
    \begin{aligned}
        Q^{(0)} &= \operatorname{leftorth}(M X), \\
        X^{(j)} &= \operatorname{leftorth}(M^{\dagger} Q^{(j - 1)}), \\
        Q^{(j)} &= \operatorname{leftorth}(M X^{(j)}), \qquad j = 1, \ldots, q.
    \end{aligned}
\end{equation}
```

The final LQ factorization gives both the updated right basis and the reduced matrix:

```math
\begin{equation}
    Q^{\dagger} M = \rho X^{\dagger}, \qquad
    \rho = u s v^{\dagger}, \qquad
    U = Q u, \quad V^{\dagger} = v^{\dagger} X^{\dagger}.
\end{equation}
```

Only the leading retained directions are passed to the existing projector contractions.
All trial directions, including oversampling directions, remain available for recycling.
The reported truncation error measures discarded sampled singular values, rather than a certified full-map residual.
`n_subspace_iter=1` is the default; `0` skips the extra stabilized power step.
`rotate_basis=true` rotates the cached right basis into reduced SVD coordinates.
Complete left sectors use canonical bases instead of redundant QR factorizations.

For a fixed map with a nonzero gap at the trial-space boundary, full-rank initial overlap lets repeated updates approach the dominant singular subspaces.
During CTMRG the map also changes; the warm starts improve as the environment settles.
A single early SI decomposition need not have the accuracy of a converged dense or Krylov SVD.

## Charge allocation

An elementary `subspace` specifies explicit trial multiplicities, clipped to the available map spaces.
A positive integer instead sets a total dimension budget, distributed using the sampled singular spectrum.
Complete symmetry multiplets are preserved, so the budget need not be filled exactly.
Fixed target spaces reserve their required multiplicities in the trial allocation.

Integer allocations are explored initially and every `charge_refresh=10` decompositions of each cut.
`charge_refresh=0` disables periodic exploration; incompatible cached spaces still trigger exploration.
`charge_probe=nothing` probes every shared charge up to the trial budget per sector.
A positive `charge_probe=p` adds at most $p$ multiplicities per sector on refreshes of a compatible cache, including previously absent charges.
Initialization and recovery always probe broadly.
Large $p$ recovers broad probing; small $p$ can require several refreshes to reallocate many directions.

CTMRG creates separate warm starts for every direction and unit-cell position, and fresh caches for each boundary contraction.
Sequential sweeps rotate those slots with the environment.
Standalone calls can reuse a decomposition instance with an explicit truncation:

```julia
using MatrixAlgebraKit: TruncatedAlgorithm
svd_alg = SVDAdjoint(; fwd_alg = TruncatedAlgorithm(
    SubspaceIteration(; subspace = ℂ^12), truncspace(ℂ^8),
))
U, S, V, error = svd_trunc(M, svd_alg)
```

## Contractions and differentiation

The forward pass applies factorized environments to thin trial bases without forming the full environment.
Every internal contraction includes the dual-leg twists used by `half_infinite_environment` and `full_infinite_environment`.
Projector construction and renormalization use the existing code, including its fermionic conventions.
An inexpensive product of factor norms supplies a common scale; the subsequent CTMRG normalization removes that scalar.

`SVDAdjoint` selects the existing KrylovKit SVD reverse rule (`Arnoldi` by default).
Its linear maps and thin pullbacks also use the factorized environment, without constructing the full matrix.
An explicitly selected `TruncPullback` uses the dense Sylvester implementation and materializes the map during differentiation.
The reduced-SVD phase correction and the standard environment global-phase correction remain differentiable.

The adjoint differentiates the truncated SVD, rather than the finite sequence of range updates.
It assumes that the retained singular triplets are accurate and separated from the discarded spectrum, as for `IterSVD`.
Spectral CTMRG convergence alone does not bound this SVD approximation error; increase `subspace` or `n_subspace_iter` if needed.
It does not differentiate the discrete charge allocation or the warm-start cache.
`FixedPointGradient` consequently uses the standard environment-only fixed-point equation and scrambling gauge fix.
CTMRG stops using the usual corner and edge spectra; `info.contraction_metrics.subspace_error` reports the complete trial-space change as a diagnostic.
Backpropagation through sweeps uses the same SVD adjoint and likewise requires sufficiently accurate retained triplets.

## Costs

For square half environments of dimension $m$, full-infinite range updates and reduced-matrix construction require $4(q + 1)$ half-environment applications and $2(q + 1)$ thin QR/LQ factorizations per cut, in sampled sectors.
One application to $k$ directions costs $O(m^2 k)$, thin QR/LQ costs $O(m k^2)$, and the reduced SVD costs $O(k^3)$.
The cached basis contains $mk$ entries per cut.
Half-infinite projectors use the same iteration with two enlarged corners as the factors.
