# [Subspace-iteration CTMRG projectors](@id subspace_iteration)

## Goal and compression

Given a large CTMRG full-infinite environment ``M \in \mathbb{C}^{m \times m}`` and ``\chi \leq k \leq m``, we seek **range finders** ``X,Y`` satisfying

```math
\begin{aligned}
    X &\in \mathbb{C}^{m \times k}, &
    X^{\dagger} X &= I_k,
    \\
    Y &\in \mathbb{C}^{m \times k}, &
    Y^{\dagger} Y &= I_k,
\end{aligned}
```

such that the leading ``\chi`` singular directions of ``M`` can be recovered from the compressed matrix

```math
\rho = Y^{\dagger} M X \in \mathbb{C}^{k \times k}.
```

``\mathrm{range}(X)`` approximates the dominant right singular subspace; ``\mathrm{range}(Y)`` approximates the dominant left singular subspace.

!!! note

    In CTMRG, ``M`` is made from two half-infinite environments,

    ```math
    M = L R, \qquad L, R \in \mathbb{C}^{m \times m}.
    ```

    Since ``m`` is usually much larger than ``k``, it is cheaper to compute ``M X = L(R X)``, ``M^{\dagger} Y = R^{\dagger}(L^{\dagger} Y)``, and ``\rho = (Y^{\dagger} L)(R X)``.

## Trial-space selection

| `subspace` | Charge allocation |
|:--|:--|
| Elementary space | Explicit multiplicities, clipped to the available spaces. |
| Positive integer | A total dimension budget allocated using the sampled singular spectrum. |

`trunc` independently selects the retained space: `subspace=24, trunc=truncrank(16)` allows both charge distributions to change.
Complete multiplets are preserved, so integer budgets may not be filled exactly.
Fixed-space truncation reserves its required multiplicities in the trial allocation.

For integer budgets:

1. Initially and every `charge_refresh=10` sweeps, probe every shared charge, reusing compatible cached directions and adding orthogonal Gaussian directions.
   Exploration temporarily uses up to the trial budget per sector.
2. Apply the selected range update and reduced SVD, then keep the strongest trial directions within the total budget.
3. Between refreshes, recycle the selected space normally.

Refreshes select directions in reduced SVD coordinates; convergence is confirmed on a refresh sweep.
`charge_probe=nothing` preserves broad exploration. Set `charge_probe=p` to add at
most `p` multiplicities per shared sector on refreshes of a compatible cache,
clipped to the same per-sector budget cap. Previously absent charges receive up to
`p` trial directions. Required fixed-space multiplicities remain reserved.
Initialization and incompatible-cache recovery always use broad exploration.
A sufficiently large `p` recovers the broad probe; small values can require several
refreshes to move many directions into a newly important charge.

Set `charge_refresh=0` for initial and incompatible-cache exploration only.

## Updating range finders

Introduce notations to describe how ``X, Y`` are updated during CTMRG:

- **Step:** ``n`` labels the CTMRG step, with incoming cached bases ``X_{n-1},Y_{n-1}`` from the previous step.
  The environment ``M_n`` approaches a fixed point ``M`` as CTMRG converges.
- **Update schemes**: *Independent* schemes update ``X_n, Y_n`` independently, while in *alternating* schemes ``X_n, Y_n`` are related.
- **Initialization:** orthogonalize Gaussian samples to obtain ``X_0`` and, for the independent updates, an independent ``Y_0``.
- **Orthogonalization:** for a full-rank map ``A``, ``\mathrm{leftorth}(A)`` and ``\mathrm{rightorth}(A)`` are the orthonormal factors obtained from thin QR and LQ.

Within each CTMRG step, ``q \geq 0`` counts extra inner iterations with ``M_n`` held fixed.
Superscript ``(j)`` labels the inner iteration.
All updates use the factorized contractions above without forming ``M``.

### Independent scheme

- **Initialize:**

  ```math
  X_n^{(0)} = X_{n-1}, \qquad Y_n^{(0)} = Y_{n-1}.
  ```

- **Iterate**, ``j = 1,\ldots,q + 1``:

  ```math
  \begin{aligned}
      X_n^{(j)} &= \mathrm{leftorth}(M_n^{\dagger} M_n X_n^{(j - 1)}), \\
      Y_n^{(j)} &= \mathrm{leftorth}(M_n M_n^{\dagger} Y_n^{(j - 1)}).
  \end{aligned}
  ```

- **Compress:**

  ```math
  X_n = X_n^{(q + 1)}, \qquad Y_n = Y_n^{(q + 1)}, \qquad
  \rho_n = Y_n^{\dagger} M_n X_n.
  ```

**Complete trial spaces.** If ``k = m`` in every symmetry sector, we skip the power steps and use fixed canonical unitary bases ``X_n,Y_n``:

```math
X_n X_n^{\dagger} = I, \qquad Y_n Y_n^{\dagger} = I, \qquad M_n = Y_n \rho_n X_n^{\dagger}.
```

Thus ``\rho_n`` is an exact change of coordinates: its singular values are those of ``M_n``, and lifting recovers its singular vectors.
All singular directions are already represented, so power filtering adds no information; skipping it avoids roundoff from normal products and redundant basis changes.

### Alternating scheme

- **Initialize:**

  ```math
  Q_n^{(0)} = \mathrm{leftorth}(M_n X_{n-1}).
  ```

- **Iterate**, ``j = 1,\ldots,q`` (skip if ``q = 0``):

  ```math
  \begin{aligned}
      X_n^{(j)} &= \mathrm{leftorth}(M_n^{\dagger} Q_n^{(j - 1)}), \\
      Q_n^{(j)} &= \mathrm{leftorth}(M_n X_n^{(j)}).
  \end{aligned}
  ```

- **``X`` update**: With ``Q_n = Q^{(q)}_n``, perform QR decomposition

  ```math
  M^\dagger_n Q_n = X_n T_n
  \qquad (X^\dagger_n X_n = I_k).
  ```

There are two ways to update ``Y_n``.

- **Simple ``Y`` update**: Directly set ``Y_n = Q_n``.
  Then ``\rho_n`` is already given by ``T^\dagger_n``, because

  ```math
  \rho_n = Y_n^{\dagger} M_n X_n
  = Q^\dagger_n M_n X_n
  = T^\dagger_n X^\dagger_n X_n = T^\dagger_n.
  ```

- **Refined ``Y`` update:** Perform QR decomposition

  ```math
  M_n X_n = Y_n \widetilde{T}_n \qquad
  (Y_n^{\dagger} Y_n = I_k).
  ```

  Then ``\rho_n`` is already given by ``\widetilde{T}_n``, because

  ```math
  \rho_n = Y_n^{\dagger} M_n X_n
  = Y_n^{\dagger} Y_n \widetilde{T}_n = \widetilde{T}_n.
  ```

### Range comparison

All three give the same right-range recurrence in sampled sectors:

```math
\mathrm{range}(X_n)
= \mathrm{range}\!\left((M_n^{\dagger} M_n)^{q + 1} X_{n-1}\right).
```

Their left-range recurrences differ:

| Variant | ``\mathrm{range}(Y_n)`` |
| --- | --- |
| Independent | ``\mathrm{range}\!\left((M_n M_n^{\dagger})^{q + 1} Y_{n-1}\right)`` |
| Alternating | ``\mathrm{range}\!\left(M_n (M_n^{\dagger} M_n)^q X_{n-1}\right)`` |
| Refined alternating | ``\mathrm{range}\!\left(M_n (M_n^{\dagger} M_n)^{q + 1} X_{n-1}\right)`` |

## Why the dominant subspaces are recovered

Write the SVD ``M = U \Sigma V^{\dagger}``, with singular vectors ``u_i,v_i`` and singular values ``\sigma_1 \geq \sigma_2 \geq \cdots``.
Let ``U_\chi, V_\chi \in \mathbb{C}^{m \times \chi}`` denote the leading ``\chi`` singular directions, with singular values collected in ``\Sigma_\chi``.

### Range recurrences filter the singular directions

For a fixed map ``M_n = M``, repeated active updates give

```math
\mathrm{range}(X_n)
= \mathrm{range}\!\left((M^{\dagger} M)^{n(q + 1)} X_0\right).
```

The reason this favors the dominant directions is

```math
(M^{\dagger} M) v_i = \sigma_i^2 v_i, \qquad
(M M^{\dagger}) u_i = \sigma_i^2 u_i, \qquad
M v_i = \sigma_i u_i.
```

- **Right space:** each application of ``M^{\dagger} M`` suppresses a component along ``v_j`` relative to one along ``v_i`` by ``(\sigma_j/\sigma_i)^2`` when ``\sigma_j < \sigma_i``.
  With full-rank initial overlap and ``\sigma_k > \sigma_{k+1}``, the range converges to the leading ``k`` right directions.
- **Left space:** the independent recurrence performs the same filtering with ``M M^{\dagger}``.
  The alternating recurrences use ``M v_i = \sigma_i u_i`` to turn the filtered right directions into left directions; refinement adds one further right-to-left application.
- **During CTMRG:** ``M_n`` changes, so the fixed-map convergence statement does not apply directly.
  As ``M_n`` settles, these recurrences let the cached spaces approach its dominant subspaces, provided the gap persists and updates remain active.

Let ``U_k, V_k \in \mathbb{C}^{m \times k}`` collect the leading ``k`` singular directions of the limiting map ``M``.
Under the overlap, gap, and active-update assumptions above, as ``M_n`` converges towards ``M``, we expect

```math
\mathrm{range}(X_n) \to \mathrm{range}(V_k),
\quad
\mathrm{range}(Y_n) \to \mathrm{range}(U_k).
```

Since ``k \geq \chi``, these limiting ranges contain the leading ``\chi`` directions we want to retain.

### Compression preserves the dominant directions

Ideally, after ``X, Y`` successfully find the dominant ``k`` singular directions,

```math
\mathrm{range}(V_\chi) \subseteq \mathrm{range}(X), \quad
\mathrm{range}(U_\chi) \subseteq \mathrm{range}(Y).
```

Since ``X^{\dagger} X = Y^{\dagger} Y = I_k``, when restricting to the dominant singular subspace,

```math
X X^{\dagger} V_\chi = V_\chi, \qquad Y Y^{\dagger} U_\chi = U_\chi.
```

Using ``M V_\chi = U_\chi \Sigma_\chi`` and ``M^{\dagger} U_\chi = V_\chi \Sigma_\chi`` in ``\rho = Y^{\dagger} M X`` gives

```math
\begin{aligned}
    \rho(X^{\dagger} V_\chi) &= (Y^{\dagger} U_\chi)\Sigma_\chi, \\
    \rho^{\dagger}(Y^{\dagger} U_\chi) &= (X^{\dagger} V_\chi)\Sigma_\chi.
\end{aligned}
```

Thus ``X^{\dagger} V_\chi`` and ``Y^{\dagger} U_\chi`` collect orthonormal right and left singular directions of ``\rho``, with the original singular values ``\Sigma_\chi``.

Compression cannot increase any singular value, so these captured leading directions remain leading in ``\rho``.

### Lifting returns to the full space

Compute ``\rho = u s v^{\dagger}`` and embed all ``k`` directions back through the rangefinders:

```math
\widehat{U} = Y u, \qquad \widehat{V} = X v.
```

- **Exact containment:** the leading ``\chi`` lifted directions span the target subspaces when ``\sigma_\chi > \sigma_{\chi+1}``.
- **Approximate containment:** as the range recurrences improve the captured spaces, the lifted subspaces approach the targets, provided this gap stays nonzero.

## Rotating range finders

After extracting the singular directions, optionally rotate all ``k`` trial directions, including oversampling directions, before caching them for the next step:

```math
X_n \leftarrow X_n v_n = \widehat{V}_n, \qquad Y_n \leftarrow Y_n u_n = \widehat{U}_n.
```

The rotated range finders directly give ``Y_n^{\dagger} M_n X_n = s_n``.
The range recurrence relations are unchanged.

- **Cache:** independent updates retain ``X_n,Y_n``; alternating variants retain only ``X_n``.

## Convergence and differentiation

Range finders are updated every sweep. Corner and edge spectra provide the initial
convergence check, as in ordinary CTMRG. Once these converge, scrambled bond-gauge
alignment is applied on the remaining sweeps. Convergence additionally requires
elementwise changes of the aligned environment and changes of the complete trial
subspaces at every position and direction to satisfy the CTMRG tolerance:

```math
\epsilon(B_{n-1}, B_n)
= \frac{\|B_n-B_{n-1}(B_{n-1}^{\dagger}B_n)\|_F}{\sqrt{k}}.
```

For orthonormal bases this is the RMS sine of the principal angles. The residual
form avoids subtracting nearly equal norms. Its maximum over the cached bases
must satisfy the CTMRG tolerance, including oversampling directions.

`FixedPointGradient` differentiates the joint fixed point of the environment and
cached range finders. Charge allocations are held fixed during differentiation.
The new range finders are expressed in the input environment's enlarged spaces;
aligning the output environment back to that frame keeps the pair consistent.
The adjoint uses the same bond-gauge alignment and fixes trial-basis frames before
solving the coupled equations. `gradient_alg=nothing` instead differentiates through
all sweeps.

## Computational and memory costs

For square half environments of dimension ``m``, the table counts active range updates and construction of ``\rho`` per unit-cell position and CTMRG move direction, at fixed ``k,q`` in sampled sectors.
One half-environment application to ``k`` directions costs ``O(m^2 k)``; thin QR/LQ costs ``O(m k^2)``; the reduced SVD costs ``O(k^3)``.

| Variant | Half-environment applications | QR/LQ count | Basis cache |
| --- | ---: | ---: | ---: |
| Independent | ``8(q + 1) + 2`` | ``2(q + 1)`` | ``2mk`` |
| Alternating | ``4(q + 1)`` | ``2(q + 1)`` | ``mk`` |
| Refined alternating | ``4(q + 1) + 3`` | ``2(q + 1) + 1`` | ``mk`` |

- **Projector construction:** excluded above; standard alternating additionally applies ``R`` to ``\chi`` directions, costing ``O(m^2\chi)``.
- **Memory:** all require ``O(mk + k^2)`` temporary workspace, excluding shared corner tensors.
- **Comparison:** alternating reduces operation counts and the basis cache; all retain the same asymptotic scaling.
  Iteration counts at matched accuracy can differ.
