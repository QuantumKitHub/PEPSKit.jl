# [Subspace-iteration CTMRG projectors](@id subspace_iteration)

## Goal and compression

Given a large CTMRG full-infinite environment $M \in \mathbb{C}^{m \times m}$ and $\chi \leq k \leq m$, we seek **range finders** $X,Y$ satisfying

```math
\begin{equation}
    \begin{aligned}
        X &\in \mathbb{C}^{m \times k}, &
        X^{\dagger} X &= I_k,
        \\
        Y &\in \mathbb{C}^{m \times k}, &
        Y^{\dagger} Y &= I_k,
    \end{aligned}
\end{equation}
```

such that the leading $\chi$ singular directions of $M$ can be recovered from the compressed matrix

```math
\begin{equation}
    \rho = Y^{\dagger} M X \in \mathbb{C}^{k \times k}.
\end{equation}
```

$\mathrm{range}(X)$ approximates the dominant right singular subspace; $\mathrm{range}(Y)$ approximates the dominant left singular subspace.

!!! note

    In CTMRG, $M$ is made from two half-infinite environments,

    ```math
    \begin{equation}
        M = L R, \qquad L, R \in \mathbb{C}^{m \times m}.
    \end{equation}
    ```

    Since $m$ is usually much larger than $k$, it is cheaper to compute $M X = L(R X)$, $M^{\dagger} Y = R^{\dagger}(L^{\dagger} Y)$, and $\rho = (Y^{\dagger} L)(R X)$.

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
2. Apply the alternating range update and reduced SVD, then keep the strongest trial directions within the total budget.
3. Between refreshes, recycle the selected space normally.

Refreshes select directions in reduced SVD coordinates; convergence is confirmed on a refresh sweep.
`charge_probe=nothing` preserves broad exploration.
Set `charge_probe=p` to add at most `p` multiplicities per shared sector on refreshes of a compatible cache, clipped to the same per-sector budget cap.
Previously absent charges receive up to `p` trial directions.
Required fixed-space multiplicities remain reserved.
Initialization and incompatible-cache recovery always use broad exploration.
A sufficiently large `p` recovers the broad probe; small values can require several refreshes to move many directions into a newly important charge.

Set `charge_refresh=0` for initial and incompatible-cache exploration only.

## Updating range finders

Introduce notations to describe how $X, Y$ are updated during CTMRG:

- **Step:** $n$ labels the CTMRG step, with incoming cached basis $X_{n-1}$ from the previous step.
  The environment $M_n$ approaches a fixed point $M$ as CTMRG converges.
- **Initialization:** orthogonalize a Gaussian sample to obtain $X_0$.
- **Orthogonalization:** for a full-rank map $A$, $\mathrm{leftorth}(A)$ and $\mathrm{rightorth}(A)$ are the orthonormal factors obtained from thin QR and LQ.

Within each CTMRG step, `n_subspace_iter=q` counts extra stabilized inner iterations with $M_n$ held fixed (default: `q=1`).
Superscript $(j)$ labels the inner iteration.
The updates use the factorized contractions above without forming $M$.

- **Initialize:**

  ```math
  \begin{equation}
      Q_n^{(0)} = \mathrm{leftorth}(M_n X_{n-1}).
  \end{equation}
  ```

- **Iterate**, $j = 1,\ldots,q$ (skip if $q = 0$):

  ```math
  \begin{equation}
      \begin{aligned}
          X_n^{(j)} &= \mathrm{leftorth}(M_n^{\dagger} Q_n^{(j - 1)}), \\
          Q_n^{(j)} &= \mathrm{leftorth}(M_n X_n^{(j)}).
      \end{aligned}
  \end{equation}
  ```

- **$X$ update**: With $Q_n = Q_n^{(q)}$, perform QR decomposition

  ```math
  \begin{equation}
      M_n^{\dagger} Q_n = X_n T_n
      \qquad (X_n^{\dagger} X_n = I_k).
  \end{equation}
  ```

Set $Y_n = Q_n$ and reuse the triangular factor:

```math
\begin{equation}
    \rho_n = Y_n^{\dagger} M_n X_n
    = Q_n^{\dagger} M_n X_n
    = T_n^{\dagger} X_n^{\dagger} X_n = T_n^{\dagger}.
\end{equation}
```

The implementation computes the equivalent LQ of $Q_n^{\dagger} M_n$ and normalizes $\rho_n$ before its SVD.
Only $X_n$ is cached for the next sweep.
Sectors whose left trial basis spans the complete output space use a canonical basis instead of QR; a complete left space gives an exact decomposition without power filtering.

## Why the dominant subspaces are recovered

Write the SVD $M = U \Sigma V^{\dagger}$, with singular vectors $u_i,v_i$ and singular values $\sigma_1 \geq \sigma_2 \geq \cdots$.
Let $U_{\chi}, V_{\chi} \in \mathbb{C}^{m \times \chi}$ denote the leading $\chi$ singular directions, with singular values collected in $\Sigma_{\chi}$.

### Range recurrences filter the singular directions

For a fixed map $M_n = M$, repeated active updates give

```math
\begin{equation}
    \mathrm{range}(X_n)
    = \mathrm{range}\!\left((M^{\dagger} M)^{n(q + 1)} X_0\right).
\end{equation}
```

The reason this favors the dominant directions is

```math
\begin{equation}
    (M^{\dagger} M) v_i = \sigma_i^2 v_i, \qquad
    (M M^{\dagger}) u_i = \sigma_i^2 u_i, \qquad
    M v_i = \sigma_i u_i.
\end{equation}
```

- **Right space:** each application of $M^{\dagger} M$ suppresses a component along $v_j$ relative to one along $v_i$ by $(\sigma_j/\sigma_i)^2$ when $\sigma_j < \sigma_i$.
  With full-rank initial overlap and $\sigma_k > \sigma_{k+1}$, the range converges to the leading $k$ right directions.
- **Left space:** $M v_i = \sigma_i u_i$ turns the filtered right directions into left directions.
- **During CTMRG:** $M_n$ changes, so the fixed-map convergence statement does not apply directly.
  As $M_n$ settles, these recurrences let the cached spaces approach its dominant subspaces, provided the gap persists and updates remain active.

Let $U_k, V_k \in \mathbb{C}^{m \times k}$ collect the leading $k$ singular directions of the limiting map $M$.
Under the overlap, gap, and active-update assumptions above, as $M_n$ converges towards $M$, we expect

```math
\begin{equation}
    \mathrm{range}(X_n) \to \mathrm{range}(V_k),
    \quad
    \mathrm{range}(Y_n) \to \mathrm{range}(U_k).
\end{equation}
```

Since $k \geq \chi$, these limiting ranges contain the leading $\chi$ directions we want to retain.

### Compression preserves the dominant directions

Ideally, after $X, Y$ successfully find the dominant $k$ singular directions,

```math
\begin{equation}
    \mathrm{range}(V_{\chi}) \subseteq \mathrm{range}(X), \quad
    \mathrm{range}(U_{\chi}) \subseteq \mathrm{range}(Y).
\end{equation}
```

Since $X^{\dagger} X = Y^{\dagger} Y = I_k$, when restricting to the dominant singular subspace,

```math
\begin{equation}
    X X^{\dagger} V_{\chi} = V_{\chi}, \qquad Y Y^{\dagger} U_{\chi} = U_{\chi}.
\end{equation}
```

Using $M V_{\chi} = U_{\chi} \Sigma_{\chi}$ and $M^{\dagger} U_{\chi} = V_{\chi} \Sigma_{\chi}$ in $\rho = Y^{\dagger} M X$ gives

```math
\begin{equation}
    \begin{aligned}
        \rho(X^{\dagger} V_{\chi}) &= (Y^{\dagger} U_{\chi})\Sigma_{\chi}, \\
        \rho^{\dagger}(Y^{\dagger} U_{\chi}) &= (X^{\dagger} V_{\chi})\Sigma_{\chi}.
    \end{aligned}
\end{equation}
```

Thus $X^{\dagger} V_{\chi}$ and $Y^{\dagger} U_{\chi}$ collect orthonormal right and left singular directions of $\rho$, with the original singular values $\Sigma_{\chi}$.

Compression cannot increase any singular value, so these captured leading directions remain leading in $\rho$.

### Lifting returns to the full space

Compute $\rho = u s v^{\dagger}$ and embed all $k$ directions back through the rangefinders:

```math
\begin{equation}
    \widehat{U} = Y u, \qquad \widehat{V} = X v.
\end{equation}
```

- **Exact containment:** the leading $\chi$ lifted directions span the target subspaces when $\sigma_{\chi} > \sigma_{\chi+1}$.
- **Approximate containment:** as the range recurrences improve the captured spaces, the lifted subspaces approach the targets, provided this gap stays nonzero.

## Rotating range finders

With `rotate_basis=true`, rotate all $k$ right trial directions, including oversampling directions, before caching:

```math
\begin{equation}
    X_n \leftarrow X_n v_n = \widehat{V}_n.
\end{equation}
```

This preserves the range recurrence.
Charge refreshes always select directions in reduced SVD coordinates.

## Convergence and differentiation

Range finders are updated every sweep.
Corner and edge spectra provide the initial convergence check, as in ordinary CTMRG.
Once these converge, scrambled bond-gauge alignment is applied on the remaining sweeps.
Convergence additionally requires elementwise changes of the aligned environment and changes of the complete trial subspaces at every position and direction to satisfy the CTMRG tolerance:

```math
\begin{equation}
    \epsilon(B_{n-1}, B_n)
    = \frac{\|B_n-B_{n-1}(B_{n-1}^{\dagger}B_n)\|_F}{\sqrt{k}}.
\end{equation}
```

For orthonormal bases this is the RMS sine of the principal angles.
The residual form avoids subtracting nearly equal norms.
Its maximum over the cached bases must satisfy the CTMRG tolerance, including oversampling directions.

`FixedPointGradient` differentiates the joint fixed point of the environment and cached range finders.
Charge allocations are held fixed during differentiation.
The new range finders are expressed in the input environment's enlarged spaces; aligning the output environment back to that frame keeps the pair consistent.
The adjoint uses the same bond-gauge alignment and fixes trial-basis frames before solving the coupled equations.
`gradient_alg=nothing` instead differentiates through all sweeps.

## Computational and memory costs

For square half environments of dimension $m$, range updates and construction of $\rho$ require $4(q + 1)$ half-environment applications and $2(q + 1)$ thin QR/LQ factorizations per position and move direction, in sampled sectors.
One half-environment application to $k$ directions costs $O(m^2 k)$, thin QR/LQ costs $O(m k^2)$, and the reduced SVD costs $O(k^3)$.
Projector construction additionally applies $R$ to $\chi$ directions, costing $O(m^2 \chi)$.
The basis cache stores $mk$ entries per cut, with $O(mk + k^2)$ temporary workspace excluding shared corner tensors.
