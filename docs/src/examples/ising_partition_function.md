```@raw html
<style>
    #documenter-page table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    #documenter-page pre, #documenter-page div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "847911120149e40a5b86c8d8f00a7788bc19f57f04ccd20e29e09ffc5fb79b0f"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>2D classical Ising partition function using CTMRG</h1><p>All previous examples dealt with quantum systems, describing their states by <code>InfinitePEPS</code> that can be contracted using CTMRG or VUMPS techniques. Here, we shift our focus towards classical physics and consider the 2D classical Ising model with the partition function</p><p class="tex">$$\mathcal{Z}(\beta) = \sum_{\{s\}} \exp(-\beta H(s)) \text{ with } H(s) = -J \sum_{\langle i, j \rangle} s_i s_j .$$</p><p>The idea is to encode the partition function into an infinite square lattice of rank-4 tensors which can then be contracted using CTMRG. These rank-4 tensors are represented by <a href="@ref"><code>InfinitePartitionFunction</code></a> states, as we will see.</p><p>But first, let's seed the RNG and import all required modules:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(234923);</code></pre>


<pre class='language-julia'><code class='language-julia'>using LinearAlgebra, TensorKit, PEPSKit, QuadGK</code></pre>



```
## Defining the partition function
```@raw html
<div class="markdown">
<p>The first step is to define the rank-4 tensor that, when contracted on a square lattice, evaluates to the partition function value at a given <span class="tex">\(\beta\)</span>. Since we later want to compute the magnetization and energy, we define the appropriate rank-4 tensors as well:</p></div>

<pre class='language-julia'><code class='language-julia'>function classical_ising(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4] := O[3 4; 2 1] * nt[-3; 3] * nt[-4; 4] * nt[-2; 2] * nt[-1; 1]

    # magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4] := M[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* nt
    @tensor e_hor[-1 -2; -3 -4] :=
        O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * e[-4; 4]
    @tensor e_vert[-1 -2; -3 -4] :=
        O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * e[-3; 3] * nt[-4; 4]
    e = e_hor + e_vert

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end;</code></pre>



<div class="markdown"><p>So let's initialize these tensors at inverse temperature <span class="tex">\(\beta=0.6\)</span> and construct the corresponding <code>InfinitePartitionFunction</code>:</p></div>

<pre class='language-julia'><code class='language-julia'>beta = 0.6;</code></pre>


<pre class='language-julia'><code class='language-julia'>O, M, E = classical_ising(; beta);</code></pre>


<pre class='language-julia'><code class='language-julia'>Z = InfinitePartitionFunction(O);</code></pre>



```
## Contracting the partition function
```@raw html
<div class="markdown">
<p>Next, we can contract the partition function as per usual by constructing a <code>CTMRGEnv</code> and calling <code>leading_boundary</code>:</p></div>

<pre class='language-julia'><code class='language-julia'>χenv = 20;</code></pre>


<pre class='language-julia'><code class='language-julia'>env₀ = CTMRGEnv(Z, χenv);</code></pre>


<pre class='language-julia'><code class='language-julia'>env, = leading_boundary(env₀, Z; tol=1e-8, maxiter=500);</code></pre>



<div class="markdown"><p>Note that CTMRG environments for partition functions differ from the PEPS environments only by the edge tensors - instead of two legs connecting the edges and the PEPS-PEPS sandwich, there is only one leg connecting the edges and the partition function tensor:</p></div>

<pre class='language-julia'><code class='language-julia'>space.(env.edges)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash404067">4×1×1 Array{TensorMapSpace{ComplexSpace, 2, 1}, 3}:
[:, :, 1] =
 (ℂ^20 ⊗ ℂ^2) ← ℂ^20
 (ℂ^20 ⊗ ℂ^2) ← ℂ^20
 (ℂ^20 ⊗ (ℂ^2)') ← ℂ^20
 (ℂ^20 ⊗ (ℂ^2)') ← ℂ^20</pre>


<div class="markdown"><p>To compute the value of the partition function, we have to contract <code>Z</code> with the converged environment using <a href="@ref"><code>network_value</code></a>. Additionally, we will compute the magnetization and energy (per site), again using <a href="@ref"><code>expectation_value</code></a> but this time also specifying the index in the unit cell, where we want to insert the local tensor:</p></div>

<pre class='language-julia'><code class='language-julia'>λ = network_value(Z, env);</code></pre>


<pre class='language-julia'><code class='language-julia'>m = expectation_value(Z, (1, 1) =&gt; M, env);</code></pre>


<pre class='language-julia'><code class='language-julia'>e = expectation_value(Z, (1, 1) =&gt; E, env);</code></pre>



```
## Comparing against the exact Onsager solution
```@raw html
<div class="markdown">
<p>In order to assess our results, we will compare against the <a href="https://en.wikipedia.org/wiki/Square_lattice_Ising_model#Exact_solution">exact Onsager solution</a>. To that end, we compute the exact free energy, magnetization and energy per site (where <code>quadgk</code> performs the integral from <span class="tex">\(0\)</span> to <span class="tex">\(\pi/2\)</span>):</p></div>

<pre class='language-julia'><code class='language-julia'>function classical_ising_exact(; beta=log(1 + sqrt(2)) / 2, J=1.0)
    K = beta * J

    k = 1 / sinh(2 * K)^2
    F = quadgk(
        theta -&gt; log(cosh(2 * K)^2 + 1 / k * sqrt(1 + k^2 - 2 * k * cos(2 * theta))), 0, pi
    )[1]
    f = -1 / beta * (log(2) / 2 + 1 / (2 * pi) * F)

    m = 1 - (sinh(2 * K))^(-4) &gt; 0 ? (1 - (sinh(2 * K))^(-4))^(1 / 8) : 0

    E = quadgk(theta -&gt; 1 / sqrt(1 - (4 * k) * (1 + k)^(-2) * sin(theta)^2), 0, pi / 2)[1]
    e = -J * cosh(2 * K) / sinh(2 * K) * (1 + 2 / pi * (2 * tanh(2 * K)^2 - 1) * E)

    return f, m, e
end;</code></pre>


<pre class='language-julia'><code class='language-julia'>f_exact, m_exact, e_exact = classical_ising_exact(; beta);</code></pre>



<div class="markdown"><p>And indeed, we do find agreement between the exact and CTMRG values (keeping in mind that energy accuracy is limited by the environment dimension and the lack of proper extrapolation):</p></div>

<pre class='language-julia'><code class='language-julia'>@show (-log(λ) / beta - f_exact) / f_exact</code></pre>
<pre class="code-output documenter-example-output" id="var-hash171174">-1.1009271732942546e-15 + 1.7075080586064297e-16im</pre>

<pre class='language-julia'><code class='language-julia'>@show (abs(m) - abs(m_exact)) / abs(m_exact)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash133013">0.0</pre>

<pre class='language-julia'><code class='language-julia'>@show (e - e_exact) / e_exact # accuracy limited by bond dimension and maxiter</code></pre>
<pre class="code-output documenter-example-output" id="var-hash957078">-0.02373206809909008 - 6.466073768041896e-17im</pre>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
LinearAlgebra 1.11.0<br>
PEPSKit 0.5.0<br>
QuadGK 2.11.2<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

