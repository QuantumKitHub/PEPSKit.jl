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
    input_sha = "04b1bf91fa5989c8422063a795a45ee272b798f063b64640db18cc2e3e6fa456"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>Hubbard model imaginary time evolution using simple update</h1><p>Once again, we consider the Hubbard model but this time we obtain the ground-state PEPS by imaginary time evoluation. In particular, we'll use the <a href="@ref"><code>SimpleUpdate</code></a> algorithm. As a reminder, we define the Hubbard model as</p><p class="tex">$$H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- + c_{i,\sigma}^- c_{j,\sigma}^+ \right) + U \sum_i n_{i,\uparrow}n_{i,\downarrow} - \mu \sum_i n_i$$</p><p>with <span class="tex">\(\sigma \in \{\uparrow,\downarrow\}\)</span> and <span class="tex">\(n_{i,\sigma} = c_{i,\sigma}^+ c_{i,\sigma}^-\)</span>.</p><p>Let's get started by seeding the RNG and importing the required modules:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(1298351928);</code></pre>


<pre class='language-julia'><code class='language-julia'>using TensorKit, PEPSKit</code></pre>



```
## Defining the Hamiltonian
```@raw html
<div class="markdown">
<p>First, we define the Hubbard model at <span class="tex">\(t=1\)</span> hopping and <span class="tex">\(U=6\)</span> using <code>Trivial</code> sectors for the particle and spin symmetries:</p></div>

<pre class='language-julia'><code class='language-julia'>t, U = 1, 6;</code></pre>


<pre class='language-julia'><code class='language-julia'>H = hubbard_model(Float64, Trivial, Trivial, InfiniteSquare(Nr, Nc); t, U, mu=U / 2);</code></pre>



```
## Running the simple update algorithm
```@raw html
<div class="markdown">
<p>Next, we'll specify the virtual PEPS bond dimension and define the fermionic physical and virtual spaces. For the PEPS ansatz we choose a <span class="tex">\(2 \times 2\)</span> unit cell:</p></div>

<pre class='language-julia'><code class='language-julia'>Dbond = 8;</code></pre>


<pre class='language-julia'><code class='language-julia'>physical_space = Vect[fℤ₂](0 =&gt; 2, 1 =&gt; 2);</code></pre>


<pre class='language-julia'><code class='language-julia'>virtual_space = Vect[fℤ₂](0 =&gt; Dbond / 2, 1 =&gt; Dbond / 2);</code></pre>


<pre class='language-julia'><code class='language-julia'>Nr, Nc = 2, 2;</code></pre>



<div class="markdown"><p>The simple update algorithm evolves an infinite PEPS with weights on the virtual bonds, so we here need to intialize an <a href="@ref"><code>InfiniteWeightPEPS</code></a>. By default, the bond weights will be identity. Note that we here use tensors with real <code>Float64</code> entries:</p></div>

<pre class='language-julia'><code class='language-julia'>wpeps = InfiniteWeightPEPS(rand, Float64, physical_space, virtual_space; unitcell=(Nr, Nc));</code></pre>



<div class="markdown"><p>Before starting the simple update routine, we normalize the vertex tensors:</p></div>

<pre class='language-julia'><code class='language-julia'>for ind in CartesianIndices(wpeps.vertices)
    wpeps.vertices[ind] /= norm(wpeps.vertices[ind], Inf)
end;</code></pre>



<div class="markdown"><p>Let's set algorithm parameters: The plan is to successively decrease the time interval of the Trotter-Suzuki as well as the convergence tolerance such that we obtain a more accurate result at each iteration. To run the simple update, we call <code>simpleupdate</code> where we use the keyword <code>bipartite=false</code> - meaning that we use the full <span class="tex">\(2 \times 2\)</span> unit cell without assuming a bipartite structure. Thus, we can start evolving:</p></div>

<pre class='language-julia'><code class='language-julia'>dts = [1e-2, 1e-3, 4e-4, 1e-4];</code></pre>


<pre class='language-julia'><code class='language-julia'>tols = [1e-6, 1e-8, 1e-8, 1e-8];</code></pre>


<pre class='language-julia'><code class='language-julia'>maxiter = 20000;</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    wpeps_su = wpeps
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        trscheme = truncerr(1e-10) & truncdim(Dbond)
        alg = SimpleUpdate(dt, tol, maxiter, trscheme)
        global wpeps_su, = simpleupdate(wpeps_su, H, alg; bipartite=false)
    end
end;</code></pre>



<div class="markdown"><p>To obtain the evolved <code>InfiniteWeightPEPS</code> as an actual PEPS without weights on the bonds, we can just call the following constructor:</p></div>

<pre class='language-julia'><code class='language-julia'>peps = InfinitePEPS(wpeps_su);</code></pre>



```
## Computing the ground-state energy
```@raw html
<div class="markdown">
<p>In order to compute the energy expectation value with evolved PEPS, we need to converge a CTMRG environment on it. We first converge an environment with a small enviroment dimension and then use that to initialize another run with bigger environment dimension. We'll use <code>trscheme=truncdim(χ)</code> for that such that the dimension is increased during the second CTMRG run:</p></div>

<pre class='language-julia'><code class='language-julia'>χenv₀, χenv = 6, 20;</code></pre>


<pre class='language-julia'><code class='language-julia'>env_space = Vect[fℤ₂](0 =&gt; χenv₀ / 2, 1 =&gt; χenv₀ / 2);</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    env = CTMRGEnv(rand, Float64, peps, env_space)
    for χ in [χenv₀, χenv]
        global env, = leading_boundary(
            env, peps; alg=:sequential, tol=1e-5, trscheme=truncdim(χ)
        )
    end
end;</code></pre>



<div class="markdown"><p>We measure the energy by computing the <code>H</code> expectation value, where we have to make sure to normalize to obtain the energy per site:</p></div>

<pre class='language-julia'><code class='language-julia'>E = expectation_value(peps, H, env) / (Nr * Nc);</code></pre>


<pre class='language-julia'><code class='language-julia'>@show E</code></pre>
<pre class="code-output documenter-example-output" id="var-hash629819">-3.6513224945235674</pre>


<div class="markdown"><p>Finally, we can compare the obtained ground-state energy against the literature, namely the QMC estimates from <a href="@cite qin_benchmark_2016">Qin et al.</a>. We find that the result generally agree:</p></div>

<pre class='language-julia'><code class='language-julia'>Es_exact = Dict(0 =&gt; -1.62, 2 =&gt; -0.176, 4 =&gt; 0.8603, 6 =&gt; -0.6567, 8 =&gt; -0.5243);</code></pre>


<pre class='language-julia'><code class='language-julia'>E_exact = Es_exact[U] - U / 2;</code></pre>


<pre class='language-julia'><code class='language-julia'>@show (E - E_exact) / E_exact</code></pre>
<pre class="code-output documenter-example-output" id="var-hash825781">-0.001470589732937472</pre>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
PEPSKit 0.5.0<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

