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
    input_sha = "db44a2ada097f4949f9829797025ede6dc3802bbd2f20f5a758232be3d3c7598"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>Néel order in the U(1)-symmetric XXZ model</h1><p>Here, we want to look at a special case of the Heisenberg model, where the <span class="tex">\(x\)</span> and <span class="tex">\(y\)</span> couplings are equal, called the XXZ model</p><p class="tex">$$H_0 = J \big(\sum_{\langle i, j \rangle} S_i^x S_j^x + S_i^y S_j^y + \Delta S_i^z S_j^z \big) .$$</p><p>For appropriate <span class="tex">\(\Delta\)</span>, the model enters an antiferromagnetic phase (Néel order) which we will force by adding staggered magnetic charges to <span class="tex">\(H_0\)</span>. Furthermore, since the XXZ Hamiltonian obeys a <span class="tex">\(U(1)\)</span> symmetry, we will make use of that and work with <span class="tex">\(U(1)\)</span>-symmetric PEPS and CTMRG environments. For simplicity, we will consider spin-<span class="tex">\(1/2\)</span> operators.</p><p>But first, let's make this example deterministic and import the required packages:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(2928528935);</code></pre>


<pre class='language-julia'><code class='language-julia'>using TensorKit, PEPSKit</code></pre>


<pre class='language-julia'><code class='language-julia'>using MPSKit: add_physical_charge</code></pre>



```
## Constructing the model
```@raw html
<div class="markdown">
<p>Let us define the XXZ Hamiltonian with the parameters</p></div>

<pre class='language-julia'><code class='language-julia'>J, Delta, spin = 1.0, 1.0, 1//2;</code></pre>



<div class="markdown"><p>with <span class="tex">\(U(1)\)</span>-symmetric tensors</p></div>

<pre class='language-julia'><code class='language-julia'>symmetry = U1Irrep;</code></pre>



<div class="markdown"><p>on a <span class="tex">\(2 \times 2\)</span> unit cell:</p></div>

<pre class='language-julia'><code class='language-julia'>lattice = InfiniteSquare(2, 2);</code></pre>


<pre class='language-julia'><code class='language-julia'>H₀ = heisenberg_XXZ(ComplexF64, symmetry, lattice; J, Delta, spin);</code></pre>



<div class="markdown"><p>This ensures that our PEPS ansatz can support the bipartite Néel order. As discussed above, we encode the Néel order directly in the ansatz by adding staggered auxiliary physical charges:</p></div>

<pre class='language-julia'><code class='language-julia'>S_aux = [
    U1Irrep(-1//2) U1Irrep(1//2)
    U1Irrep(1//2) U1Irrep(-1//2)
];</code></pre>


<pre class='language-julia'><code class='language-julia'>H = add_physical_charge(H₀, S_aux);</code></pre>



```
## Specifying the symmetric virtual spaces
```@raw html
<div class="markdown">
<p>Before we create an initial PEPS and CTM environment, we need to think about which symmetric spaces we need to construct. Since we want to exploit the global <span class="tex">\(U(1)\)</span> symmetry of the model, we will use TensorKit's <code>U1Space</code>s where we specify dimensions for each symmetry sector:</p></div>

<pre class='language-julia'><code class='language-julia'>V_peps = U1Space(0 =&gt; 2, 1 =&gt; 1, -1 =&gt; 1);</code></pre>


<pre class='language-julia'><code class='language-julia'>V_env = U1Space(0 =&gt; 6, 1 =&gt; 4, -1 =&gt; 4, 2 =&gt; 2, -2 =&gt; 2);</code></pre>



<div class="markdown"><p>From the virtual spaces, we will need to construct a unit cell (a matrix) of spaces which will be supplied to the PEPS constructor. The same is true for the physical spaces, which we will just extract from the Hamiltonian <code>LocalOperator</code>:</p></div>

<pre class='language-julia'><code class='language-julia'>virtual_spaces = fill(V_peps, size(lattice)...);</code></pre>


<pre class='language-julia'><code class='language-julia'>physical_spaces = H.lattice;</code></pre>



```
## Ground state search
```@raw html
<div class="markdown">
<p>From this point onwards it's business as usual: Create an initial PEPS and environment (using the symmetric spaces), specify the algorithmic parameters and optimize:</p></div>

<pre class='language-julia'><code class='language-julia'>boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace));</code></pre>


<pre class='language-julia'><code class='language-julia'>gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge);</code></pre>


<pre class='language-julia'><code class='language-julia'>optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=100, ls_maxiter=2, ls_maxfg=2);</code></pre>


<pre class='language-julia'><code class='language-julia'>peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces);</code></pre>


<pre class='language-julia'><code class='language-julia'>env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);</code></pre>



<div class="markdown"><p>Finally, we can optimize the PEPS with respect to the XXZ Hamiltonian. Note that the optimization might take a while since precompilation of symmetric AD code takes longer and because symmetric tensors do create a bit of overhead (which does pay off at larger bond and environment dimensions):</p></div>

<pre class='language-julia'><code class='language-julia'>peps, env, E, info = fixedpoint(H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg);</code></pre>


<pre class='language-julia'><code class='language-julia'>@show E</code></pre>
<pre class="code-output documenter-example-output" id="var-hash629819">-0.6689439877742164</pre>


<div class="markdown"><p>Note that for the specified parameters <span class="tex">\(J=\Delta=1\)</span>, we simulated the same Hamiltonian as in the <a href="">Heisenberg example</a>. In that example, with a non-symmetric <span class="tex">\(D=2\)</span> PEPS simulation, we reached a ground-state energy of around <span class="tex">\(E_\text{D=2} = -0.6625\dots\)</span>. Again comparing against <a href="@cite sandvik_computational_2011">Sandvik's</a> accurate QMC estimate <span class="tex">\(E_{\text{ref}}=−0.6694421\)</span>, we see that we already got closer to the reference energy.</p></div>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
MPSKit 0.12.6<br>
PEPSKit 0.5.0<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

