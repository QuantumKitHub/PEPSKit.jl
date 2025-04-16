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
    input_sha = "ca1dbf02c01d275e7f30ee6ad0b9581045bcb08edc7db2616b39e259ff696743"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>Fermi-Hubbard model with <span class="tex">\(f\mathbb{Z}_2 \boxtimes U(1)\)</span> symmetry, at large <span class="tex">\(U\)</span> and half-filling</h1><p>In this example, we will demonstrate how to handle fermionic PEPS tensors and how to optimize them. To that end, we consider the two-dimensional Hubbard model</p><p class="tex">$$H = -t \sum_{\langle i,j \rangle} \sum_{\sigma} \left( c_{i,\sigma}^+ c_{j,\sigma}^- + c_{i,\sigma}^- c_{j,\sigma}^+ \right) + U \sum_i n_{i,\uparrow}n_{i,\downarrow} - \mu \sum_i n_i$$</p><p>where <span class="tex">\(\sigma \in \{\uparrow,\downarrow\}\)</span> and <span class="tex">\(n_{i,\sigma} = c_{i,\sigma}^+ c_{i,\sigma}^-\)</span> is the fermionic number operator. As in previous examples, using fermionic degrees of freedom is a matter of creating tensors with the right symmetry sectors - the rest of the simulation workflow remains the same.</p><p>First though, we make the example deterministic by seeding the RNG, and we make our imports:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(2928528937);</code></pre>


<pre class='language-julia'><code class='language-julia'>using TensorKit, PEPSKit</code></pre>


<pre class='language-julia'><code class='language-julia'>using MPSKit: add_physical_charge</code></pre>



```
## Defining the fermionic Hamiltonian
```@raw html
<div class="markdown">
<p>Let us start by fixing the parameters of the Hubbard model. We're going to use a hopping of <span class="tex">\(t=1\)</span> and a large <span class="tex">\(U=8\)</span> on a <span class="tex">\(2 \times 2\)</span> unit cell:</p></div>

<pre class='language-julia'><code class='language-julia'>t = 1.0;</code></pre>


<pre class='language-julia'><code class='language-julia'>U = 8.0;</code></pre>


<pre class='language-julia'><code class='language-julia'>lattice = InfiniteSquare(2, 2);</code></pre>



<div class="markdown"><p>In order to create fermionic tensors, one needs to define symmetry sectors using TensorKit's <a href="@extref"><code>FermionParity</code></a>. Not only do we want use fermion parity but we also want our particles to exploit the global <span class="tex">\(U(1)\)</span> symmetry. The combined product sector can be obtained using the <a href="https://jutho.github.io/TensorKit.jl/stable/lib/sectors/#TensorKitSectors.deligneproduct-Tuple{Sector,%20Sector}">Deligne product</a>, called through <code>⊠</code> which is obtained by typing <code>\boxtimes+TAB</code>. We will not impose any extra spin symmetry, so we have:</p></div>

<pre class='language-julia'><code class='language-julia'>fermion = fℤ₂;</code></pre>


<pre class='language-julia'><code class='language-julia'>particle_symmetry = U1Irrep;</code></pre>


<pre class='language-julia'><code class='language-julia'>spin_symmetry = Trivial;</code></pre>


<pre class='language-julia'><code class='language-julia'>S = fermion ⊠ particle_symmetry;</code></pre>



<div class="markdown"><p>The next step is defining graded virtual PEPS and environment spaces using <code>S</code>. Here we also use the symmetry sector to impose half-filling. That is all we need to define the Hubbard Hamiltonian:</p></div>

<pre class='language-julia'><code class='language-julia'>D, χ = 1, 1;</code></pre>


<pre class='language-julia'><code class='language-julia'>V_peps = Vect[S]((0, 0) =&gt; 2 * D, (1, 1) =&gt; D, (1, -1) =&gt; D);</code></pre>


<pre class='language-julia'><code class='language-julia'>V_env = Vect[S](
    (0, 0) =&gt; 4 * χ, (1, -1) =&gt; 2 * χ, (1, 1) =&gt; 2 * χ, (0, 2) =&gt; χ, (0, -2) =&gt; χ
);</code></pre>


<pre class='language-julia'><code class='language-julia'>S_aux = S((1, -1));</code></pre>


<pre class='language-julia'><code class='language-julia'>H₀ = hubbard_model(ComplexF64, particle_symmetry, spin_symmetry, lattice; t, U);</code></pre>


<pre class='language-julia'><code class='language-julia'>H = add_physical_charge(H₀, fill(S_aux, size(H₀.lattice)...));</code></pre>



```
## Finding the ground state
```@raw html
<div class="markdown">
<p>Again, the procedure of ground state optimization is very similar to before. First, we define all algorithmic parameters:</p></div>

<pre class='language-julia'><code class='language-julia'>boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace));</code></pre>


<pre class='language-julia'><code class='language-julia'>gradient_alg = (; tol=1e-6, alg=:eigsolver, maxiter=10, iterscheme=:diffgauge);</code></pre>


<pre class='language-julia'><code class='language-julia'>optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=100, ls_maxiter=2, ls_maxfg=2);</code></pre>



<div class="markdown"><p>Second, we initialize a PEPS state and environment (which we converge) constructed from symmetric physical and virtual spaces:</p></div>

<pre class='language-julia'><code class='language-julia'>physical_spaces = H.lattice;</code></pre>


<pre class='language-julia'><code class='language-julia'>virtual_spaces = fill(V_peps, size(lattice)...);</code></pre>


<pre class='language-julia'><code class='language-julia'>peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces);</code></pre>


<pre class='language-julia'><code class='language-julia'>env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);</code></pre>



<div class="markdown"><p>And third, we start the ground state search (this does take quite long):</p></div>

<pre class='language-julia'><code class='language-julia'>peps, env, E, info = fixedpoint(H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg);</code></pre>


<pre class='language-julia'><code class='language-julia'>@show E</code></pre>
<pre class="code-output documenter-example-output" id="var-hash629819">7.041937580964341</pre>


<div class="markdown"><p>Finally, let's compare the obtained energy against a reference energy from a QMC study by <a href="@cite qin_benchmark_2016">Qin et al.</a>. With the parameters specified above, they obtain an energy of <span class="tex">\(E_\text{ref} \approx 4 \times -0.5244140625 = -2.09765625\)</span> (the factor 4 comes from the <span class="tex">\(2 \times 2\)</span> unit cell that we use here). Thus, we find:</p></div>

<pre class='language-julia'><code class='language-julia'>E_ref = -2.09765625;</code></pre>


<pre class='language-julia'><code class='language-julia'>@show (E - E_ref) / E_ref</code></pre>
<pre class="code-output documenter-example-output" id="var-hash102569">-4.3570503179271345</pre>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
MPSKit 0.12.6<br>
PEPSKit 0.5.0<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

