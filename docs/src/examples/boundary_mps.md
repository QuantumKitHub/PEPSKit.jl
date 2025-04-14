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
    input_sha = "64649b91ba63790d441e122517aa5bdee3bff1870a75a85af7ecec4fa08b6634"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>Boundary MPS contractions using VUMPS and PEPOs</h1><p>Instead of using CTMRG to contract an infinite PEPS, one can also use an boundary MPSs ansatz to contract the infinite network. In particular, we will here use VUMPS to do so.</p><p>Before we start, we'll fix the random seed for reproducability:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(29384293742893);</code></pre>



<div class="markdown"><p>Besides <code>TensorKit</code> and <code>PEPSKit</code>, we here also need <a href="https://quantumkithub.github.io/MPSKit.jl/stable/"><code>MPSKit</code></a> which implements the VUMPS algorithm as well as the required MPS operations:</p></div>

<pre class='language-julia'><code class='language-julia'>using TensorKit, PEPSKit, MPSKit</code></pre>



```
## Computing a PEPS norm
```@raw html
<div class="markdown">
<p>We start by initializing a random initial infinite PEPS:</p></div>

<pre class='language-julia'><code class='language-julia'>peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(2));</code></pre>



<div class="markdown"><p>To compute its norm, we need to construct the transfer operator corresponding to the partition function representing the overlap <span class="tex">\(\langle \psi_\text{PEPS} | \psi_\text{PEPS} \rangle\)</span>:</p></div>

<pre class='language-julia'><code class='language-julia'>transfer = InfiniteTransferPEPS(peps₀, 1, 1);</code></pre>



<div class="markdown"><p>We then find its leading boundary MPS fixed point, where the corresponding eigenvalue encodes the norm of the state. To that end, let us first we build an initial guess for the boundary MPS, choosing a bond dimension of 20:</p></div>

<pre class='language-julia'><code class='language-julia'>mps₀ = initializeMPS(transfer, [ComplexSpace(20)]);</code></pre>



<div class="markdown"><p>Note that this will just construct a MPS with random Gaussian entries based on the virtual spaces of the supplied transfer operator. Of course, one might come up with a better initial guess (leading to better convergence) depending on the application. To find the leading boundary MPS fixed point, we call <code>leading_boundary</code> using the VUMPS algorithm from MPSKit:</p></div>

<pre class='language-julia'><code class='language-julia'>mps, env, ϵ = leading_boundary(mps₀, transfer, VUMPS(; verbosity=2));</code></pre>



<div class="markdown"><p>The norm of the state per unit cell is then given by the expectation value <span class="tex">\(\langle \psi_\text{MPS} | \mathbb{T} | \psi_\text{MPS} \rangle\)</span>:</p></div>

<pre class='language-julia'><code class='language-julia'>norm_vumps = abs(prod(expectation_value(mps, transfer)));</code></pre>



<div class="markdown"><p>This can be compared to the result obtained using CTMRG, where we see that the results match:</p></div>

<pre class='language-julia'><code class='language-julia'>env_ctmrg, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(20)), peps₀; verbosity=2);</code></pre>


<pre class='language-julia'><code class='language-julia'>norm_ctmrg = abs(norm(peps₀, env_ctmrg));</code></pre>


<pre class='language-julia'><code class='language-julia'>@show abs(norm_vumps - norm_ctmrg) / norm_vumps</code></pre>
<pre class="code-output documenter-example-output" id="var-hash370666">3.1316590722294163e-7</pre>


```
## Working with unit cells
```@raw html
<div class="markdown">
<p>For PEPS with non-trivial unit cells, the principle is exactly the same. The only difference is that now the transfer operator of the PEPS norm partition function has multiple lines, each of which can be represented by an <code>InfiniteTransferPEPS</code> object. Such a multi-line transfer operator is represented by a <code>MultilineTransferPEPS</code> object. In this case, the boundary MPS is an <code>MultilineMPS</code> object, which should be initialized by specifying a virtual space for each site in the partition function unit cell.</p><p>First, we construct a PEPS with a <span class="tex">\(2 \times 2\)</span> unit cell using the <code>unitcell</code> keyword argument and then define the corresponding transfer PEPS:</p></div>

<pre class='language-julia'><code class='language-julia'>peps₀_2x2 = InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell=(2, 2));</code></pre>


<pre class='language-julia'><code class='language-julia'>transfer_2x2 = PEPSKit.MultilineTransferPEPS(peps₀_2x2, 1);</code></pre>



<div class="markdown"><p>Now, the procedure is the same as before: We compute the norm once using VUMPS, once using CTMRG and then compare.</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    mps₀_2x2 = initializeMPS(transfer_2x2, fill(ComplexSpace(20), 2, 2))
    mps_2x2, = leading_boundary(mps₀_2x2, transfer_2x2, VUMPS(; verbosity=2))
    norm_2x2_vumps = abs(prod(expectation_value(mps_2x2, transfer_2x2)))

    env_ctmrg_2x2, = leading_boundary(
        CTMRGEnv(peps₀_2x2, ComplexSpace(20)), peps₀_2x2; verbosity=2 
    )
    norm_2x2_ctmrg = abs(norm(peps₀_2x2, env_ctmrg_2x2))
    
    @show abs(norm_2x2_vumps - norm_2x2_ctmrg) / norm_2x2_vumps
end</code></pre>
<pre class="code-output documenter-example-output" id="var-mps_2x2">0.046417037097529805</pre>


<div class="markdown"><p>Again, the results are compatible. Note that for larger unit cells and non-Hermitian PEPS the VUMPS algorithm may become unstable, in which case the CTMRG algorithm is recommended.</p><h2>Contracting PEPO overlaps</h2><p>Using exactly the same machinery, we can contract partition functions which encode the expectation value of a PEPO for a given PEPS state. As an example, we can consider the overlap of the PEPO correponding to the partition function of 3D classical Ising model with our random PEPS from before and evaluate the overlap <span class="tex">\(\langle \psi_\text{PEPS} | O_\text{PEPO} | \psi_\text{PEPS} \rangle\)</span>.</p><p>The classical Ising PEPO is defined as follows:</p></div>

<pre class='language-julia'><code class='language-julia'>function ising_pepo(β; unitcell=(1, 1, 1))
    t = ComplexF64[exp(β) exp(-β); exp(-β) exp(β)]
    q = sqrt(t)

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * 
        q[-1; 1] *
        q[-2; 2] *
        q[-3; 3] *
        q[-4; 4] *
        q[-5; 5] *
        q[-6; 6]
    O = TensorMap(o, ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)')

    return InfinitePEPO(O; unitcell)
end;</code></pre>



<div class="markdown"><p>To evaluate the overlap, we instantiate the PEPO and the corresponding <code>InfiniteTransferPEPO</code> in the right direction, on the right row of the partition function (trivial here):</p></div>

<pre class='language-julia'><code class='language-julia'>pepo = ising_pepo(1);</code></pre>


<pre class='language-julia'><code class='language-julia'>transfer_pepo = InfiniteTransferPEPO(peps₀, pepo, 1, 1);</code></pre>



<div class="markdown"><p>As before, we converge the boundary MPS using VUMPS and then compute the expectation value:</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    mps₀_pepo = initializeMPS(transfer_pepo, [ComplexSpace(20)])
    mps_pepo, = leading_boundary(mps₀_pepo, transfer_pepo, VUMPS(; verbosity=2))
    @show abs(prod(expectation_value(mps_pepo, transfer_pepo)))
end</code></pre>
<pre class="code-output documenter-example-output" id="var-mps₀_pepo">102.19789137363612</pre>


<div class="markdown"><p>These objects and routines can be used to optimize PEPS fixed points of 3D partition functions, see for example <a href="@cite vanderstraeten_residual_2018">Vanderstraeten et al.</a></p></div>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
MPSKit 0.12.6<br>
PEPSKit 0.5.0<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

