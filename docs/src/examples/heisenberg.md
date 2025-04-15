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
    input_sha = "7103889def794a5e618bfc772ce19cad7765c6d45cb9ff9519289bb5e420def9"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>Optimizing the 2D Heisenberg model</h1><p>In this example we want to provide a basic rundown of PEPSKit's optimization workflow for PEPS. To that end, we will consider the two-dimensional Heisenberg model on a square lattice</p><p class="tex">$$H = \sum_{\langle i,j \rangle} J_x S^{x}_i S^{x}_j + J_y S^{y}_i S^{y}_j + J_z S^{z}_i S^{z}_j$$</p><p>Here, we want to set <span class="tex">\(J_x=J_y=J_z=1\)</span> where the Heisenberg model is in the antiferromagnetic regime. Due to the bipartite sublattice structure of antiferromagnetic order one needs a PEPS ansatz with a <span class="tex">\(2 \times 2\)</span> unit cell. This can be circumvented by performing a unitary sublattice rotation on all B-sites resulting in a change of parameters to <span class="tex">\((J_x, J_y, J_z)=(-1, 1, -1)\)</span>. This gives us a unitarily equivalent Hamiltonian (with the same spectrum) with a ground state on a single-site unit cell.</p></div>


<div class="markdown"><p>Let us get started by fixing the random seed of this example to make it deterministic:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(123456789);</code></pre>



<div class="markdown"><p>We're going to need only two packages: <code>TensorKit</code>, since we use that for all the underlying tensor operations, and <code>PEPSKit</code> itself. So let us import these:</p></div>

<pre class='language-julia'><code class='language-julia'>using TensorKit, PEPSKit</code></pre>



```
## Defining the Heisenberg Hamiltonian
```@raw html
<div class="markdown">
<p>To create the sublattice rotated Heisenberg Hamiltonian on an infinite square lattice, we use the <code>heisenberg_XYZ</code> method from <a href="https://quantumkithub.github.io/MPSKitModels.jl/dev/">MPSKitModels</a> which is redefined for the <code>InfiniteSquare</code> and reexported in PEPSKit:</p></div>

<pre class='language-julia'><code class='language-julia'>H = heisenberg_XYZ(InfiniteSquare(); Jx=-1, Jy=1, Jz=-1);</code></pre>



```
## Setting up the algorithms and initial guesses
```@raw html
<div class="markdown">
<p>Next, we set the simulation parameters. During optimization, the PEPS will be contracted using CTMRG and the PEPS gradient will be computed by differentiating through the CTMRG routine using AD. Since the algorithmic stack that implements this is rather elaborate, the amount of settings one can configure is also quite large. To reduce this complexity, PEPSKit defaults to (presumably) reasonable settings which also dynamically adapts to the user-specified parameters.</p><p>First, we set the bond dimension <code>Dbond</code> of the virtual PEPS indices and the environment dimension <code>χenv</code> of the virtual corner and transfer matrix indices.</p></div>

<pre class='language-julia'><code class='language-julia'>Dbond = 2;</code></pre>


<pre class='language-julia'><code class='language-julia'>χenv = 16;</code></pre>



<div class="markdown"><p>To configure the CTMRG algorithm, we create a <code>NamedTuple</code> containing different keyword arguments. To see a description of all arguments, see the docstring of <a href="@ref"><code>leading_boundary</code></a>. Here, we want to converge the CTMRG environments up to a specific tolerance and during the CTMRG run keep all index dimensions fixed:</p></div>

<pre class='language-julia'><code class='language-julia'>boundary_alg = (; tol=1e-10, trscheme=(; alg=:fixedspace));</code></pre>



<div class="markdown"><p>Let us also configure the optimizer algorithm. We are going to optimize the PEPS using the L-BFGS optimizer from <a href="https://github.com/Jutho/OptimKit.jl">OptimKit</a>. Again, we specify the convergence tolerance (for the gradient norm) as well as the maximal number of iterations and the BFGS memory size (which is used to approximate the Hessian):</p></div>

<pre class='language-julia'><code class='language-julia'>optimizer_alg = (; alg=:lbfgs, tol=1e-4, maxiter=100, lbfgs_memory=16);</code></pre>



<div class="markdown"><p>Additionally, during optimization, we want to reuse the previous CTMRG environment to initialize the CTMRG run of the current optimization step:</p></div>

<pre class='language-julia'><code class='language-julia'>reuse_env = true;</code></pre>



<div class="markdown"><p>And to control the output information, we set the <code>verbosity</code>:</p></div>

<pre class='language-julia'><code class='language-julia'>verbosity = 1;</code></pre>



<div class="markdown"><p>Next, we initialize a random PEPS which will be used as an initial guess for the optimization. To get a PEPS with physical dimension 2 (since we have a spin-1/2 Hamiltonian) with complex-valued random Gaussian entries, we set:</p></div>

<pre class='language-julia'><code class='language-julia'>peps₀ = InfinitePEPS(randn, ComplexF64, 2, Dbond);</code></pre>



<div class="markdown"><p>The last thing we need before we can start the optimization is an initial CTMRG environment. Typically, a random environment which we converge on <code>peps₀</code> serves as a good starting point:</p></div>

<pre class='language-julia'><code class='language-julia'>env_random = CTMRGEnv(randn, ComplexF64, peps₀, ℂ^χenv);</code></pre>


<pre class='language-julia'><code class='language-julia'>env₀, info_ctmrg = leading_boundary(env_random, peps₀; boundary_alg...);</code></pre>



<div class="markdown"><p>Besides the converged environment, <code>leading_boundary</code> also returns a <code>NamedTuple</code> of informational quantities such as the last (maximal) SVD truncation error:</p></div>

<pre class='language-julia'><code class='language-julia'>info_ctmrg.truncation_error</code></pre>
<pre class="code-output documenter-example-output" id="var-hash924061">0.0017266955527366114</pre>


```
## Ground state search
```@raw html
<div class="markdown">
<p>Finally, we can start the optimization by calling <code>fixedpoint</code> on <code>H</code> with our settings for the boundary (CTMRG) algorithm and the optimizer. This might take a while (especially the precompilation of AD code in this case):</p></div>

<pre class='language-julia'><code class='language-julia'>peps, env, E, info_opt = fixedpoint(
    H, peps₀, env₀; boundary_alg, optimizer_alg, reuse_env, verbosity
);</code></pre>



<div class="markdown"><p>Note that <code>fixedpoint</code> returns the final optimized PEPS, the last converged environment, the final energy estimate as well as a <code>NamedTuple</code> of diagnostics. This allows us to, e.g., analyze the number of cost function calls or the history of gradient norms to evaluate the convergence rate:</p></div>

<pre class='language-julia'><code class='language-julia'>@show info_opt.fg_evaluations</code></pre>
<pre class="code-output documenter-example-output" id="var-hash451283">95</pre>

<pre class='language-julia'><code class='language-julia'>@show info_opt.gradnorms[1:10:end]</code></pre>
<pre class="code-output documenter-example-output" id="var-hash798835">9-element Vector{Float64}:
 1.5288545029023473
 0.42745931621092037
 0.04712909560745584
 0.02455989405793848
 0.012403251650572806
 0.003187578031911067
 0.001660979743126857
 0.0008591829333440263
 0.0003360213833815268</pre>


<div class="markdown"><p>Let's now compare the optimized energy against an accurate Quantum Monte Carlo estimate by <a href="@cite sandvik_computational_2011">Sandvik</a>, where the energy per site was found to be <span class="tex">\(E_{\text{ref}}=−0.6694421\)</span>. From our simple optimization we find:</p></div>

<pre class='language-julia'><code class='language-julia'>@show E</code></pre>
<pre class="code-output documenter-example-output" id="var-hash629819">-0.6625142760601819</pre>


<div class="markdown"><p>While this energy is in the right ballpark, there is still quite some deviation from the accurate reference energy. This, however, can be attributed to the small bond dimension - an optimization with larger bond dimension would approach this value much more closely.</p><p>A more reasonable comparison would be against another finite bond dimension PEPS simulation. For example, Juraj Hasik's data from <span class="tex">\(J_1\text{-}J_2\)</span><a href="https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat">PEPS simulations</a> yields <span class="tex">\(E_{D=2,\chi=16}=-0.660231\dots\)</span> which is more in line with what we find here.</p></div>


```
## Compute the correlation lengths and transfer matrix spectra
```@raw html
<div class="markdown">
<p>In practice, in order to obtain an accurate and variational energy estimate, one would need to compute multiple energies at different environment dimensions and extrapolate in, e.g., the correlation length or the second gap of the transfer matrix spectrum. For that, we would need the <code>correlation_length</code> function, which computes the horizontal and vertical correlation lengths and transfer matrix spectra for all unit cell coordinates:</p></div>

<pre class='language-julia'><code class='language-julia'>ξ_h, ξ_v, λ_h, λ_v = correlation_length(peps, env);</code></pre>


<pre class='language-julia'><code class='language-julia'>@show ξ_h</code></pre>
<pre class="code-output documenter-example-output" id="var-hash117367">1-element Vector{Float64}:
 1.0344958386399636</pre>

<pre class='language-julia'><code class='language-julia'>@show ξ_v</code></pre>
<pre class="code-output documenter-example-output" id="var-hash175800">1-element Vector{Float64}:
 1.0240547254030574</pre>


```
## Computing observables
```@raw html
<div class="markdown">
<p>As a last thing, we want to see how we can compute expectation values of observables, given the optimized PEPS and its CTMRG environment. To compute, e.g., the magnetization, we first need to define the observable:</p></div>

<pre class='language-julia'><code class='language-julia'>σ_z = TensorMap([1.0 0.0; 0.0 -1.0], ℂ^2, ℂ^2)</code></pre>
<pre class="code-output documenter-example-output" id="var-σ_z">TensorMap(ℂ^2 ← ℂ^2):
 1.0   0.0
 0.0  -1.0
</pre>


<div class="markdown"><p>In order to be able to contract it with the PEPS and environment, we define need to define a <code>LocalOperator</code> and specify on which physical spaces and sites the observable acts. That way, the PEPS-environment-operator contraction gets automatically generated (also works for multi-site operators!). See the <a href="@ref"><code>LocalOperator</code></a> docstring for more details. The magnetization is just a single-site observable, so we have:</p></div>

<pre class='language-julia'><code class='language-julia'>M = LocalOperator(fill(ℂ^2, 1, 1), (CartesianIndex(1, 1),) =&gt; σ_z);</code></pre>



<div class="markdown"><p>To evaluate the expecation value, we call:</p></div>

<pre class='language-julia'><code class='language-julia'>@show expectation_value(peps, M, env)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash221986">0.5956380763180077 - 2.7776452606652597e-16im</pre>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
PEPSKit 0.5.0<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

