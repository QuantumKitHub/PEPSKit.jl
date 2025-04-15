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
    input_sha = "2b256b71b6de545f047d4715bdace175c5017ee23bd042478f67e003666a93ad"
    julia_version = "1.11.4"
-->

<div class="markdown"><h1>Optimizing the <span class="tex">\(U(1)\)</span>-symmetric Bose-Hubbard model</h1><p>This example demonstrates the simulation of the two-dimensional Bose-Hubbard model. In particular, the point will be to showcase the use of internal symmetries and finite particle densities in PEPS ground state searches. As we will see, incorporating symmetries into the simulation consists of initializing a symmetric Hamiltonian, PEPS state and CTM environment - made possible through TensorKit.</p><p>But first let's seed the RNG and import the required modules:</p></div>

<pre class='language-julia'><code class='language-julia'>using Random</code></pre>


<pre class='language-julia'><code class='language-julia'>Random.seed!(2928528935);</code></pre>


<pre class='language-julia'><code class='language-julia'>using TensorKit, PEPSKit</code></pre>


<pre class='language-julia'><code class='language-julia'>using MPSKit: add_physical_charge</code></pre>



```
## Defining the model
```@raw html
<div class="markdown">
<p>We will construct the Bose-Hubbard model Hamiltonian through the <a href="https://quantumkithub.github.io/MPSKitModels.jl/dev/man/models/#MPSKitModels.bose_hubbard_model"><code>bose_hubbard_model</code> function from MPSKitModels.jl</a>, as reexported by PEPSKit. We'll simulate the model in its Mott-insulating phase where the ratio <span class="tex">\(U/t\)</span> is large, since in this phase we expect the ground state to be well approximated by a PEPS with a manifest global <span class="tex">\(U(1)\)</span> symmetry. Furthermore, we'll impose a cutoff at 2 bosons per site, set the chemical potential to zero and use a simple 1x1 unit cell:</p></div>

<pre class='language-julia'><code class='language-julia'>t = 1.0;</code></pre>


<pre class='language-julia'><code class='language-julia'>U = 30.0;</code></pre>


<pre class='language-julia'><code class='language-julia'>cutoff = 2;</code></pre>


<pre class='language-julia'><code class='language-julia'>mu = 0.0;</code></pre>


<pre class='language-julia'><code class='language-julia'>lattice = InfiniteSquare(1, 1);</code></pre>



<div class="markdown"><p>Next, we impose an explicit global U(1) symmetry as well as a fixed particle number density in our simulations. We can do this by setting the <code>symmetry</code> argument of the Hamiltonian constructor to <code>U1Irrep</code> and passing one as the particle number density keyword argument <code>n</code>:</p></div>

<pre class='language-julia'><code class='language-julia'>symmetry = U1Irrep;</code></pre>


<pre class='language-julia'><code class='language-julia'>n = 1;</code></pre>



<div class="markdown"><p>So let's instantiate the symmetric Hamiltonian:</p></div>

<pre class='language-julia'><code class='language-julia'>H = bose_hubbard_model(ComplexF64, symmetry, lattice; cutoff, t, U, n)</code></pre>
<pre class="code-output documenter-example-output" id="var-H">LocalOperator{Tuple{Pair{Tuple{CartesianIndex{2}, CartesianIndex{2}}, TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 2, 2, Vector{ComplexF64}}}, Pair{Tuple{CartesianIndex{2}, CartesianIndex{2}}, TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 2, 2, Vector{ComplexF64}}}, Pair{Tuple{CartesianIndex{2}}, TensorMap{ComplexF64, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, 1, 1, Vector{ComplexF64}}}}, GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}}(GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}[Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1);;], ((CartesianIndex(1, 1), CartesianIndex(1, 2)) =&gt; TensorMap((Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1) ⊗ Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1)) ← (Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1) ⊗ Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1))):
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -2.0000000000000004 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 -2.0000000000000004 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -1.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 -1.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
, (CartesianIndex(1, 1), CartesianIndex(2, 1)) =&gt; TensorMap((Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1) ⊗ Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1)) ← (Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1) ⊗ Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1))):
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 -1.4142135623730951 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -2.0000000000000004 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 -2.0000000000000004 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)):
[:, :, 1, 1] =
 -1.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](0)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 -1.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](0), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](1)) ← (Irrep[TensorKitSectors.U₁](1), Irrep[TensorKitSectors.U₁](1)):
[:, :, 1, 1] =
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](-1)) ← (Irrep[TensorKitSectors.U₁](-1), Irrep[TensorKitSectors.U₁](-1)):
[:, :, 1, 1] =
 0.0 + 0.0im
, (CartesianIndex(1, 1),) =&gt; TensorMap(Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1) ← Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1)):
* Data for sector (Irrep[TensorKitSectors.U₁](0),) ← (Irrep[TensorKitSectors.U₁](0),):
 0.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](1),) ← (Irrep[TensorKitSectors.U₁](1),):
 30.0 + 0.0im
* Data for sector (Irrep[TensorKitSectors.U₁](-1),) ← (Irrep[TensorKitSectors.U₁](-1),):
 0.0 + 0.0im
))</pre>


<div class="markdown"><p>Before we continue, it might be interesting to inspect the corresponding lattice physical spaces:</p></div>

<pre class='language-julia'><code class='language-julia'>physical_spaces = H.lattice</code></pre>
<pre class="code-output documenter-example-output" id="var-physical_spaces">1×1 Matrix{GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}}:
 Rep[TensorKitSectors.U₁](0=&gt;1, 1=&gt;1, -1=&gt;1)</pre>


<div class="markdown"><p>Note that the physical space contains <span class="tex">\(U(1)\)</span> charges -1, 0 and +1. Indeed, imposing a particle number density of +1 corresponds to shifting the physical charges by -1 to 're-center' the physical charges around the desired density. When we do this with a cutoff of two bosons per site, i.e. starting from <span class="tex">\(U(1)\)</span> charges 0, 1 and 2 on the physical level, we indeed get the observed charges.</p><h2>Characterizing the virtual spaces</h2><p>When running PEPS simulations with explicit internal symmetries, specifying the structure of the virtual spaces of the PEPS and its environment becomes a bit more involved. For the environment, one could in principle allow the virtual space to be chosen dynamically during the boundary contraction using CTMRG by using a truncation scheme that allows for this (e.g. using alg=:truncdim or alg=:truncbelow to truncate to a fixed total bond dimension or singular value cutoff respectively). For the PEPS virtual space however, the structure has to be specified before the optimization.</p><p>While there are a host of techniques to do this in an informed way (e.g. starting from a simple update result), here we just specify the virtual space manually. Since we're dealing with a model at unit filling our physical space only contains integer <span class="tex">\(U(1)\)</span> irreps. Therefore, we'll build our PEPS and environment spaces using integer U(1) irreps centered around the zero charge.</p></div>

<pre class='language-julia'><code class='language-julia'>V_peps = U1Space(0 =&gt; 2, 1 =&gt; 1, -1 =&gt; 1);</code></pre>


<pre class='language-julia'><code class='language-julia'>V_env = U1Space(0 =&gt; 6, 1 =&gt; 4, -1 =&gt; 4, 2 =&gt; 2, -2 =&gt; 2);</code></pre>



```
## Finding the ground state
```@raw html
<div class="markdown">
<p>Having defined our Hamiltonian and spaces, it is just a matter of plugging this into the optimization framework in the usual way to find the ground state. So, we first specify all algorithms and their tolerances:</p></div>

<pre class='language-julia'><code class='language-julia'>boundary_alg = (; tol=1e-8, alg=:simultaneous, verbosity=2, trscheme=(; alg=:fixedspace));</code></pre>


<pre class='language-julia'><code class='language-julia'>gradient_alg = (; tol=1e-6, maxiter=10, alg=:eigsolver, iterscheme=:diffgauge);</code></pre>


<pre class='language-julia'><code class='language-julia'>optimizer_alg = (; tol=1e-4, alg=:lbfgs, verbosity=3, maxiter=200, ls_maxiter=2, ls_maxfg=2);</code></pre>



<div class="markdown"><div class="admonition is-note"><header class="admonition-header">Note</header><div class="admonition-body"><p>Taking CTMRG gradients and optimizing symmetric tensors tends to be more problematic than with dense tensors. In particular, this means that one frequently needs to tweak the <code>gradient_alg</code> and <code>optimizer_alg</code> settings. There rarely is a general-purpose set of settings which will always work, so instead one has to adjust the simulation settings for each specific application. </p></div></div><p>Keep in mind that the PEPS is constructed from a unit cell of spaces, so we have to make a matrix of <code>V_peps</code> spaces:</p></div>

<pre class='language-julia'><code class='language-julia'>virtual_spaces = fill(V_peps, size(lattice)...);</code></pre>


<pre class='language-julia'><code class='language-julia'>peps₀ = InfinitePEPS(randn, ComplexF64, physical_spaces, virtual_spaces);</code></pre>


<pre class='language-julia'><code class='language-julia'>env₀, = leading_boundary(CTMRGEnv(peps₀, V_env), peps₀; boundary_alg...);</code></pre>



<div class="markdown"><p>And at last, we optimize (which might take a bit):</p></div>

<pre class='language-julia'><code class='language-julia'>peps, env, E, info = fixedpoint(H, peps₀, env₀; boundary_alg, gradient_alg, optimizer_alg);</code></pre>


<pre class='language-julia'><code class='language-julia'>@show E</code></pre>
<pre class="code-output documenter-example-output" id="var-hash629819">-0.2732668319499373</pre>


<div class="markdown"><p>We can compare our PEPS result to the energy obtained using a cylinder-MPS calculation using a cylinder circumference of <span class="tex">\(L_y = 7\)</span> and a bond dimension of 446, which yields <span class="tex">\(E = -0.273284888\)</span>:</p></div>

<pre class='language-julia'><code class='language-julia'>E_ref = -0.273284888;</code></pre>


<pre class='language-julia'><code class='language-julia'>@show (E - E_ref) / E_ref</code></pre>
<pre class="code-output documenter-example-output" id="var-hash102569">-6.607043000021378e-5</pre>
<div class='manifest-versions'>
<p>Built with Julia 1.11.4 and</p>
MPSKit 0.12.6<br>
PEPSKit 0.5.0<br>
Random 1.11.0<br>
TensorKit 0.14.5
</div>

<!-- PlutoStaticHTML.End -->
```

