# Precompilation using PrecompileTools.jl

For certain PEPSKit applications, the "time to first execution" (TTFX) can be quite long.
If frequent recompilation is required this can become a significant time sink.
Especially in simulations involving AD code, the precompilation times of Zygote tend to be particularly bad.

Fortunately, there is an easy way out using [PrecompileTools](https://julialang.github.io/PrecompileTools.jl/stable/).
By writing a precompilation script that executes and precompiles a toy problem which is suited to one's personal problem, one can cut down significantly on the TTFX.
To see how that works in the context of PEPSKit, we will closely follow the PrecompileTools [docs](https://julialang.github.io/PrecompileTools.jl/stable/#Tutorial:-local-%22Startup%22-packages).

Let's say we have a project where we want to speed up the TTFX, located in a project environment called `YourProject`.
Inside that project folder, we generate a `Startup` module which will contain the toy problem that we want to precompile:

```
(YourProject) pkg> generate Startup
  Generating  project Startup:
    Startup/Project.toml
    Startup/src/Startup.jl

(YourProject) pkg> dev ./Startup
   Resolving package versions...
    Updating `/YourProject/Project.toml`
  [e9c42744] + Startup v0.1.0 `Startup`
    Updating `/tmp/Project1/Manifest.toml`
  [e9c42744] + Startup v0.1.0 `Startup`

(YourProject) pkg> activate Startup/
  Activating project at `/YourProject/Startup`

(Startup) pkg> add PrecompileTools YourPackages...
```

The `Startup` module should depend on `PrecompileTools` as well as all the packages (`YourPackages...`) that are required to run the precompilation toy problem.
Next, we edit the `Startup/src/Startup.jl` file and add to it all the code which we want PrecompileTools to compile.
We will here provide a basic example featuring Zygote AD code on various algorithmic combinations:

```julia
module Startup

using Random
using TensorKit, KrylovKit, OptimKit
using ChainRulesCore, Zygote
using MPSKit, MPSKitModels
using PEPSKit
using PrecompileTools

@setup_workload begin
    t₀ = time_ns()
    Random.seed!(20918352394)

    # Hyperparameters
    Dbond = 2
    χenv = 4
    gradtol = 1e-3
    maxiter = 4
    verbosity = -1
    H = heisenberg_XYZ(InfiniteSquare())

    # Algorithmic settings
    ctmrg_algs = [
        SimultaneousCTMRG(; maxiter, projector_alg=:halfinfinite, verbosity),
        SequentialCTMRG(; maxiter, projector_alg=:halfinfinite, verbosity),
    ]
    gradient_algs = [
        LinSolver(; solver_alg=BiCGStab(; tol=gradtol), iterscheme=:fixed),
        LinSolver(; solver_alg=BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
        EigSolver(; solver_alg=Arnoldi(; tol=gradtol, eager=true), iterscheme=:fixed),
    ]

    # Initialize OhMyThreads scheduler (precompilation occurs before __init__ call)
    set_scheduler!()

    @compile_workload begin
        # Initialize PEPS and environments with different unit cells, number types and symmetries
        @info "Precompiling workload: initializing PEPSs and environments"
        peps = InfinitePEPS(randn, ComplexF64, ComplexSpace(Dbond), ComplexSpace(Dbond))

        env, = leading_boundary(CTMRGEnv(peps, ComplexSpace(χenv)), peps; verbosity)

        # CTMRG
        @info "Precompiling workload: CTMRG leading_boundary"
        for ctmrg_alg in ctmrg_algs
            leading_boundary(env, peps, ctmrg_alg)
        end

        # Differentiate CTMRG leading_boundary
        @info "Precompiling workload: backpropagation of leading_boundary"
        for alg_rrule in gradient_algs
            Zygote.withgradient(peps) do ψ
                env′, = PEPSKit.hook_pullback(
                    leading_boundary, env, ψ, SimultaneousCTMRG(; verbosity); alg_rrule
                )
                return cost_function(ψ, env′, H)
            end
        end

        # Optimize via fixedpoint using LBFGS
        @info "Precompiling workload: LBFGS fixedpoint optimization"
        fixedpoint(H, peps, env, opt_alg; tol=gradtol, maxiter, verbosity)

        # Compute correlation length
        @info "Precompiling workload: correlation_length"
        correlation_length(peps, env)
    end

    duration = round((time_ns() - t₀) * 1e-9 / 60; digits=2) # minutes
    @info "Precompiling workload: finished after $duration min"
end

end
```

Finally, activate `YourProject` again - where we want to benefit from the shortened execution times - and run `using Startup`.
That way, all packages will be loaded with their precompiled code.
Of course, we may also have multiple start-up routines where the precompiled code is tailored towards the needs of the respective projects.
