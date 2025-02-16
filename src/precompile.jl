using PrecompileTools: @setup_workload, @compile_workload
using Random

@setup_workload begin
    t₀ = time_ns()
    Random.seed!(20918352394)

    # Hyperparameters
    Dbond = 2
    χenv = 4
    gradtol = 1e-3
    maxiter = 4
    verbosity = -1
    H_spin = heisenberg_XYZ(InfiniteSquare())
    H_unitcell = heisenberg_XYZ(InfiniteSquare(2, 2))

    # Algorithmic settings
    ctmrg_algs = [
        SimultaneousCTMRG(; maxiter, projector_alg=HalfInfiniteProjector, verbosity),
        SimultaneousCTMRG(; maxiter, projector_alg=FullInfiniteProjector, verbosity),
        SequentialCTMRG(; maxiter, projector_alg=HalfInfiniteProjector, verbosity),
        SequentialCTMRG(; maxiter, projector_alg=FullInfiniteProjector, verbosity),
    ]
    vumps_alg = VUMPS(;
        maxiter, alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false), verbosity
    )
    gradient_algs = [
        LinSolver(; solver=BiCGStab(; tol=gradtol), iterscheme=:fixed),
        LinSolver(; solver=BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
        GeomSum(; tol=gradtol, iterscheme=:fixed),
        ManualIter(; tol=gradtol, iterscheme=:fixed),
        EigSolver(; solver=Arnoldi(; tol=gradtol, eager=true), iterscheme=:fixed),
    ]
    ctmrg_alg_general = SimultaneousCTMRG(; verbosity)
    opt_alg = PEPSOptimize(;
        boundary_alg=ctmrg_alg_general, optimizer=LBFGS(4; maxiter, verbosity)
    )

    # Initialize OhMyThreads scheduler (precompilation occurs before __init__ call)
    set_scheduler!()

    @compile_workload begin
        # Initialize PEPS and environments with different unit cells, number types and symmetries
        @info "Precompiling workload: initializing PEPSs and environments"
        peps_complex = InfinitePEPS(
            randn, ComplexF64, ComplexSpace(Dbond), ComplexSpace(Dbond)
        )
        peps_real_unitcell = InfinitePEPS(
            randn, Float64, ComplexSpace(Dbond), ComplexSpace(Dbond); unitcell=(2, 2)
        )

        env_complex, = leading_boundary(
            CTMRGEnv(peps_complex, ComplexSpace(χenv)), peps_complex, ctmrg_alg_general
        )
        env_real_unitcell, = leading_boundary(
            CTMRGEnv(rand, Float64, peps_real_unitcell, ComplexSpace(χenv)),
            peps_real_unitcell,
            ctmrg_alg_general,
        )

        # CTMRG
        @info "Precompiling workload: CTMRG leading_boundary"
        for ctmrg_alg in ctmrg_algs
            leading_boundary(env_complex, peps_complex, ctmrg_alg)
            leading_boundary(env_real_unitcell, peps_real_unitcell, ctmrg_alg)
        end

        # Boundary MPS
        @info "Precompiling workload: VUMPS leading_boundary"
        T_single = InfiniteTransferPEPS(peps_complex, 1, 1)
        mps_single = initializeMPS(T_single, [ComplexSpace(χenv)])
        leading_boundary(mps_single, T_single, vumps_alg)

        T_multi = MultilineTransferPEPS(peps_complex, 1)
        mps_multi = initializeMPS(T_multi, fill(ComplexSpace(χenv), 2, 2))
        leading_boundary(mps_multi, T_multi, vumps_alg)

        # Differentiate CTMRG leading_boundary
        @info "Precompiling workload: backpropagation of leading_boundary"
        for alg_rrule in gradient_algs
            Zygote.withgradient(peps_complex) do ψ
                env′, = hook_pullback(
                    leading_boundary, env_complex, ψ, ctmrg_alg_general; alg_rrule
                )
                return cost_function(ψ, env′, H_spin)
            end
        end

        Zygote.withgradient(peps_real_unitcell) do ψ
            env′, = hook_pullback(
                leading_boundary,
                env_real_unitcell,
                ψ,
                ctmrg_alg_general;
                alg_rrule=gradient_algs[2],
            )
            return cost_function(ψ, env′, H_unitcell)
        end

        # Optimize via fixedpoint using LBFGS
        @info "Precompiling workload: LBFGS fixedpoint optimization"
        fixedpoint(H_spin, peps_complex, env_complex, opt_alg)

        # Compute correlation length
        @info "Precompiling workload: correlation_length"
        correlation_length(peps_complex, env_complex)
    end

    duration = round((time_ns() - t₀) * 1e-9 / 60; digits=2) # minutes
    @info "Precompiling workload: finished after $duration min"
end
