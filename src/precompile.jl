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
    H_spin = heisenberg_XYZ(InfiniteSquare())
    H_unitcell = heisenberg_XYZ(InfiniteSquare(2, 2))
    H_fermion = pwave_superconductor(InfiniteSquare())

    # Algorithmic settings
    ctmrg_algs = [
        SimultaneousCTMRG(; maxiter, projector_alg=HalfInfiniteProjector, verbosity=-1),
        SimultaneousCTMRG(; maxiter, projector_alg=FullInfiniteProjector, verbosity=-1),
        SequentialCTMRG(; maxiter, projector_alg=HalfInfiniteProjector, verbosity=-1),
        SequentialCTMRG(; maxiter, projector_alg=FullInfiniteProjector, verbosity=-1),
    ]
    vumps_alg = VUMPS(;
        maxiter,
        alg_eigsolve=MPSKit.Defaults.alg_eigsolve(; ishermitian=false),
        verbosity=-4,
    )
    gradient_algs = [
        LinSolver(; solver=BiCGStab(; tol=gradtol), iterscheme=:fixed),
        LinSolver(; solver=BiCGStab(; tol=gradtol), iterscheme=:diffgauge),
        GeomSum(; tol=gradtol, iterscheme=:fixed),
        ManualIter(; tol=gradtol, iterscheme=:fixed),
        EigSolver(; solver=Arnoldi(; tol=gradtol, eager=true), iterscheme=:fixed),
    ]
    ctmrg_alg_general = SimultaneousCTMRG(; verbosity=-1)
    opt_alg = PEPSOptimize(; boundary_alg=ctmrg_alg_general, optimizer=LBFGS(4; maxiter))

    # Initialize OhMyThreads scheduler (precompilation occurs before __init__ call)
    set_scheduler!()

    @compile_workload begin
        # Initialize PEPS and environments with different unit cells, number types and symmetries
        @info "Precompilation: initializing PEPSs and environments"
        peps_complex = InfinitePEPS(
            randn, ComplexF64, ComplexSpace(Dbond), ComplexSpace(Dbond)
        )
        peps_real_unitcell = InfinitePEPS(
            randn, Float64, ComplexSpace(Dbond), ComplexSpace(Dbond); unitcell=(2, 2)
        )
        peps_fermion = InfinitePEPS(
            Vect[FermionParity](0 => 1, 1 => 1),
            Vect[FermionParity](0 => Dbond / 2, 1 => Dbond / 2),
        )

        env_complex, = leading_boundary(
            CTMRGEnv(peps_complex, ComplexSpace(χenv)), peps_complex, ctmrg_alg_general
        )
        env_real_unitcell, = leading_boundary(
            CTMRGEnv(rand, Float64, peps_real_unitcell, ComplexSpace(χenv)),
            peps_real_unitcell,
            ctmrg_alg_general,
        )
        env_fermion, = leading_boundary(
            CTMRGEnv(peps_fermion, Vect[FermionParity](0 => χenv / 2, 1 => χenv / 2)),
            peps_fermion,
            ctmrg_alg_general,
        )

        # CTMRG
        @info "Precompilation: CTMRG leading_boundary"
        for ctmrg_alg in ctmrg_algs
            leading_boundary(env_complex, peps_complex, ctmrg_alg)
            leading_boundary(env_real_unitcell, peps_real_unitcell, ctmrg_alg)
            leading_boundary(env_fermion, peps_fermion, ctmrg_alg)
        end

        # Boundary MPS
        @info "Precompilation: VUMPS leading_boundary"
        T_single = InfiniteTransferPEPS(peps_complex, 1, 1)
        mps_single = initializeMPS(T_single, [ComplexSpace(χenv)])
        leading_boundary(mps_single, T_single, vumps_alg)

        T_multi = MultilineTransferPEPS(peps_complex, 1)
        mps_multi = initializeMPS(T_multi, fill(ComplexSpace(χenv), 2, 2))
        leading_boundary(mps_multi, T_multi, vumps_alg)

        # Differentiate CTMRG leading_boundary
        @info "Precompilation: backpropagation of leading_boundary"
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

        Zygote.withgradient(peps_fermion) do ψ
            env′, = hook_pullback(
                leading_boundary,
                env_fermion,
                ψ,
                ctmrg_alg_general;
                alg_rrule=gradient_algs[1],
            )
            return cost_function(ψ, env′, H_fermion)
        end

        # Optimize via fixedpoint using LBFGS
        @info "Precompilation: LBFGS fixedpoint optimization"
        fixedpoint(H_spin, peps_complex, env_complex, opt_alg)

        # Compute correlation length
        @info "Precompilation: correlation_length"
        correlation_length(peps_complex, env_complex)
    end

    duration = (time_ns() - t₀) * 1e-9 / 60 # minutes
    @info "Precompilation: finished after $duration min"
end
