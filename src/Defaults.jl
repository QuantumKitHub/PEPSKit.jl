"""
    module Defaults

Module containing default algorithm parameter values and arguments.

# CTMRG
- `ctmrg_tol=1e-8`: Tolerance checking singular value and norm convergence
- `ctmrg_maxiter=100`: Maximal number of CTMRG iterations per run
- `ctmrg_miniter=4`: Minimal number of CTMRG carried out
- `ctmrg_alg_type=SimultaneousCTMRG`: Default CTMRG algorithm variant
- `ctmrg_verbosity=2`: CTMRG output information verbosity
- `trscheme=FixedSpaceTruncation()`: Truncation scheme for SVDs and other decompositions
- `svd_fwd_alg=TensorKit.SDD()`: SVD algorithm that is used in the forward pass
- `svd_rrule_type = Arnoldi`: Default solver type for SVD reverse-rule algorithm
- `svd_rrule_alg`: Reverse-rule algorithm for differentiating a SVD

    ```
    svd_rrule_alg = svd_rrule_type(; tol=ctmrg_tol, krylovdim=48, verbosity=-1)
    ```

- `svd_alg`: Combination of forward and reverse SVD algorithms

    ```
    svd_alg=SVDAdjoint(; fwd_alg=svd_fwd_alg, rrule_alg=svd_rrule_alg)
    ```

- `projector_alg_type=HalfInfiniteProjector`: Default type of projector algorithm
- `projector_alg`: Algorithm to compute CTMRG projectors

    ```
    projector_alg = projector_alg_type(; svd_alg, trscheme, verbosity=0)
    ```

- `ctmrg_alg`: Algorithm for performing CTMRG runs

    ```
    ctmrg_alg = ctmrg_alg_type(
        ctmrg_tol, ctmrg_maxiter, ctmrg_miniter, 2, projector_alg
    )
    ```

# Optimization
- `gradient_alg_tol=1e-6`: Convergence tolerance for the fixed-point gradient iteration
- `gradient_alg_maxiter=30`: Maximal number of iterations for computing the CTMRG fixed-point gradient
- `gradient_alg_iterscheme=:fixed`: Scheme for differentiating one CTMRG iteration
- `gradient_linsolver`: Default linear solver for the `LinSolver` gradient algorithm

    ```
    gradient_linsolver=KrylovKit.BiCGStab(; maxiter=gradient_alg_maxiter, tol=gradient_alg_tol)
    ```

- `gradient_eigsolver`: Default eigsolver for the `EigSolver` gradient algorithm

    ```
    gradient_eigsolver = KrylovKit.Arnoldi(; maxiter=gradient_alg_maxiter, tol=gradient_alg_tol, eager=true)
    ```

- `gradient_alg`: Algorithm to compute the gradient fixed-point

    ```
    gradient_alg = LinSolver(; solver=gradient_linsolver, iterscheme=gradient_alg_iterscheme)
    ```

- `reuse_env=true`: If `true`, the current optimization step is initialized on the previous
  environment, otherwise a random environment is used
- `optimizer_tol=1e-4`: Gradient norm tolerance of the optimizer
- `optimizer_maxiter=100`: Maximal number of optimization steps
- `lbfgs_memory=20`: Size of limited memory representation of BFGS Hessian matrix
- `optimizer`: Default `OptimKit.OptimizerAlgorithm` for PEPS optimization

    ```
    optimizer=LBFGS(lbfgs_memory; maxiter=optimizer_maxiter, gradtol=optimizer_tol, verbosity=3)
    ```

# OhMyThreads scheduler
- `scheduler=Ref{Scheduler}(...)`: Multi-threading scheduler which can be accessed via `set_scheduler!`
"""
module Defaults
    # CTMRG
    const ctmrg_tol = 1e-8
    const ctmrg_maxiter = 100
    const ctmrg_miniter = 4
    const ctmrg_alg = :simultaneous # ∈ {:simultaneous, :sequential}
    const ctmrg_verbosity = 2
    const sparse = false # TODO: implement sparse CTMRG

    # SVD forward & reverse
    const trscheme = :fixedspace
    const svd_fwd_alg = :sdd # ∈ {:sdd, :svd, :iterative}
    const svd_rrule_alg = :arnoldi # ∈ {:gmres, :bicgstab, :arnoldi}
    const svd_rrule_verbosity = -1
    const krylovdim_factor = 1.4

    # Projector
    const projector_alg = :halfinfinite # ∈ {:halfinfinite, :fullinfinite}
    const projector_verbosity = 0

    # Fixed-point gradient
    const gradient_alg_tol = 1e-6
    const gradient_alg_maxiter = 30
    const gradient_linsolver = :bicgstab # ∈ {:gmres, :bicgstab}
    const gradient_eigsolver = :arnoldi
    const gradient_eigsolver_eager = true
    const gradient_alg_iterscheme = :fixed # ∈ {:fixed, :diffgauge}
    const gradient_alg = :linsolver # ∈ {:geomsum, :manualiter, :linsolver, :eigsolver}

    # Optimization
    const reuse_env = true
    const optimizer_tol = 1e-4
    const optimizer_maxiter = 100
    const lbfgs_memory = 20
    const optimizer_verbosity = 3

    # OhMyThreads scheduler defaults
    const scheduler = Ref{Scheduler}()
    """
        set_scheduler!([scheduler]; kwargs...)

    Set `OhMyThreads` multi-threading scheduler parameters.

    The function either accepts a `scheduler` as an `OhMyThreads.Scheduler` or
    as a symbol where the corresponding parameters are specificed as keyword arguments.
    For instance, a static scheduler that uses four tasks with chunking enabled
    can be set via
    ```
    set_scheduler!(StaticScheduler(; ntasks=4, chunking=true))
    ```
    or equivalently with 
    ```
    set_scheduler!(:static; ntasks=4, chunking=true)
    ```
    For a detailed description of all schedulers and their keyword arguments consult the
    [`OhMyThreads` documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).

    If no `scheduler` is passed and only kwargs are provided, the `DynamicScheduler`
    constructor is used with the provided kwargs.

    To reset the scheduler to its default value, one calls `set_scheduler!` without passing
    arguments which then uses the default `DynamicScheduler()`. If the number of used threads is
    just one it falls back to `SerialScheduler()`.
    """
    function set_scheduler!(sc=OhMyThreads.Implementation.NotGiven(); kwargs...)
        if isempty(kwargs) && sc isa OhMyThreads.Implementation.NotGiven
            scheduler[] = Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler()
        else
            scheduler[] = OhMyThreads.Implementation._scheduler_from_userinput(
                sc; kwargs...
            )
        end
        return nothing
    end
    export set_scheduler!

    function __init__()
        return set_scheduler!()
    end
end
