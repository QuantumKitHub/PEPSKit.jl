"""
    module Defaults

Module containing default algorithm parameter values and arguments.

## CTMRG

- `ctmrg_tol=$(Defaults.ctmrg_tol)` : Tolerance checking singular value and norm convergence.
- `ctmrg_maxiter=$(Defaults.ctmrg_maxiter)` : Maximal number of CTMRG iterations per run.
- `ctmrg_miniter=$(Defaults.ctmrg_miniter)` : Minimal number of CTMRG carried out.
- `ctmrg_alg=:$(Defaults.ctmrg_alg)` : Default CTMRG algorithm variant.
- `ctmrg_verbosity=$(Defaults.ctmrg_verbosity)` : CTMRG output information verbosity

## SVD forward & reverse

- `trscheme=:$(Defaults.trscheme)` : Truncation scheme for SVDs and other decompositions.
- `svd_fwd_alg=:$(Defaults.svd_fwd_alg)` : SVD algorithm that is used in the forward pass.
- `svd_rrule_tol=$(Defaults.svd_rrule_tol)` : Accuracy of SVD reverse-rule.
- `svd_rrule_min_krylovdim=$(Defaults.svd_rrule_min_krylovdim)` : Minimal Krylov dimension of the reverse-rule algorithm (if it is a Krylov algorithm).
- `svd_rrule_verbosity=$(Defaults.svd_rrule_verbosity)` : SVD gradient output verbosity.
- `svd_rrule_alg=:$(Defaults.svd_rrule_alg)` : Reverse-rule algorithm for the SVD gradient.

## Projectors

- `projector_alg=:$(Defaults.projector_alg)` : Default variant of the CTMRG projector algorithm.
- `projector_verbosity=$(Defaults.projector_verbosity)` : Projector output information verbosity.

## Fixed-point gradient

- `gradient_tol=$(Defaults.gradient_tol)` : Convergence tolerance for the fixed-point gradient iteration.
- `gradient_maxiter=$(Defaults.gradient_maxiter)` : Maximal number of iterations for computing the CTMRG fixed-point gradient.
- `gradient_verbosity=$(Defaults.gradient_verbosity)` : Gradient output information verbosity.
- `gradient_linsolver=:$(Defaults.gradient_linsolver)` : Default linear solver for the `LinSolver` gradient algorithm.
- `gradient_eigsolver=:$(Defaults.gradient_eigsolver)` : Default eigensolver for the `EigSolver` gradient algorithm.
- `gradient_eigsolver_eager=$(Defaults.gradient_eigsolver_eager)` : Enables `EigSolver` algorithm to finish before the full Krylov dimension is reached.
- `gradient_iterscheme=:$(Defaults.gradient_iterscheme)` : Scheme for differentiating one CTMRG iteration.
- `gradient_alg=:$(Defaults.gradient_alg)` : Algorithm variant for computing the gradient fixed-point.

## Optimization

- `reuse_env=$(Defaults.reuse_env)` : If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
- `optimizer_tol=$(Defaults.optimizer_tol)` : Gradient norm tolerance of the optimizer.
- `optimizer_maxiter=$(Defaults.optimizer_maxiter)` : Maximal number of optimization steps.
- `optimizer_verbosity=$(Defaults.optimizer_verbosity)` : Optimizer output information verbosity.
- `optimizer_alg=:$(Defaults.optimizer_alg)` : Default `OptimKit.OptimizerAlgorithm` for PEPS optimization.
- `lbfgs_memory=$(Defaults.lbfgs_memory)` : Size of limited memory representation of BFGS Hessian matrix.

## OhMyThreads scheduler

- `scheduler=Ref{Scheduler}(...)` : Multi-threading scheduler which can be accessed via `set_scheduler!`.
"""
module Defaults

export set_scheduler!

using OhMyThreads

# CTMRG
const ctmrg_tol = 1e-8
const ctmrg_maxiter = 100
const ctmrg_miniter = 4
const ctmrg_alg = :simultaneous # ∈ {:simultaneous, :sequential}
const ctmrg_verbosity = 2
const sparse = false # TODO: implement sparse CTMRG

# SVD forward & reverse
const trscheme = :fixedspace # ∈ {:fixedspace, :notrunc, :truncerr, :truncspace, :truncbelow}
const svd_fwd_alg = :sdd # ∈ {:sdd, :svd, :iterative}
const svd_rrule_tol = ctmrg_tol
const svd_rrule_min_krylovdim = 48
const svd_rrule_verbosity = -1
const svd_rrule_alg = :arnoldi # ∈ {:gmres, :bicgstab, :arnoldi}
const krylovdim_factor = 1.4

# Projectors
const projector_alg = :halfinfinite # ∈ {:halfinfinite, :fullinfinite}
const projector_verbosity = 0

# Fixed-point gradient
const gradient_tol = 1e-6
const gradient_maxiter = 30
const gradient_verbosity = -1
const gradient_linsolver = :bicgstab # ∈ {:gmres, :bicgstab}
const gradient_eigsolver = :arnoldi
const gradient_eigsolver_eager = true
const gradient_iterscheme = :fixed # ∈ {:fixed, :diffgauge}
const gradient_alg = :linsolver # ∈ {:geomsum, :manualiter, :linsolver, :eigsolver}

# Optimization
const reuse_env = true
const optimizer_tol = 1e-4
const optimizer_maxiter = 100
const optimizer_verbosity = 3
const optimizer_alg = :lbfgs
const lbfgs_memory = 20

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
        scheduler[] = OhMyThreads.Implementation._scheduler_from_userinput(sc; kwargs...)
    end
    return nothing
end

function __init__()
    return set_scheduler!()
end

end
