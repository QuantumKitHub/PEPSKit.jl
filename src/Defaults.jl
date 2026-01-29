"""
    module Defaults

Module containing default algorithm parameter values and arguments.

## CTMRG

* `ctmrg_tol=$(Defaults.ctmrg_tol)` : Tolerance checking singular value and norm convergence.
* `ctmrg_maxiter=$(Defaults.ctmrg_maxiter)` : Maximal number of CTMRG iterations per run.
* `ctmrg_miniter=$(Defaults.ctmrg_miniter)` : Minimal number of CTMRG carried out.
* `ctmrg_alg=:$(Defaults.ctmrg_alg)` : Default CTMRG algorithm variant.
    - `:simultaneous`: Simultaneous expansion and renormalization of all sides.
    - `:sequential`: Sequential application of left moves and rotations.
* `ctmrg_verbosity=$(Defaults.ctmrg_verbosity)` : CTMRG output information verbosity

## SVD forward & reverse

* `trunc=:$(Defaults.trunc)` : Truncation scheme for SVDs and other decompositions.
    - `:fixedspace` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `svd_fwd_alg=:$(Defaults.svd_fwd_alg)` : SVD algorithm that is used in the forward pass.
    - `:sdd`: MatrixAlgebraKit's `LAPACK_DivideAndConquer`
    - `:svd`: MatrixAlgebraKit's `LAPACK_QRIteration`
    - `:cusvdj`: MatrixAlgebraKit's `CUSOLVER_Jacobi`
    - `:cusvd`: MatrixAlgebraKit's `CUSOLVER_QRIteration`
    - `:iterative`: Iterative SVD only computing the specifed number of singular values and vectors, see [`IterSVD`](@ref PEPSKit.IterSVD)
* `svd_rrule_tol=$(Defaults.svd_rrule_tol)` : Accuracy of SVD reverse-rule.
* `svd_rrule_min_krylovdim=$(Defaults.svd_rrule_min_krylovdim)` : Minimal Krylov dimension of the reverse-rule algorithm (if it is a Krylov algorithm).
* `svd_rrule_verbosity=$(Defaults.svd_rrule_verbosity)` : SVD gradient output verbosity.
* `svd_rrule_alg=:$(Defaults.svd_rrule_alg)` : Reverse-rule algorithm for the SVD gradient.
    - `:full`: Uses a modified version of MatrixAlgebraKit's reverse-rule for `svd_compact` which doesn't solve any linear problem and instead requires access to the full SVD, see [`PEPSKit.FullSVDReverseRule`](@ref).
    - `:gmres`: GMRES iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.GMRES) for details
    - `:bicgstab`: BiCGStab iterative linear solver, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BiCGStab) for details
    - `:arnoldi`: Arnoldi Krylov algorithm, see the [KrylovKit docs](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.Arnoldi) for details
* `svd_rrule_broadening=$(Defaults.svd_rrule_broadening)` : Lorentzian broadening amplitude which smoothens the divergent term in the SVD adjoint in case of (pseudo) degenerate singular values

## Projectors

* `projector_alg=:$(Defaults.projector_alg)` : Default variant of the CTMRG projector algorithm.
    - `:halfinfinite`: Projection via SVDs of half-infinite (two enlarged corners) CTMRG environments.
    - `:fullinfinite`: Projection via SVDs of full-infinite (all four enlarged corners) CTMRG environments.
* `projector_verbosity=$(Defaults.projector_verbosity)` : Projector output information verbosity.

## Fixed-point gradient

* `gradient_tol=$(Defaults.gradient_tol)` : Convergence tolerance for the fixed-point gradient iteration.
* `gradient_maxiter=$(Defaults.gradient_maxiter)` : Maximal number of iterations for computing the CTMRG fixed-point gradient.
* `gradient_verbosity=$(Defaults.gradient_verbosity)` : Gradient output information verbosity.
* `gradient_linsolver=:$(Defaults.gradient_linsolver)` : Default linear solver for the `LinSolver` gradient algorithm.
    - `:gmres` : GMRES iterative linear solver, see [`KrylovKit.GMRES`](@extref) for details
    - `:bicgstab` : BiCGStab iterative linear solver, see [`KrylovKit.BiCGStab`](@extref) for details
* `gradient_eigsolver=:$(Defaults.gradient_eigsolver)` : Default eigensolver for the `EigSolver` gradient algorithm.
    - `:arnoldi` : Arnoldi Krylov algorithm, see [`KrylovKit.Arnoldi`](@extref) for details
* `gradient_eigsolver_eager=$(Defaults.gradient_eigsolver_eager)` : Enables `EigSolver` algorithm to finish before the full Krylov dimension is reached.
* `gradient_iterscheme=:$(Defaults.gradient_iterscheme)` : Scheme for differentiating one CTMRG iteration.
    - `:fixed` : the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge` : the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well
* `gradient_alg=:$(Defaults.gradient_alg)` : Algorithm variant for computing the gradient fixed-point.

## Optimization

* `reuse_env=$(Defaults.reuse_env)` : If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `optimizer_tol=$(Defaults.optimizer_tol)` : Gradient norm tolerance of the optimizer.
* `optimizer_maxiter=$(Defaults.optimizer_maxiter)` : Maximal number of optimization steps.
* `optimizer_verbosity=$(Defaults.optimizer_verbosity)` : Optimizer output information verbosity.
* `optimizer_alg=:$(Defaults.optimizer_alg)` : Default `OptimKit.OptimizerAlgorithm` for PEPS optimization.
    - `:gradientdescent` : Gradient descent algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:conjugategradient` : Conjugate gradient algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:lbfgs` : L-BFGS algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
* `ls_maxiter=$(Defaults.ls_maxiter)` : Maximum number of iterations for the line search in each step of the optimization.
* `ls_maxfg=$(Defaults.ls_maxfg)` : Maximum number of function evaluations for the line search in each step of the optimization.
* `lbfgs_memory=$(Defaults.lbfgs_memory)` : Size of limited memory representation of BFGS Hessian matrix.

## OhMyThreads scheduler

- `scheduler=Ref{Scheduler}(...)` : Multithreading scheduler which can be accessed via `set_scheduler!`.
"""
module Defaults

export set_scheduler!

using OhMyThreads

# CTMRG
const ctmrg_tol = 1.0e-8
const ctmrg_maxiter = 100
const ctmrg_miniter = 4
const ctmrg_alg = :simultaneous # ∈ {:simultaneous, :sequential}
const ctmrg_verbosity = 2
const sparse = false # TODO: implement sparse CTMRG

# SVD forward & reverse
const trunc = :fixedspace # ∈ {:fixedspace, :notrunc, :truncerror, :truncspace, :trunctol}
const svd_fwd_alg = :sdd # ∈ {:sdd, :svd, :iterative}
const svd_rrule_tol = ctmrg_tol
const svd_rrule_min_krylovdim = 48
const svd_rrule_verbosity = -1
const svd_rrule_alg = :full # ∈ {:full, :gmres, :bicgstab, :arnoldi}
const svd_rrule_broadening = 1.0e-13
const krylovdim_factor = 1.4

# eigh forward & reverse
const eigh_fwd_alg = :qriteration # ∈ {:qriteration, :bisection, :divideandconquer, :multiple, :lanczos, :blocklanczos}
const eigh_rrule_alg = :trunc # ∈ {:trunc, :full}
const eigh_rrule_verbosity = 0

# QR forward & reverse
# const qr_fwd_alg = :something # TODO
# const qr_rrule_alg = :something
# const qr_rrule_verbosity = :something

# Projectors
const projector_alg = :halfinfinite # ∈ {:halfinfinite, :fullinfinite}
const projector_verbosity = 0
const projector_alg_c4v = :c4v_eigh # ∈ {:c4v_eigh, :c4v_qr (TODO)}

# Fixed-point gradient
const gradient_tol = 1.0e-6
const gradient_maxiter = 30
const gradient_verbosity = -1
const gradient_linsolver = :bicgstab # ∈ {:gmres, :bicgstab}
const gradient_eigsolver = :arnoldi
const gradient_eigsolver_eager = true
const gradient_iterscheme = :fixed # ∈ {:fixed, :diffgauge}
const gradient_alg = :eigsolver # ∈ {:geomsum, :manualiter, :linsolver, :eigsolver}

# Optimization
const reuse_env = true
const optimizer_tol = 1.0e-4
const optimizer_maxiter = 100
const optimizer_verbosity = 3
const optimizer_alg = :lbfgs # ∈ {:gradientdescent, :conjugategradient, :lbfgs}
const ls_maxiter = 10
const ls_maxfg = 20
const lbfgs_memory = 20

# OhMyThreads scheduler defaults
const scheduler = Ref{Scheduler}()

"""
    set_scheduler!([scheduler]; kwargs...)

Set `OhMyThreads` multithreading scheduler parameters.

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
[OhMyThreads](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#OhMyThreads.Schedulers.Scheduler) documentation.

If no `scheduler` is passed and only kwargs are provided, the `DynamicScheduler`
constructor is used with the provided kwargs.

To reset the scheduler to its default value, one calls `set_scheduler!` without passing
arguments which then uses the default `DynamicScheduler()`. If the number of used threads is
just one it falls back to `SerialScheduler()`.
"""
function set_scheduler!(sc = OhMyThreads.Implementation.NotGiven(); kwargs...)
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
