"""
    module Defaults

Module containing default algorithm parameter values and arguments.

## CTMRG

* `ctmrg_tol=$(Defaults.ctmrg_tol)` : Tolerance checking singular value and norm convergence.
* `ctmrg_maxiter=$(Defaults.ctmrg_maxiter)` : Maximal number of CTMRG iterations per run.
* `ctmrg_miniter=$(Defaults.ctmrg_miniter)` : Minimal number of CTMRG carried out.
* `ctmrg_alg=:$(Defaults.ctmrg_alg)` : Default CTMRG algorithm variant.
    - `:SimultaneousCTMRG` : Simultaneous expansion and renormalization of all sides.
    - `:SequentialCTMRG` : Sequential application of left moves and rotations.
* `ctmrg_verbosity=$(Defaults.ctmrg_verbosity)` : CTMRG output information verbosity

## SVD forward & reverse

* `trunc=:$(Defaults.trunc)` : Truncation scheme for SVDs and other decompositions.
    - `:FixedSpaceTruncation` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`

* `rrule_degeneracy_atol=$(Defaults.rrule_degeneracy_atol)` : Broadening amplitude which
  smoothens the divergent term in the retained contributions of an SVD or eigh pullback, in
  case of (pseudo) degenerate singular values
* `svd_fwd_alg=:$(Defaults.svd_fwd_alg)` : SVD algorithm that is used in the forward pass.
    - `:DefaultAlgorithm` : MatrixAlgebraKit's default SVD algorithm for a given matrix type.
    - `:DivideAndConquer` : MatrixAlgebraKit's [`DivideAndConquer`](@extref MatrixAlgebraKit.DivideAndConquer)
    - `:QRIteration` : MatrixAlgebraKit's [`QRIteration`](@extref MatrixAlgebraKit.QRIteration)
    - `:Bisection` : MatrixAlgebraKit's [`Bisection`](@extref MatrixAlgebraKit.Bisection)
    - `:Jacobi` : MatrixAlgebraKit's [`Jacobi`](@extref MatrixAlgebraKit.Jacobi)
    - `:SVDViaPolar` : MatrixAlgebraKit's [`SVDViaPolar`](@extref MatrixAlgebraKit.SVDViaPolar)
    - `:SafeDivideAndConquer` : MatrixAlgebraKit's [`SafeDivideAndConquer`](@extref MatrixAlgebraKit.SafeDivideAndConquer)
    - `:GKL` : Iterative Krylov-based SVD only computing the specifed number of
        singular values and vectors, see [`IterSVD`](@ref PEPSKit.IterSVD) for details.
* `svd_rrule_tol=$(Defaults.svd_rrule_tol)` : Accuracy of SVD reverse-rule.
* `svd_rrule_min_krylovdim=$(Defaults.svd_rrule_min_krylovdim)` : Minimal Krylov dimension of the reverse-rule algorithm (if it is a Krylov algorithm).
* `svd_rrule_verbosity=$(Defaults.svd_rrule_verbosity)` : SVD gradient output verbosity.
* `svd_rrule_alg=:$(Defaults.svd_rrule_alg)` : Reverse-rule algorithm for the SVD gradient.
    - `:FullPullback` : MatrixAlgebraKit's [`svd_pullback!`](@extref MatrixAlgebraKit.svd_pullback!) that requires access to the full spectrum
    - `:TruncPullback` : MatrixAlgebraKit's [`svd_trunc_pullback!`](@extref MatrixAlgebraKit.svd_trunc_pullback!) solving a Sylvester equation on the truncated subspace
    - `:GMRES` : GMRES iterative linear solver, see [`KrylovKit.GMRES`](@extref)
    - `:BiCGStab` : BiCGStab iterative linear solver, see [`KrylovKit.BiCGStab`](@extref)
    - `:Arnoldi` : Arnoldi Krylov algorithm, see the [`KrylovKit.Arnoldi`](@extref)

## `eigh` forward & reverse

* `eigh_fwd_alg=:$(Defaults.eigh_fwd_alg)` : `eigh` algorithm that is used in the forward pass.
    - `:DefaultAlgorithm` : MatrixAlgebraKit's default Eigh algorithm for a given matrix type.
    - `:DivideAndConquer` : MatrixAlgebraKit's [`DivideAndConquer`](@extref MatrixAlgebraKit.DivideAndConquer)
    - `:QRIteration` : MatrixAlgebraKit's [`QRIteration`](@extref MatrixAlgebraKit.QRIteration)
    - `:Bisection` : MatrixAlgebraKit's [`Bisection`](@extref MatrixAlgebraKit.Bisection)
    - `:Jacobi` : MatrixAlgebraKit's [`Jacobi`](@extref MatrixAlgebraKit.Jacobi)
    - `:RobustRepresentations` : MatrixAlgebraKit's [`RobustRepresentations`](@extref MatrixAlgebraKit.RobustRepresentations)
    - `:Lanczos` : Lanczos algorithm for symmetric/Hermitian matrices, see [`KrylovKit.Lanczos`](@extref)
    - `:BlockLanczos` : Block version of `:Lanczos` for repeated extremal eigenvalues, see [`KrylovKit.BlockLanczos`](@extref)
* `eigh_rrule_alg=:$(Defaults.eigh_rrule_alg)` : Reverse-rule algorithm for the `eigh` gradient.
    - `:FullPullback` : MatrixAlgebraKit's [`eigh_pullback!`](@extref MatrixAlgebraKit.eigh_pullback!) that requires access to the full spectrum
    - `:TruncPullback` : MatrixAlgebraKit's [`eigh_trunc_pullback!`](@extref MatrixAlgebraKit.eigh_trunc_pullback!) solving a Sylvester equation on the truncated subspace
* `eigh_rrule_verbosity=$(Defaults.eigh_rrule_verbosity)` : eigh gradient output verbosity.

## Projectors

* `projector_alg=:$(Defaults.projector_alg)` : Default variant of the CTMRG projector algorithm.
    - `:HalfInfiniteProjector` : Projection via SVDs of half-infinite (two enlarged corners) CTMRG environments.
    - `:FullInfiniteProjector` : Projection via SVDs of full-infinite (all four enlarged corners) CTMRG environments.
* `projector_verbosity=$(Defaults.projector_verbosity)` : Projector output information verbosity.
* `projector_alg_c4v=:$(Defaults.projector_alg_c4v)` : Default variant of the C4v CTMRG projector algorithm.
    - `:C4vEighProjector` : Projection via truncated Eigh of an enlarged corner.
    - `:C4vQRProjector` : Projection via QR decomposition of a column-enlarged corner.

## Fixed-point gradient

* `gradient_alg=:$(Defaults.gradient_alg)` : Algorithm variant for computing the implicit gradient of the contraction routine.
* `gradient_tol=$(Defaults.gradient_tol)` : Convergence tolerance for the gradient algorithm.
* `gradient_maxiter=$(Defaults.gradient_maxiter)` : Maximal number of iterations for the gradient computation.
* `gradient_verbosity=$(Defaults.gradient_verbosity)` : Gradient output information verbosity.
* `gradient_fixedpoint_solver_alg=:$(Defaults.gradient_fixedpoint_solver_alg)` : Default solver algorithm for the `FixedPointGradient` gradient algorithm.
    - `:GMRES` : GMRES iterative linear solver, see [`KrylovKit.GMRES`](@extref) for details
    - `:BiCGStab` : BiCGStab iterative linear solver, see [`KrylovKit.BiCGStab`](@extref) for details
    - `:Arnoldi` : Arnoldi Krylov algorithm, see [`KrylovKit.Arnoldi`](@extref) for details
    - `:GeomSum` : Geometric sum approximation of the Neumann series of the inverse Jacobian, see [`PEPSKit.GeomSum`](@ref) for details
    - `:ManualIter` : Manual fixed-point iteration, see [`PEPSKit.ManualIter`](@ref) for details
* `gradient_fixedpoint_solver_eager=$(Defaults.gradient_fixedpoint_solver_eager)` : Enables `:Arnoldi` solver algorithm to finish before the full Krylov dimension is reached.

## Optimization

* `reuse_env=$(Defaults.reuse_env)` : If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `optimizer_tol=$(Defaults.optimizer_tol)` : Gradient norm tolerance of the optimizer.
* `optimizer_maxiter=$(Defaults.optimizer_maxiter)` : Maximal number of optimization steps.
* `optimizer_verbosity=$(Defaults.optimizer_verbosity)` : Optimizer output information verbosity.
* `optimizer_alg=:$(Defaults.optimizer_alg)` : Default `OptimKit.OptimizerAlgorithm` for PEPS optimization.
    - `:GradientDescent` : Gradient descent algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:ConjugateGradient` : Conjugate gradient algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:LBFGS` : L-BFGS algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
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
const ctmrg_alg = :SimultaneousCTMRG # ∈ {:SimultaneousCTMRG, :SequentialCTMRG}
const ctmrg_verbosity = 2
const sparse = false # TODO: implement sparse CTMRG

# SVD forward & reverse
const trunc = :FixedSpaceTruncation # ∈ {:FixedSpaceTruncation, :notrunc, :truncerror, :truncspace, :trunctol}
const rrule_degeneracy_atol = 1.0e-13
const svd_fwd_alg = :DefaultAlgorithm # ∈ {:<MatrixAlgebraKit.SVDAlgorithms>, :GKL}
const svd_rrule_tol = ctmrg_tol
const svd_rrule_min_krylovdim = 48
const svd_rrule_verbosity = -1
const svd_rrule_alg = :FullPullback # ∈ {:FullPullback, :TruncPullback, :GMRES, :BiCGStab, :Arnoldi}
const krylovdim_factor = 1.4

# eigh forward & reverse
const eigh_fwd_alg = :DefaultAlgorithm # ∈ {:<MatrixAlgebraKit.EighAlgorithms>, :Lanczos, :BlockLanczos}
const eigh_rrule_alg = :FullPullback # ∈ {:FullPullback, :TruncPullback}
const eigh_rrule_verbosity = 0

# QR forward & reverse
const qr_fwd_alg = :DefaultAlgorithm
const qr_fwd_positive = true
const qr_rrule_alg = :FullPullback
const qr_rrule_verbosity = 0

# Projectors
const projector_alg = :HalfInfiniteProjector # ∈ {:HalfInfiniteProjector, :FullInfiniteProjector}
const projector_verbosity = 0
const projector_alg_c4v = :C4vEighProjector # ∈ {:C4vEighProjector, :C4vQRProjector}

# Fixed-point gradient
const gradient_tol = 1.0e-6
const gradient_maxiter = 30
const gradient_verbosity = -1
const gradient_alg = :FixedPointGradient
const gradient_fixedpoint_solver_alg = :Arnoldi # ∈ {:GMRES, :BiCGStab, :Arnoldi, :GeomSum, :ManualIter}
const gradient_fixedpoint_solver_eager = true

# Optimization
const reuse_env = true
const optimizer_tol = 1.0e-4
const optimizer_maxiter = 100
const optimizer_verbosity = 3
const optimizer_alg = :LBFGS # ∈ {:GradientDescent, :ConjugateGradient, :LBFGS}
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
