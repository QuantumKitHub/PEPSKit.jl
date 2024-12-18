module PEPSKit

using LinearAlgebra, Statistics, Base.Threads, Base.Iterators, Printf
using Base: @kwdef
using Compat
using Accessors: @set
using VectorInterface
using TensorKit, KrylovKit, MPSKit, TensorOperations
using TensorKit: ℂ, ℝ  # To avoid conflict with Manifolds
using ChainRulesCore, Zygote
using LoggingExtras
using MPSKit: loginit!, logiter!, logfinish!, logcancel!
using MPSKitModels
using FiniteDifferences
using OhMyThreads: tmap

include("utility/util.jl")
include("utility/diffable_threads.jl")
include("utility/svd.jl")
include("utility/rotations.jl")
include("utility/mirror.jl")
include("utility/diffset.jl")
include("utility/hook_pullback.jl")
include("utility/autoopt.jl")

include("states/abstractpeps.jl")
include("states/infinitepeps.jl")
include("states/infiniteweightpeps.jl")

include("operators/transferpeps.jl")
include("operators/infinitepepo.jl")
include("operators/transferpepo.jl")
include("operators/derivatives.jl")
include("operators/localoperator.jl")
include("operators/lattices/squarelattice.jl")
include("operators/models.jl")

include("environments/ctmrg_environments.jl")
include("environments/transferpeps_environments.jl")
include("environments/transferpepo_environments.jl")

include("algorithms/contractions/localoperator.jl")
include("algorithms/contractions/ctmrg_contractions.jl")

include("algorithms/ctmrg/sparse_environments.jl")
include("algorithms/ctmrg/ctmrg.jl")
include("algorithms/ctmrg/projectors.jl")
include("algorithms/ctmrg/simultaneous.jl")
include("algorithms/ctmrg/sequential.jl")
include("algorithms/ctmrg/gaugefix.jl")

include("algorithms/time_evolution/gatetools.jl")
include("algorithms/time_evolution/simpleupdate.jl")

include("algorithms/toolbox.jl")

include("utility/symmetrization.jl")

include("algorithms/peps_opt.jl")

"""
    module Defaults
        # CTMRG
        const ctmrg_tol = 1e-8
        const ctmrg_maxiter = 100
        const ctmrg_miniter = 4
        const sparse = false
        const trscheme = FixedSpaceTruncation()
        const fwd_alg = TensorKit.SDD()
        const rrule_alg = Arnoldi(; tol=ctmrg_tol, krylovdim=48, verbosity=-1)
        const svd_alg = SVDAdjoint(; fwd_alg, rrule_alg)
        const projector_alg_type = HalfInfiniteProjector
        const projector_alg = projector_alg_type(svd_alg, trscheme, 2)
        const ctmrg_alg = SimultaneousCTMRG(
            ctmrg_tol, ctmrg_maxiter, ctmrg_miniter, 2, projector_alg
        )

        # Optimization
        const optim_alg = quasi_Newton
        const record_group = [
            RecordCost(),
            RecordGradientNorm(),
            RecordConditionNumber(),
            RecordCostUnitCell(),
            RecordTime(),
        ]
        
        const optim_kwargs = (;
            memory_size=32,
            stopping_criterion=StopAfterIteration(100) | StopWhenGradientNormLess(1e-4),
            record=record_group,
            return_state=true,
        )
        const fpgrad_maxiter = 30
        const fpgrad_tol = 1e-6
        const gradient_linsolver = KrylovKit.BiCGStab(; maxiter=fpgrad_maxiter, tol=fpgrad_tol)
        const iterscheme = :fixed
        const reuse_env = true
        const gradient_alg = LinSolver(; solver=gradient_linsolver, iterscheme)

        # OhMyThreads scheduler defaults
        const scheduler = Ref{Scheduler}(Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler())
    end

Module containing default values that represent typical algorithm parameters.

- `ctmrg_maxiter`: Maximal number of CTMRG iterations per run
- `ctmrg_miniter`: Minimal number of CTMRG carried out
- `ctmrg_tol`: Tolerance checking singular value and norm convergence
- `trscheme`: Truncation scheme for SVDs and other decompositions
- `fwd_alg`: SVD algorithm that is used in the forward pass
- `rrule_alg`: Reverse-rule for differentiating that SVD
- `svd_alg`: Combination of `fwd_alg` and `rrule_alg`
- `projector_alg_type`: Default type of projector algorithm
- `projector_alg`: Algorithm to compute CTMRG projectors
- `ctmrg_alg`: Algorithm for performing CTMRG runs
- `fpgrad_maxiter`: Maximal number of iterations for computing the CTMRG fixed-point gradient
- `fpgrad_tol`: Convergence tolerance for the fixed-point gradient iteration
- `reuse_env`: If `true`, the current optimization step is initialized on the previous environment
- `gradient_linsolver`: Default linear solver for the `LinSolver` gradient algorithm
- `iterscheme`: Scheme for differentiating one CTMRG iteration
- `gradient_alg`: Algorithm to compute the gradient fixed-point
# TODO
- `scheduler`: Multi-threading scheduler which can be accessed via `set_scheduler!`
"""
module Defaults
    using TensorKit, KrylovKit, OhMyThreads
    using Manopt
    using PEPSKit:
        LinSolver,
        FixedSpaceTruncation,
        SVDAdjoint,
        HalfInfiniteProjector,
        SimultaneousCTMRG,
        RecordConditionNumber,
        RecordUnitCellGradientNorm

    # CTMRG
    const ctmrg_tol = 1e-8
    const ctmrg_maxiter = 100
    const ctmrg_miniter = 4
    const sparse = false
    const trscheme = FixedSpaceTruncation()
    const fwd_alg = TensorKit.SDD()
    const rrule_alg = Arnoldi(; tol=ctmrg_tol, krylovdim=48, verbosity=-1)
    const svd_alg = SVDAdjoint(; fwd_alg, rrule_alg)
    const projector_alg_type = HalfInfiniteProjector
    const projector_alg = projector_alg_type(svd_alg, trscheme, 2)
    const ctmrg_alg = SimultaneousCTMRG(
        ctmrg_tol, ctmrg_maxiter, ctmrg_miniter, 2, projector_alg
    )

    # Optimization
    const optim_alg = quasi_Newton
    const record_group = [
        RecordCost() => :cost,
        RecordGradientNorm() => :gradient_norm,
        RecordConditionNumber() => :condition, # TODO: implement PEPS record actions
        RecordUnitCellGradientNorm() => :unitcell_gradient_norm,
        RecordTime() => :time,
    ]
    const debug_group = [
        (:Iteration, "Optim %-5d"),
        (:Cost, "f(x) = %.8f"),
        (:GradientNorm, "   ‖∂f‖ = %.8f   "),
        (:Stepsize, "   step size = %.8f   "),
        DebugTime(; prefix="time =", mode=:iterative),
        DebugWarnIfCostIncreases(:Always; tol=1e-12),
        :Stop,
        "\n",
    ]
    const stopping_criterion = StopAfterIteration(100) | StopWhenGradientNormLess(1e-4)
    const optim_maxiter = 100
    const optim_tol = 1e-4
    const optim_kwargs = (;
        stopping_criterion=StopAfterIteration(optim_maxiter) |
                           StopWhenGradientNormLess(optim_tol),
        record=record_group,
        debug=debug_group,
    )
    const fpgrad_maxiter = 30
    const fpgrad_tol = 1e-6
    const gradient_linsolver = KrylovKit.BiCGStab(; maxiter=fpgrad_maxiter, tol=fpgrad_tol)
    const iterscheme = :fixed
    const reuse_env = true
    const gradient_alg = LinSolver(; solver=gradient_linsolver, iterscheme)

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

using .Defaults: set_scheduler!
export set_scheduler!
export SVDAdjoint, IterSVD, NonTruncSVDAdjoint
export CTMRGEnv, SequentialCTMRG, SimultaneousCTMRG
export FixedSpaceTruncation, HalfInfiniteProjector, FullInfiniteProjector
export LocalOperator
export expectation_value, costfun, product_peps, correlation_length
export leading_boundary
export PEPSOptimize, GeomSum, ManualIter, LinSolver
export fixedpoint

export absorb_weight
export su_iter, simpleupdate, SimpleUpdate

export InfinitePEPS, InfiniteTransferPEPS
export SUWeight, InfiniteWeightPEPS
export InfinitePEPO, InfiniteTransferPEPO
export initializeMPS, initializePEPS
export ReflectDepth, ReflectWidth, Rotate, RotateReflect
export symmetrize!, symmetrize_retract_and_finalize!
export showtypeofgrad
export InfiniteSquare, vertices, nearest_neighbours, next_nearest_neighbours
export transverse_field_ising, heisenberg_XYZ, j1_j2
export pwave_superconductor, hubbard_model, tj_model

end # module
