module PEPSKit

using LinearAlgebra, Statistics, Base.Threads, Base.Iterators, Printf
using Compat
using Accessors: @set, @reset
using VectorInterface
using TensorKit, KrylovKit, MPSKit, OptimKit, TensorOperations
using ChainRulesCore, Zygote
using LoggingExtras
import MPSKit: leading_boundary, loginit!, logiter!, logfinish!, logcancel!
using MPSKitModels
using FiniteDifferences
using OhMyThreads: tmap

include("Defaults.jl")  # Include first to allow for docstring interpolation with Defaults values

include("utility/util.jl")
include("utility/diffable_threads.jl")
include("utility/svd.jl")
include("utility/rotations.jl")
include("utility/mirror.jl")
include("utility/hook_pullback.jl")
include("utility/autoopt.jl")
include("utility/retractions.jl")

include("networks/tensors.jl")
include("networks/local_sandwich.jl")
include("networks/infinitesquarenetwork.jl")

include("states/infinitepeps.jl")
include("states/infiniteweightpeps.jl")
include("states/infinitepartitionfunction.jl")

include("operators/infinitepepo.jl")
include("operators/transfermatrix.jl")
include("operators/localoperator.jl")
include("operators/lattices/squarelattice.jl")
include("operators/models.jl")

include("environments/ctmrg_environments.jl")
include("environments/vumps_environments.jl")

include("algorithms/contractions/ctmrg_contractions.jl")
include("algorithms/contractions/localoperator.jl")
include("algorithms/contractions/vumps_contractions.jl")
include("algorithms/contractions/bondenv/benv_tools.jl")
include("algorithms/contractions/bondenv/als_solve.jl")

include("algorithms/ctmrg/sparse_environments.jl")
include("algorithms/ctmrg/ctmrg.jl")
include("algorithms/ctmrg/projectors.jl")
include("algorithms/ctmrg/simultaneous.jl")
include("algorithms/ctmrg/sequential.jl")
include("algorithms/ctmrg/gaugefix.jl")

include("algorithms/truncation/truncationschemes.jl")
include("algorithms/truncation/fullenv_truncation.jl")
include("algorithms/truncation/bond_truncation.jl")

include("algorithms/time_evolution/evoltools.jl")
include("algorithms/time_evolution/simpleupdate.jl")

include("algorithms/toolbox.jl")

include("utility/symmetrization.jl")

include("algorithms/optimization/fixed_point_differentiation.jl")
include("algorithms/optimization/peps_optimization.jl")

include("algorithms/select_algorithm.jl")

using .Defaults: set_scheduler!
export set_scheduler!
export SVDAdjoint, IterSVD
export CTMRGEnv, SequentialCTMRG, SimultaneousCTMRG
export FixedSpaceTruncation, HalfInfiniteProjector, FullInfiniteProjector
export LocalOperator
export expectation_value, cost_function, product_peps, correlation_length, network_value
export leading_boundary
export PEPSOptimize, GeomSum, ManualIter, LinSolver, EigSolver
export fixedpoint

export absorb_weight
export ALSTruncation, FullEnvTruncation
export su_iter, simpleupdate, SimpleUpdate

export InfiniteSquareNetwork
export InfinitePartitionFunction
export InfinitePEPS, InfiniteTransferPEPS
export SUWeight, InfiniteWeightPEPS
export InfinitePEPO, InfiniteTransferPEPO
export initializeMPS, initializePEPS
export ReflectDepth, ReflectWidth, Rotate, RotateReflect
export symmetrize!, symmetrize_retract_and_finalize!
export showtypeofgrad
export InfiniteSquare, vertices, nearest_neighbours, next_nearest_neighbours
export transverse_field_ising, heisenberg_XYZ, heisenberg_XXZ, j1_j2, bose_hubbard_model
export pwave_superconductor, hubbard_model, tj_model

end # module
