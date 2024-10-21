module PEPSKit

using LinearAlgebra, Statistics, Base.Threads, Base.Iterators, Printf
using Base: @kwdef
using Compat
using Accessors
using VectorInterface
using TensorKit, KrylovKit, MPSKit, OptimKit, TensorOperations
using ChainRulesCore, Zygote
using LoggingExtras
using MPSKit: loginit!, logiter!, logfinish!, logcancel!
using MPSKitModels


include("utility/util.jl")
include("utility/svd.jl")
include("utility/rotations.jl")
include("utility/diffset.jl")
include("utility/hook_pullback.jl")
include("utility/autoopt.jl")
include("utility/loginfo.jl")

include("states/abstractpeps.jl")
include("states/infinitepeps.jl")

include("operators/transferpeps.jl")
# include("operators/infinitepepo.jl")
# include("operators/transferpepo.jl")
# include("operators/derivatives.jl")
include("operators/localoperator.jl")
include("operators/lattices/squarelattice.jl")
include("operators/models.jl")

include("environments/ctmrg_environments.jl")
# include("environments/transferpeps_environments.jl")
# include("environments/transferpepo_environments.jl")
include("environments/vumps_environments.jl")

abstract type Algorithm end
include("algorithms/contractions/localoperator.jl")
include("algorithms/contractions/ctmrg_contractions.jl")

include("algorithms/ctmrg/ctmrg.jl")
include("algorithms/ctmrg/gaugefix.jl")

include("algorithms/vumps/vumps.jl")

include("algorithms/toolbox.jl")

include("algorithms/peps_opt.jl")

include("utility/symmetrization.jl")

include("utility/defaults.jl")

export SVDAdjoint, IterSVD, NonTruncSVDAdjoint
export FixedSpaceTruncation, ProjectorAlg, CTMRG, CTMRGEnv, correlation_length
export LocalOperator
export expectation_value, costfun, product_peps
export leading_boundary
export PEPSOptimize, GeomSum, ManualIter, LinSolver
export fixedpoint

export InfinitePEPS, InfiniteTransferPEPS
export initializeMPS, initializePEPS
export ReflectDepth, ReflectWidth, Rotate, RotateReflect
export symmetrize!, symmetrize_retract_and_finalize!
export showtypeofgrad
export InfiniteSquare, vertices, nearest_neighbours, next_nearest_neighbours
export transverse_field_ising, heisenberg_XYZ, j1_j2, pwave_superconductor

end # module
