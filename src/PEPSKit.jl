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

include("states/abstractpeps.jl")
include("states/infinitepeps.jl")

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

include("algorithms/ctmrg/ctmrg.jl")
include("algorithms/ctmrg/gaugefix.jl")

# include("algorithms/vumps/vumps.jl")

include("algorithms/toolbox.jl")

include("algorithms/peps_opt.jl")

include("utility/symmetrization.jl")

"""
    module Defaults
        const ctmrg_maxiter = 100
        const ctmrg_miniter = 4
        const ctmrg_tol = 1e-12
        const fpgrad_maxiter = 100
        const fpgrad_tol = 1e-6
    end

Module containing default values that represent typical algorithm parameters.

- `ctmrg_maxiter = 100`: Maximal number of CTMRG iterations per run
- `ctmrg_miniter = 4`: Minimal number of CTMRG carried out
- `ctmrg_tol = 1e-12`: Tolerance checking singular value and norm convergence
- `fpgrad_maxiter = 100`: Maximal number of iterations for computing the CTMRG fixed-point gradient
- `fpgrad_tol = 1e-6`: Convergence tolerance for the fixed-point gradient iteration
"""
module Defaults
    using TensorKit, KrylovKit, OptimKit
    using PEPSKit: LinSolver, FixedSpaceTruncation, SVDAdjoint
    const ctmrg_maxiter = 100
    const ctmrg_miniter = 4
    const ctmrg_tol = 1e-8
    const fpgrad_maxiter = 30
    const fpgrad_tol = 1e-6
    const ctmrgscheme = :simultaneous
    const reuse_env = true
    const trscheme = FixedSpaceTruncation()
    const iterscheme = :fixed
    const fwd_alg = TensorKit.SVD()
    const rrule_alg = GMRES(; tol=1e1ctmrg_tol)
    const svd_alg = SVDAdjoint(; fwd_alg, rrule_alg)
    const optimizer = LBFGS(32; maxiter=100, gradtol=1e-4, verbosity=2)
    const gradient_linsolver = KrylovKit.BiCGStab(;
        maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol
    )
    const gradient_alg = LinSolver(; solver=gradient_linsolver, iterscheme)
end

export SVDAdjoint, IterSVD, NonTruncSVDAdjoint
export FixedSpaceTruncation, ProjectorAlg, CTMRG, CTMRGEnv, correlation_length
export LocalOperator
export expectation_value, costfun, product_peps
export leading_boundary
export PEPSOptimize, GeomSum, ManualIter, LinSolver
export fixedpoint

export InfinitePEPS, InfiniteTransferPEPS
export InfinitePEPO, InfiniteTransferPEPO
export initializeMPS, initializePEPS
export ReflectDepth, ReflectWidth, Rotate, RotateReflect
export symmetrize!, symmetrize_retract_and_finalize!
export showtypeofgrad
export InfiniteSquare, vertices, nearest_neighbours, next_nearest_neighbours
export transverse_field_ising, heisenberg_XYZ, j1_j2, pwave_superconductor

end # module
