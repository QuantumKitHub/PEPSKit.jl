"""
    module Defaults

Module containing default values that represent typical algorithm parameters.

- `contr_maxiter`: Maximum number of iterations for the contraction algorithm.
- `contr_miniter`: Minimum number of iterations for the contraction algorithm.
- `contr_tol`: Tolerance for the contraction algorithm.
- `fpgrad_maxiter`: Maximum number of iterations for the fixed-point gradient algorithm.
- `fpgrad_tol`: Tolerance for the fixed-point gradient algorithm.
- `verbosity`: Level of verbosity for the algorithm.
- `contractionscheme`: Scheme for contracting the environment.
- `reuse_env`: Whether to reuse the environment.
- `trscheme`: Truncation scheme.
- `iterscheme`: Iteration scheme.
- `fwd_alg`: Forward algorithm for the SVD.
- `rrule_alg`: Rule algorithm for the SVD.
- `svd_alg`: SVD algorithm.
- `optimizer`: Optimizer for the algorithm.
- `gradient_linsolver`: Linear solver for the gradient.
- `gradient_alg`: Algorithm for the gradient.
- `_finalize`: Function to finalize the algorithm.
"""
module Defaults
    const VERBOSE_NONE = 0
    const VERBOSE_WARN = 1
    const VERBOSE_CONV = 2
    const VERBOSE_ITER = 3
    const VERBOSE_ALL = 4

    using TensorKit, KrylovKit, OptimKit
    using PEPSKit: LinSolver, FixedSpaceTruncation, SVDAdjoint
    const eltype = ComplexF64
    const contr_maxiter = 100
    const contr_miniter = 4
    const contr_tol = 1e-8
    const fpgrad_maxiter = 30
    const fpgrad_tol = 1e-6
    const verbosity = VERBOSE_ITER
    const ctmrgscheme = :simultaneous
    const reuse_env = true
    const trscheme = FixedSpaceTruncation()
    const iterscheme = :fixed
    const fwd_alg = TensorKit.SVD()
    const rrule_alg = GMRES(; tol=1e1contr_tol)
    const svd_alg = SVDAdjoint(; fwd_alg, rrule_alg)
    const optimizer = LBFGS(32; maxiter=100, gradtol=1e-8, verbosity=2)
    const gradient_linsolver = KrylovKit.BiCGStab(;
        maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol
    )
    const gradient_alg = LinSolver(; solver=gradient_linsolver, iterscheme)

    _finalize(iter, state, opp, envs) = (state, envs)
end