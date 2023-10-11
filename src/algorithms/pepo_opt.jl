# First go at PEPO fixed point optimization using PEPSKit

## Characterize PEPO optimization manifold

# point on optimization manifold contains PEPS, its 'boundary'
# the norm of the gradient is for convenience for now, but should be removed
mutable struct PEPOOptPoint{T,E,F}
    state::T
    envs::E
    normgrad::F
end
PEPOOptPoint(state, envs) = PEPOOptPoint(state, envs, Inf)

# the gradient is just a 2D array of PEPS tensors, is this sufficient?
function _retract(x::PEPOOptPoint, dx, α::Number)
    # move the state, keep the rest?
    new_state = copy(x.state)
    new_state.A .+= dx .* α
    return PEPOOptPoint(new_state, x.envs, x.normgrad), dx
end

_inner(x, dx1, dx2) = real(dot(dx1, dx2))

function _add!(Y, X, a)
    return add!(Y, X, a)
end

function _scale!(η, β)
    return scale!(η, β)
end

## Utility; TODO: remove all of this

function _nthroot(x::Real, n::Integer)
    return if isodd(n) || x ≥ 0
        copysign(abs(x)^(1//n), x)
    else
        throw(
            DomainError(
                "Exponentiation yielding a complex result requires a complex argument.  Replace nthroot(x, n) with complex(x)^(1//n).",
            ),
        )
    end
end
function _hacked_root(x, n)
    m = abs(x)
    p = angle(x)
    mY = _nthroot(m, n)
    pYs = p / n .+ 2 * pi / n .* (0:(n - 1))
    I = argmin(minimum([abs.(pYs .- p); abs.(2 * pi .- pYs .+ p)]; dims=1)) # find phase with minimal angular distance
    return mY * exp(pYs[I] * 1im)
end

isverbose(alg::VUMPS) = alg.verbose
isverbose(alg::GradientGrassmann) = alg.method.verbosity >= 0


## Characterize environments for PEPO optimization

mutable struct PEPOOptEnv{T,O,F} <: Cache
    peps_boundary::T
    pepo_boundary::O
    alg::F
end

# TODO: think about name; just picked something to avoid conflict with MPSKit.environments
function pepo_opt_environments(
    peps::InfinitePEPS,
    pepo::InfinitePEPO,
    boundary_alg;
    vspaces=[oneunit(space(peps, 1, 1))],
    hermitian=false,
    kwargs...,
)
    # boundary_alg handles everything
    peps_boundary = environments(peps, boundary_alg; vspaces, hermitian)
    pepo_boundary = environments(peps, pepo, boundary_alg; vspaces, hermitian)

    return PEPOOptEnv(peps_boundary, pepo_boundary, boundary_alg)
end

# I did overload recalculate!, this seemed to make sense
function MPSKit.recalculate!(
    envs::PEPOOptEnv,
    peps::InfinitePEPS,
    pepo::InfinitePEPO;
    tol=algtol(envs.alg),
    hermitian=false,
    kwargs...,
)
    recalculate!(envs.peps_boundary, peps; tol, hermitian, kwargs...)
    recalculate!(envs.pepo_boundary, peps, pepo; tol, hermitian, kwargs...)

    return envs
end


## PEPO fixed point optimization algorithm

# first attempt at a PEPO-fixed-point-optimization algorithm, bit of a mess...
struct PEPOOptimize{A}
    optim_method::OptimKit.OptimizationAlgorithm
    optim_finalize!::Function
    boundary_method::A
    tol_min::Float64 # some basic tolerance scaling
    tol_max::Float64
    tol_factor::Float64
    symm::SymmetrizationStyle # symmetrization
    hermitian::Bool

    function PEPOOptimize(;
        optim_method=ConjugateGradient,
        (optim_finalize!)=OptimKit._finalize!,
        optim_tol=Defaults.tol,
        optim_maxiter=Defaults.maxiter,
        verbosity=2,
        boundary_method=VUMPS,
        boundary_maxiter=Defaults.maxiter,
        boundary_finalize=MPSKit.Defaults._finalize,
        tol_min=1e-12,
        tol_max=1e-5,
        tol_factor=1e-3,
        symm=None(),
        hermitian=false,
    )
        if isa(optim_method, OptimKit.OptimizationAlgorithm)
            # We were given an optimisation method, just use it.
            m = optim_method
        elseif optim_method <: OptimKit.OptimizationAlgorithm
            # We were given an optimisation method type, construct an instance of it.
            m = optim_method(;
                gradtol=optim_tol, maxiter=optim_maxiter, verbosity=verbosity
            )
        else
            msg = "optim_method should be either an instance or a subtype of `OptimKit.OptimizationAlgorithm`."
            throw(ArgumentError(msg))
        end
        if isa(boundary_method, Union{<:VUMPS,CTMRG,GradientGrassmann})
            bm = boundary_method
        elseif boundary_method <: Union{Type{CTMRG},MPSKit.Algorithm}
            # total syntax clusterfuck, need to clean this up
            if boundary_method <: VUMPS
                bm = boundary_method(;
                    tol_galerkin=tol_max,
                    verbose=verbosity >= 5,
                    maxiter=boundary_maxiter,
                    finalize=boundary_finalize,
                )
            elseif boundary_method <: GradientGrassmann
                bm = boundary_method(;
                    tol=tol_max, verbosity=verbosity - 4, maxiter=boundary_maxiter
                )
            elseif boundary_method <: CTMRG
                bm = method(; tol=tol_max, verbose=verbosity >= 5, maxiter=boundary_maxiter)
            else
                msg = "Unknown boundary contraction method."
                throw(ArgumentError(msg))
            end
        else
            msg = "boundary_method should be a valid boundary contraction algorithm."
            throw(ArgumentError(msg))
        end
        return new{typeof(bm)}(
            m, optim_finalize!, bm, tol_min, tol_max, tol_factor, symm, hermitian
        )
    end
end

# default PEPO optimization cost function for given PEPO and optimization algorithm
function pepo_opt_costfun(
    op::InfinitePEPO, alg::PEPOOptimize)
    D, W, H = size(op)
    nrm = D * W * H
    function pepo_opt_fg(x::PEPOOptPoint)
        # unpack state
        peps = symmetrize(x.state, alg.symm)
        envs = x.envs
        ng = x.normgrad

        # recompute environment with scaled tolerance
        boundary_tol = min(max(alg.tol_min, ng * alg.tol_factor), alg.tol_max)
        recalculate!(
            envs,
            peps,
            op;
            tol=boundary_tol,
            hermitian=alg.hermitian,
        )

        # compute cost function
        lambdas_peps = expectation_value(peps, envs.peps_boundary)
        lambdas_pepo = expectation_value(peps, op, envs.pepo_boundary)
        f = -log(real(prod(lambdas_pepo) / prod(lambdas_peps))) / nrm

        # compute gradient
        ∂p_peps = ∂∂peps(peps, envs.peps_boundary)
        ∂p_pepo = ∂∂peps(peps, op, envs.pepo_boundary)
        grad = - (2 / nrm) .* ∂p_pepo ./ lambdas_pepo .+ (2 / nrm) .* ∂p_peps ./ lambdas_peps
        grad = symmetrize(grad, alg.symm)
        # TODO: test if gradient is actually correct

        # TODO: decide whether we want to:
        #   Actually update the state in place after symmetrization?
        #   Update the environments and norm of gradient after each cost function
        #   evaluation, or move this to the optim_finalize to only update after line search
        #   has terminated?
        x.state = peps # is this sensible?
        x.envs = envs
        x.normgrad = sum(norm.(grad))

        # some temporary debugging verbosity, to be removed
        if isverbose(alg.boundary_method)
            lambda_peps = prod(lambdas_peps)
            lambda_pepo = prod(lambdas_pepo)

            lambda_pepo_root = _hacked_root(lambda_pepo, D * W * H)
            rel_im_pepo = abs(imag(lambda_pepo_root) / abs(lambda_pepo_root))
            @printf(
                "\n\tCurrent lambda_pepo: %f + %fim;\tRelative im. part: %e;\n",
                real(lambda_pepo_root),
                imag(lambda_pepo_root),
                rel_im_pepo,
            )

            lambda_peps_root = _hacked_root(lambda_peps, D * W)
            rel_im_peps = abs(imag(lambda_peps_root) / abs(lambda_peps_root))
            @printf(
                "\tCurrent lambda_peps: %f + %fim;\tRelative im. part: %e;\n\n",
                real(lambda_peps_root),
                imag(lambda_peps_root),
                rel_im_peps,
            )

            @printf("\tCurrent f: %f\n\n", f)

            lambdas_peps_bis = map(∂p_peps, peps.A) do ∂p, p
                @tensor ∂p[1; 2 3 4 5] * conj(p[1; 2 3 4 5])
            end
            lambdas_pepo_bis = map(∂p_pepo, peps.A) do ∂p, p
                @tensor ∂p[1; 2 3 4 5] * conj(p[1; 2 3 4 5])
            end
            @show diff_peps = maximum(abs.(lambdas_peps .- lambdas_peps_bis))
            @show diff_pepo = maximum(abs.(lambdas_pepo .- lambdas_pepo_bis))
        end

        return f, grad
    end
    return pepo_opt_fg
end


## The actual leading boundary routine

function MPSKit.leading_boundary(
    state::InfinitePEPS,
    op::InfinitePEPO,
    alg::PEPOOptimize,
    envs=pepo_opt_environments(state, op, alg.boundary_method; hermitian=alg.hermitian),
    fg=pepo_opt_costfun(op, alg),
)
    res = optimize(
        fg,
        PEPOOptPoint(state, envs),
        alg.optim_method;
        retract=_retract,
        inner=_inner,
        (scale!)=_scale!,
        (add!)=_add!,
        (finalize!)=alg.optim_finalize!,
    )
    (x, fx, gx, numfg, normgradhistory) = res
    return x, fx, normgradhistory[end]
end
