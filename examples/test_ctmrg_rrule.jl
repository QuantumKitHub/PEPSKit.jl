using LinearAlgebra
using TensorKit, MPSKitModels, OptimKit, Zygote
using PEPSKit, KrylovKit, VectorInterface

# Square lattice Heisenberg Hamiltonian
# Sublattice-rotate to get (1, 1, 1) → (-1, 1, -1), transformed to GS with single-site unit cell
function square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
    Sx, Sy, Sz, _ = spinmatrices(1//2)
    Vphys = ℂ^2
    σx = TensorMap(Sx, Vphys, Vphys)
    σy = TensorMap(Sy, Vphys, Vphys)
    σz = TensorMap(Sz, Vphys, Vphys)

    @tensor H[-1 -3; -2 -4] :=
        Jx * σx[-1, -2] * σx[-3, -4] +
        Jy * σy[-1, -2] * σy[-3, -4] +
        Jz * σz[-1, -2] * σz[-3, -4]

    return NLocalOperator{NearestNeighbor}(H)
end

# Parameters
H = square_lattice_heisenberg(; Jx=-1, Jy=1, Jz=-1)
χbond = 2
χenv = 4
ctmalg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=2)

# Cost function for 'non-builtin' optimization approach
ψ₀ = InfinitePEPS(2, χbond)
env_init = CTMRGEnv(ψ₀; Venv=ℂ^χenv)

# env₀ = leading_boundary(ψ₀, ctmalg, deepcopy(env_init))


# direct copy of PEPSKit optimization code: check to see did not break anything
optalg = PEPSOptimize(;
    boundary_alg=ctmalg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=100),
    reuse_env=true,
    verbosity=2,
)
fixedpoint(deepcopy(ψ₀), H, optalg, deepcopy(env_init))


# dummy function to hook into
using PEPSKit: GradMode
using ChainRulesCore
myleadingboundary(gradmode, args...) = leading_boundary(args...)

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(myleadingboundary),
    ::NaiveAD,
    state,
    alg::CTMRG,
    envinit,
)
    envs, pullback = rrule_via_ad(config, leading_boundary, state, alg, envinit)
    function myleadingboundary_pullback(Δenvs)
        ∂fun, ∂state, ∂alg, ∂envinit = pullback(Δenvs)
        @show typeof(∂state)
        return (NoTangent(), ∂fun, ∂state, ∂alg, ∂envinit)
    end
    return envs, myleadingboundary_pullback
end

# check if the rrule is working
x, f, normgrad = optimize(
    (deepcopy(ψ₀), deepcopy(env_init)),
    LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2); # ConjugateGradient(; gradtol=1e-6, verbosity=4, maxiter=100),
    inner=PEPSKit.my_inner, retract=PEPSKit.my_retract,
    (scale!)=VectorInterface.scale!, (add!)=VectorInterface.add!,
) do (ψ, envs)
    E, g = withgradient(ψ) do ψ
        envs′ = myleadingboundary(NaiveAD(), ψ, ctmalg, envs)
        return costfun(ψ, envs′, H)
    end

    ∂E∂A = g[1]
    if !(∂E∂A isa InfinitePEPS)
        ∂E∂A = InfinitePEPS(∂E∂A.A)
    end
    @assert !isnan(norm(∂E∂A))
    return E, ∂E∂A
end

# check if our custom rule does what it should:
function ChainRulesCore.rrule(
    ::typeof(myleadingboundary), gradmode, state, alg::CTMRG, envinit
)
    envs = leading_boundary(state, alg, envinit)

    function leading_boundary_pullback(::AbstractZero)
        return NoTangent(), NoTangent(), ZeroTangent(), NoTangent(), ZeroTangent()
    end
    function leading_boundary_pullback(Δenvs′)
        Δenvs = CTMRGEnv(unthunk(Δenvs′)...)

        # find partial gradients of single ctmrg iteration
        _, envvjp = pullback(state, envs) do A, x
            return PEPSKit.gauge_fix(x, PEPSKit.ctmrg_iter(A, x, alg)[1])
        end

        ∂f∂A(x) = InfinitePEPS(envvjp(x)[1]...)
        ∂f∂x(x) = CTMRGEnv(envvjp(x)[2]...)

        # evaluate the geometric sum
        ∂F∂envs = PEPSKit.fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)
        # somehow the costfun seems to be generating a very weird tangent type,
        # so we need to manually construct something that works with it?
        weird_tangent = ChainRulesCore.Tangent{typeof(∂F∂envs)}(; A = ∂F∂envs.A)
        return NoTangent(), NoTangent(), weird_tangent, NoTangent(), ZeroTangent()
    end

    return envs, leading_boundary_pullback
end

x, f, normgrad = optimize(
    (deepcopy(ψ₀), deepcopy(env_init)),
    LBFGS(4; maxiter=100, gradtol=1e-4, verbosity=2); # ConjugateGradient(; gradtol=1e-6, verbosity=4, maxiter=100),
    inner=PEPSKit.my_inner,    retract=PEPSKit.my_retract,
    (scale!)=VectorInterface.scale!, (add!)=VectorInterface.add!,
) do (ψ, envs)
    E, g = withgradient(ψ) do ψ
        envs′ = myleadingboundary(ManualIter(), ψ, ctmalg, envs)
        reuse_env && (envs = envs′)
        return costfun(@showtypeofgrad(ψ), envs′, H)
    end

    ∂E∂A = getindex(@showtypeofgrad(g), 1)
    if !(∂E∂A isa InfinitePEPS)
        
        ∂E∂A′ = @showtypeofgrad InfinitePEPS(∂E∂A.A)
    else
        ∂E∂A′ = ∂E∂A
    end
    @show typeof(∂E∂A)
    # @assert !isnan(norm(∂E∂A))
    return E, ∂E∂A′
end

E, pull = Zygote.pullback(costfun, ψ₀, env_init, H);

pull(E)