using Test
using Random
using LinearAlgebra
using PEPSKit, MPSKit
using TensorKit
using KrylovKit
using OptimKit
using Mooncake
using MatrixAlgebraKit
using PEPSKit: LoggingExtras

const MCExt = Base.get_extension(PEPSKit, :PEPSKitMooncakeExt)
@assert !isnothing(MCExt)

Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(Core.current_scope)}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(time)}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(Base.CoreLogging.with_logstate), Any, Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(Base.CoreLogging._invoked_min_enabled_level), Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(PEPSKit.LoggingExtras.withlevel), Any, Int}

Mooncake.tangent_type(::Type{<:Base.HashArrayMappedTries.HAMT}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{Base.HashArrayMappedTries.Leaf}) = Mooncake.NoTangent

## Setup

function three_dimensional_classical_ising(; beta, J = 1.0)
    K = beta * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    # magnetization tensor
    M = copy(O)
    M[2, 2, 2, 2, 2, 2] *= -1
    @tensor m[-1 -2; -3 -4 -5 -6] :=
        M[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]

    # bond interaction tensor and energy-per-site tensor
    e = ComplexF64[-J J; J -J] .* q
    @tensor e_x[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] * e[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_y[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * e[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    @tensor e_z[-1 -2; -3 -4 -5 -6] :=
        O[1 2; 3 4 5 6] * e[-1; 1] * q[-2; 2] * q[-3; 3] * q[-4; 4] * q[-5; 5] * q[-6; 6]
    e = e_x + e_y + e_z

    # fixed tensor map space for all three
    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS), TensorMap(m, TMS), TensorMap(e, TMS)
end

## Test

# initialize
beta = 0.2391 # slightly lower temperature than βc ≈ 0.2216544
O, M, E = three_dimensional_classical_ising(; beta)
χpeps = ℂ^2
χenv = ℂ^12

# cover all different flavors
ctm_styles = [:SequentialCTMRG, :SimultaneousCTMRG]
projector_algs = [:HalfInfiniteProjector, :FullInfiniteProjector]

@testset "PEPO CTMRG runthroughs for unitcell=$(unitcell)" for unitcell in
    [(1, 1, 1), (1, 1, 2)]
    Random.seed!(81812781144)

    # contract
    T = InfinitePEPO(O; unitcell = unitcell)
    psi0 = initializePEPS(T, χpeps)
    n = InfiniteSquareNetwork(psi0, T)
    env0 = CTMRGEnv(n, χenv)

    @test spacetype(typeof(T)) === ComplexSpace
    @test spacetype(T) === ComplexSpace
    @test sectortype(typeof(T)) === Trivial
    @test sectortype(T) === Trivial

    @testset "PEPO CTMRG contraction using $alg with $projector_alg" for (
            alg, projector_alg,
        ) in Iterators.product(ctm_styles, projector_algs)
        env, = leading_boundary(env0, n; alg, maxiter = 150, projector_alg)
    end
end

#=f1=Mooncake.zero_fcodual
rule_tester=Mooncake.build_rrule(Complex,1.0,1.0)
rule_tester(f1(Complex),f1(1.0),f1(1.0))=#

#@show Mooncake.is_primitive(Mooncake.DefaultCtx, Mooncake.ReverseMode, Tuple{Complex, Float64, Float64}, Base.get_world_counter())
#@show Base.which(Mooncake.rrule!!, (Complex,Float64, Float64))
@testset "Fixed-point computation for 3D classical ising model" begin
    Random.seed!(81812781144)

    # prep
    ctm_alg = SimultaneousCTMRG(; maxiter = 150, tol = 1.0e-8, verbosity = 2)
    gradient_alg = FixedPointGradient(;
        solver_alg = KrylovKit.Arnoldi(; maxiter = 30, tol = 1.0e-6, eager = true),
    )
    opt_alg = LBFGS(32; maxiter = 50, gradtol = 1.0e-5, verbosity = 3)
    function pepo_retract(x, η, α)
        x´_partial, ξ = PEPSKit.peps_retract(x[1:2], η, α)
        x´ = (x´_partial..., deepcopy(x[3]))
        return x´, ξ
    end
    function pepo_transport!(ξ, x, η, α, x´)
        return PEPSKit.peps_transport!(ξ, x[1:2], η, α, x´[1:2])
    end

    # contract
    T = InfinitePEPO(O; unitcell = (1, 1, 1))
    psi0 = initializePEPS(T, χpeps)
    env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), χenv)
    env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, T), χenv)

    function energ_test(ψ, env2, env3)
        n2 = InfiniteSquareNetwork(ψ)
        env2′, info = PEPSKit.hook_pullback(
            leading_boundary, env2, n2, ctm_alg; alg_rrule = gradient_alg
        )
        n3 = InfiniteSquareNetwork(ψ, T)
        env3′, info = PEPSKit.hook_pullback(
            leading_boundary, env3, n3, ctm_alg; alg_rrule = gradient_alg
        )
        PEPSKit.update!(env2, env2′)
        PEPSKit.update!(env3, env3′)
        λ3 = network_value(n3, env3)
        λ2 = network_value(n2, env2)
        return -log(abs(λ3 / λ2))
    end
    @show energ_test(psi0, env2_0, env3_0)

    # optimize free energy per site
    (psi_final, env2_final, env3_final), f, = optimize(
        (psi0, env2_0, env3_0),
        opt_alg;
        inner = PEPSKit.real_inner,
        retract = pepo_retract,
        (transport!) = (pepo_transport!),
    ) do (psi, env2, env3)
        function energ(ψ)
            n2 = InfiniteSquareNetwork(ψ)
            #=env2′, info = PEPSKit.hook_pullback(
                leading_boundary, env2, n2, ctm_alg; alg_rrule = gradient_alg
            )
            n3 = InfiniteSquareNetwork(ψ, T)
            env3′, info = PEPSKit.hook_pullback(
                leading_boundary, env3, n3, ctm_alg; alg_rrule = gradient_alg
            )=#
            env2′, info = PEPSKit.leading_boundary(env2, n2, ctm_alg)
            n3 = InfiniteSquareNetwork(ψ, T)
            env3′, info = PEPSKit.leading_boundary(env3, n3, ctm_alg)
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
            λ3 = network_value(n3, env3)
            λ2 = network_value(n2, env2)
            return -log(abs(λ3 / λ2))
        end
        cache = prepare_gradient_cache(energ, psi)
        E, gs = value_and_gradient!!(cache, energ, psi)
        _, g = Mooncake.arrayify(psi, gs[2])
        return E, g
    end

    # check energy
    n3_final = InfiniteSquareNetwork(psi_final, T)
    m = PEPSKit.contract_local_tensor((1, 1, 1), M, n3_final, env3_final)
    nrm3 = PEPSKit._contract_site((1, 1), n3_final, env3_final)

    # compare to Monte-Carlo result from https://www.worldscientific.com/doi/abs/10.1142/S0129183101002383
    @test abs(m / nrm3) ≈ 0.667162 rtol = 1.0e-2
end
