
using Test
using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit
using Zygote

## Setup

function three_dimensional_classical_ising(; beta, J=1.0)
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
ctm_styles = [:sequential, :simultaneous]
projector_algs = [:halfinfinite, :fullinfinite]

@testset "PEPO CTMRG runthroughs for unitcell=$(unitcell)" for unitcell in
                                                               [(1, 1, 1), (1, 1, 2)]
    Random.seed!(81812781144)

    # contract
    T = InfinitePEPO(O; unitcell=unitcell)
    psi0 = initializePEPS(T, χpeps)
    n = InfiniteSquareNetwork(psi0, T)
    env0 = CTMRGEnv(n, χenv)

    @testset "PEPO CTMRG contraction using $alg with $projector_alg" for (
        alg, projector_alg
    ) in Iterators.product(
        ctm_styles, projector_algs
    )
        env, = leading_boundary(env0, n; alg, maxiter=150, projector_alg)
    end
end

@testset "Fixed-point computation for 3D classical ising model" begin
    Random.seed!(81812781144)

    # prep
    ctm_alg = SimultaneousCTMRG(; maxiter=150, tol=1e-8, verbosity=2)
    alg_rrule = EigSolver(;
        solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true),
        iterscheme=:diffgauge,
    )
    opt_alg = LBFGS(32; maxiter=50, gradtol=1e-5, verbosity=3)
    function pepo_retract(x, η, α)
        peps = deepcopy(x[1])
        peps.A .+= η.A .* α
        env2 = deepcopy(x[2])
        env3 = deepcopy(x[3])
        return (peps, env2, env3), η
    end

    # contract
    T = InfinitePEPO(O; unitcell=(1, 1, 1))
    psi0 = initializePEPS(T, χpeps)
    env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), χenv)
    env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, T), χenv)

    # optimize free energy per site
    (psi_final, env2_final, env3_final), f, = optimize(
        (psi0, env2_0, env3_0), opt_alg; retract=pepo_retract, inner=PEPSKit.real_inner
    ) do (psi, env2, env3)
        E, gs = withgradient(psi) do ψ
            n2 = InfiniteSquareNetwork(ψ)
            env2′, info = PEPSKit.hook_pullback(
                leading_boundary, env2, n2, ctm_alg; alg_rrule
            )
            n3 = InfiniteSquareNetwork(ψ, T)
            env3′, info = PEPSKit.hook_pullback(
                leading_boundary, env3, n3, ctm_alg; alg_rrule
            )
            PEPSKit.ignore_derivatives() do
                PEPSKit.update!(env2, env2′)
                PEPSKit.update!(env3, env3′)
            end
            λ3 = network_value(n3, env3)
            λ2 = network_value(n2, env2)
            return -log(real(λ3 / λ2))
        end
        g = only(gs)
        return E, g
    end

    # check energy
    n3_final = InfiniteSquareNetwork(psi_final, T)
    m = PEPSKit.contract_local_tensor((1, 1, 1), M, n3_final, env3_final)
    nrm3 = PEPSKit._contract_site((1, 1), n3_final, env3_final)

    # compare to Monte-Carlo result from https://www.worldscientific.com/doi/abs/10.1142/S0129183101002383
    @test abs(m / nrm3) ≈ 0.667162 rtol = 1e-2
end
