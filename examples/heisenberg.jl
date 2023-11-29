using Revise, PEPSKit, TensorKit, Zygote, MPSKit
using MPSKitModels, LinearAlgebra, OptimKit
using PEPSKit:
    NORTH, SOUTH, WEST, EAST, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST, @diffset
using JLD2, ChainRulesCore

#function that evaluates the expectation value of the Hamiltonian
function H_expectation_value(
    ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv, H::AbstractTensorMap{S,2,2}
) where {S}
    E = 0.
    for r in 1:size(ψ, 1), c in 1:size(ψ, 2)
        ρ_hor = two_site_rho(r, c, ψ, env)

        @tensor n_hor = ρ_hor[1 2; 1 2]
        @tensor E_hor = H[3 4; 1 2] * ρ_hor[1 2; 3 4]

        ρ_ver = two_site_rho(r, c, rotl90(ψ), PEPSKit.rotate_north(env, EAST))
        @tensor n_ver = ρ_ver[1 2; 1 2]
        @tensor E_ver = H[3 4; 1 2] * ρ_ver[1 2; 3 4]
        
        E = E + E_hor / n_hor + E_ver/n_ver
    end
    return E
end

#function that builds the relevant two site operators
function SqLatHeisenberg()
    Sx, Sy, Sz, _ = spinmatrices(1//2)

    Dphys = ComplexSpace(2)
    σx = TensorMap(Sx, Dphys, Dphys)
    σy = TensorMap(Sy, Dphys, Dphys)
    σz = TensorMap(Sz, Dphys, Dphys)

    @tensor H[-1 -3; -2 -4] :=
        -σx[-1, -2] * σx[-3, -4] + σy[-1, -2] * σy[-3, -4] + -σz[-1, -2] * σz[-3, -4]

    return H
end
H = SqLatHeisenberg()

function cfun(x)
    (ψ, env) = x

    function fun(peps)
        env = leading_boundary(peps, alg_ctm, env)
        x = H_expectation_value(peps, env, H)
        return x
    end
    env = leading_boundary(ψ, alg_ctm, env)
    E = H_expectation_value(ψ, env, H)
    ∂E = fun'(ψ)

    @assert !isnan(norm(∂E))
    return E, ∂E
end

# my_retract is not an in place function which should not change x
function my_retract(x, dx, α::Number)
    (ϕ, env0) = x
    ψ = deepcopy(ϕ)
    env = deepcopy(env0)
    ψ.A .+= dx.A .* α
    #env = leading_boundary(ψ, alg_ctm,env)
    return (ψ, env), dx
end

my_inner(x, dx1, dx2) = real(dot(dx1, dx2))

function my_add!(Y, X, a)
    Y.A .+= X.A .* a
    return Y
end

function my_scale!(η, β)
    rmul!(η.A, β)
    return η
end

function init_psi(d::Int, D::Int, Lx::Int, Ly::Int)
    Pspaces = fill(ℂ^d, Lx, Ly)
    Nspaces = fill(ℂ^D, Lx, Ly)
    Espaces = fill(ℂ^D, Lx, Ly)

    Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1)))

    A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return TensorMap(rand, ComplexF64, P ← N * E * S * W)
    end

    return InfinitePEPS(A)
end

alg_ctm = CTMRG(; verbose=1, tol=1e-4, trscheme=truncdim(10), miniter=4, maxiter=200)

function main(; d=2, D=2, Lx=1, Ly=1)
    ψ = init_psi(d, D, Lx, Ly)
    env = leading_boundary(ψ, alg_ctm)

    @info "Starting optimization"
    @info "Initial energy: $(H_expectation_value(ψ, env, H))"
    E = H_expectation_value(ψ, env, H)

    for counter in 1:1000
        @info "Iteration $(counter)"
        ψ_trial = init_psi(d, D, Lx, Ly)
        env_trial = leading_boundary(ψ_trial, alg_ctm)
        E_trial = H_expectation_value(ψ_trial, env_trial, H)
        if real(E_trial) < real(E)
            ψ = ψ_trial
            env = env_trial
            E = E_trial
            @info "New energy: $(E)"
        end
    end
    #=
    optimize(
        cfun,
        (ψ, env),
        ConjugateGradient(; verbosity=2);
        inner=my_inner,
        retract=my_retract,
        (scale!)=my_scale!,
        (add!)=my_add!,
    )
    =#
    return ψ
end

main()
