using Revise, PEPSKit, TensorKit, Zygote, MPSKit
using MPSKitModels, LinearAlgebra, OptimKit
using PEPSKit:
    NORTH, SOUTH, WEST, EAST, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST,    @diffset
using JLD2, ChainRulesCore
using TensorOperations, VectorInterface
using Test

#HELPER FUNCTIONS
################

#Helper function for making the hamiltonian
function SqLatHeisenberg() 
    Sx, Sy, Sz, _ = spinmatrices(1//2)
    Dphys = ℂ^2
    σx = TensorMap(Sx, Dphys, Dphys)
    σy = TensorMap(Sy, Dphys, Dphys)
    σz = TensorMap(Sz, Dphys, Dphys)

    @tensor H[-1 -3; -2 -4] := -σx[-1, -2] * σx[-3, -4] + σy[-1, -2] * σy[-3, -4] + -σz[-1, -2] * σz[-3, -4]

    return H
end
#Helper function to calculate the energy
function H_expectation_value(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv, H::AbstractTensorMap{S,2,2}) where {S}
    E = 0.0
    for r in 1:size(ψ, 1), c in 1:size(ψ, 2)
        ρ₂ = PEPSKit.ρ₂_horizontal(r, c, ψ, env)

        @planar norm_state = ρ₂[1 2; 1 2]
        @planar energy_horizonal = H[1 3; 2 4] * ρ₂[2 4; 1 3]

        E = E + (energy_horizonal*(-1)^r )/norm_state
    end
    if !(E ≈ real(E)) 
        @warn "Zygote seems to manage to generate a state for which the energy is infinite. This is not good."
        @show E 
    end
    return real(E)
end

#costfunction for optimization of the PEPS
function cfun(x)
    (ψ, env) = x

    function fun(peps)
        env = leading_boundary(peps, alg_ctm, env)
        x = H_expectation_value(peps, env, SqLatHeisenberg())
        return x
    end

    env = leading_boundary(ψ, alg_ctm, env)
    E = H_expectation_value(ψ, env, SqLatHeisenberg())
    ∂E = fun'(ψ)
    @assert !isnan(norm(∂E))    
    return E, ∂E
end

#on top of the costfunction OptimKit.jl also needs to know how the tangent space around ψ₀ looks.
#retraction = take a step in the dx direction starting from x and return (new x, new tangent)
function my_retract(x, dx, α::Number)
    (ϕ, env0) = x
    ψ = deepcopy(ϕ)
    env = deepcopy(env0)
    ψ.A .+= dx.A .* α
    env = leading_boundary(ψ, alg_ctm,env)
    return (ψ, env), dx
end

#computes the inner product dx1, dx2 at position x. Here there is no x dependence !
my_inner(x, dx1, dx2) = real(dot(dx1, dx2))

#add a*X to Y
function my_add!(Y, X, a)
    Y.A .+= X.A .* a
    return Y
end

#scale η by β
function my_scale!(η, β)
    rmul!(η.A, β)
    return η
end


#ACTUAL TEST (First well do bosons)
#############

#some initial state
ph = ℂ^2
D = ℂ^1
A = map(zeros(1, 1)) do a
    TensorMap(rand, ComplexF64, ph ← D * D * D' * D')
end;
ψ₀  = InfinitePEPS(A);
#that has environments
alg_ctm = CTMRG(; tol=1e-10)
env₀ = leading_boundary(ψ₀, alg_ctm)
#leading to an energy
@info "The current energy is: " H_expectation_value(ψ₀, env₀, SqLatHeisenberg())

#one iteration of the costfunction gives : For now this gives an error. 
cfun((ψ₀, env₀))

# if all goes well we can now just use optimize.jl its GD algorithm to optimize the PEPS...
optimize(
    cfun,
    (ψ₀, leading_boundary(ψ₀, alg_ctm)),
    ConjugateGradient(; verbosity=2);
    inner=my_inner,
    retract=my_retract,
    (scale!)=my_scale!,
    (add!)=my_add!,
)


