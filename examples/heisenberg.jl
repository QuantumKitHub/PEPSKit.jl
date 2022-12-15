using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using MPSKitModels, LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore


function two_site_rho(r::Int, c::Int, ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv)
    cp = mod1(c+1,size(ψ,2))
    @tensor ρ[-11,-20;-12,-18] := env.corners[NORTHWEST,r,c][1,3] * 
        env.edges[WEST,r,c][2,7,9,1] * 
        env.corners[SOUTHWEST,r,c][4,2] * 
        env.edges[NORTH,r,c][3,5,8,13] *
        env.edges[SOUTH,r,c][14,6,10,4] * 
        ψ[r,c][-12,5,15,6,7] * 
        conj(ψ[r,c][-11,8,19,10,9]) * 
        env.edges[NORTH,r,cp][13,16,22,23] * 
        env.edges[SOUTH,r,cp][28,17,21,14] * 
        ψ[r,cp][-18,16,25,17,15] * 
        conj(ψ[r,cp][-20,22,26,21,19]) * 
        env.corners[NORTHEAST,r,cp][23,24] * 
        env.edges[EAST,r,cp][24,25,26,27] * 
        env.corners[SOUTHEAST,r,cp][27,28]
    return ρ
end


function iCtmGsEh(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv, H::AbstractTensorMap{S,2,2}) where S
    #Es = Matrix{eltype(H)}(undef,size(ψ,1),size(ψ,2))
    E = 0.0
    for r in 1:size(ψ,1), c in 1:size(ψ,2)
        ρ = two_site_rho(r, c, ψ, env)
        nn = @tensor ρ[1,2;1,2]
        Eh = @tensor H[1,2;3,4]*ρ[1,2;3,4]
        Eh = Eh / nn
        E = E + Eh 
        #@diffset Es[r,c] = Eh;
    end
    return real(E)
end

function H_expectation_value(ψ::InfinitePEPS, env::PEPSKit.CTMRGEnv, H::AbstractTensorMap{S,2,2}) where S
    Eh = iCtmGsEh(ψ, env, H)

    ψ1 = rotl90(ψ)
    env1 = PEPSKit.rotate_north(env,EAST);
    Ev = iCtmGsEh(ψ1, env1, H)
    E = real(Eh + Ev)
    return E
end

function SqLatHeisenberg()
    Sx,Sy,Sz,_ = spinmatrices(1//2)

    Dphys = ComplexSpace(2)
    σx = TensorMap(Sx, Dphys, Dphys)
    σy = TensorMap(Sy, Dphys, Dphys)
    σz = TensorMap(Sz, Dphys, Dphys)

    @tensor H[-1 -3; -2 -4] := 
        -σx[-1,-2] * σx[-3,-4] + 
        σy[-1,-2] * σy[-3,-4] + 
        -σz[-1,-2] * σz[-3,-4]

    return H
end

H = SqLatHeisenberg()


function cfun(x)
    (ψ,env) = x

    function fun(peps)
        env = leading_boundary(peps, alg_ctm, env)
        x = H_expectation_value(peps, env, H)   
        return x
    end

    ∂E = fun'(ψ)
    env = leading_boundary(ψ, alg_ctm, env)
    E = H_expectation_value(ψ, env, H)

    @assert !isnan(norm(∂E))
    return E,∂E
end

# my_retract is not an in place function which should not change x
function my_retract(x,dx,α::Number)
    (ϕ,env0) = x
    ψ = deepcopy(ϕ)
    env = deepcopy(env0)
    ψ.A .+= dx.A .* α
    #env = leading_boundary(ψ, alg_ctm,env)
    return (ψ,env),dx
end

my_inner(x,dx1,dx2) = real(dot(dx1,dx2))

function my_add!(Y, X, a)
    Y.A .+= X.A .* a
    return Y
end

function my_scale!(η, β)
    rmul!(η.A, β)
    return η
end


function init_psi(d::Int, D::Int, Lx::Int, Ly::Int)
    Pspaces = fill(ℂ^d,Lx,Ly)
    Nspaces = fill(ℂ^D,Lx,Ly)
    Espaces = fill(ℂ^D,Lx,Ly)

    Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1)))

    A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return TensorMap(rand, ComplexF64, P ← N * E * S * W)
    end

    return InfinitePEPS(A)
end


alg_ctm = CTMRG(
            verbose=10000,
            tol=1e-10,
            trscheme=truncdim(10),
            miniter=4,
            maxiter=200
        )

function main(;d=2,D=2,Lx=1,Ly=1)
    ψ = init_psi(d,D,Lx,Ly)   
    env = leading_boundary(ψ, alg_ctm) 
    optimize(
        cfun, 
        (ψ,env),
        ConjugateGradient(verbosity=3); 
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!
    )
    return ψ
end

main()
