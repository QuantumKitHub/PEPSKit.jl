using Test
using Random
using PEPSKit
using TensorKit
using PEPSKit: NORTH, SOUTH, WEST, EAST, NORTHWEST, NORTHEAST, SOUTHEAST, SOUTHWEST, rotate_north, left_move, ctmrg_iter
using Zygote
using ChainRulesCore
using ChainRulesTestUtils
using FiniteDifferences

## Test utility
# -------------
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::AbstractTensorMap)
    return TensorMap(randn, scalartype(x), space(x))
end
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::CTMRGEnv)
    Ctans = x.corners
    Etans = x.edges
    for i in eachindex(x.corners)
        Ctans[i] = rand_tangent(rng, x.corners[i])
    end
    for i in eachindex(x.edges)
        Etans[i] = rand_tangent(rng, x.edges[i])
    end
    return CTMRGEnv(Ctans,Etans)
end
function ChainRulesTestUtils.test_approx(actual::AbstractTensorMap,
                                         expected::AbstractTensorMap, msg=""; kwargs...)
    for (c, b) in blocks(actual)
        ChainRulesTestUtils.@test_msg msg isapprox(b, block(expected, c); kwargs...)
    end
end
function ChainRulesTestUtils.test_approx(actual::InfinitePEPS,
                                         expected::InfinitePEPS, msg=""; kwargs...)
    for i in eachindex(size(actual,1))
        for j in eachindex(size(actual,2))
            ChainRulesTestUtils.@test_msg msg isapprox(actual[i,j], expected[i,j]; kwargs...)
        end
    end
end
function ChainRulesTestUtils.test_approx(actual::CTMRGEnv,
    expected::CTMRGEnv, msg=""; kwargs...)
    for i in eachindex(actual.corners)
        ChainRulesTestUtils.@test_msg msg isapprox(actual.corners[i], expected.corners[i]; kwargs...)
    end
    for i in eachindex(actual.edges)
        ChainRulesTestUtils.@test_msg msg isapprox(actual.edges[i], expected.edges[i]; kwargs...)
    end
end

function FiniteDifferences.to_vec(t::T) where {T<:TensorKit.TrivialTensorMap}
    vec, from_vec = to_vec(t.data)
    return vec, x -> T(from_vec(x), codomain(t), domain(t))
end
function FiniteDifferences.to_vec(t::AbstractTensorMap)
    vec = mapreduce(vcat, blocks(t)) do (c, b)
        if scalartype(t) <: Real
            return reshape(b, :) .* sqrt(dim(c))
        else
            v = reshape(b, :) .* sqrt(dim(c))
            return vcat(real(v), imag(v))
        end
    end

    function from_vec(x)
        t′ = similar(t)
        T = scalartype(t)
        ctr = 0
        for (c, b) in blocks(t′)
            n = length(b)
            if T <: Real
                copyto!(b, reshape(x[(ctr + 1):(ctr + n)], size(b)) ./ sqrt(dim(c)))
            else
                v = x[(ctr + 1):(ctr + 2n)]
                copyto!(b,
                        complex.(x[(ctr + 1):(ctr + n)], x[(ctr + n + 1):(ctr + 2n)]) ./
                        sqrt(dim(c)))
            end
            ctr += T <: Real ? n : 2n
        end
        return t′
    end

    return vec, from_vec
end
FiniteDifferences.to_vec(t::TensorKit.AdjointTensorMap) = to_vec(copy(t))

## Test spaces, tested functions and CTMRG algorithm
# --------------------------------------------------
χbond = 2
χenv = 4
Pspaces = [ComplexSpace(2),Vect[FermionParity](0=>1, 1=>1)]
Vspaces = [ComplexSpace(χbond), Vect[FermionParity](0=>χbond/2, 1=>χbond/2)]
Espaces = [ComplexSpace(χenv), Vect[FermionParity](0=>χenv/2, 1=>χenv/2)]
functions = [left_move, ctmrg_iter, leading_boundary]
Random.seed!(42039482030)
tol = 1e-8
boundary_alg = CTMRG(; trscheme=truncdim(χenv), tol=tol, miniter=4, maxiter=100, fixedspace=true, verbosity=0)


## Gauge invariant function of the environment
# --------------------------------------------
function rho(env)
    #
    @tensor ρ[-1 -2 -3 -4 -5 -6 -7 -8] := 
        env.edges[WEST, 1, 1][1 -1 -2; 4] *
        env.corners[NORTHWEST, 1, 1][4; 5] *
        env.edges[NORTH, 1, 1][5 -3 -4; 8] *
        env.corners[NORTHEAST, 1, 1][8; 9] *
        env.edges[EAST, 1, 1][9 -5 -6; 12] *
        env.corners[SOUTHEAST, 1, 1][12; 13] *
        env.edges[SOUTH, 1, 1][13 -7 -8; 16] *
        env.corners[SOUTHWEST, 1, 1][16; 1]
    return ρ    
end

## Tests
# ------
@testset "Reverse rules for composite parts of the CTMRG fixed point with spacetype $(Vspaces[i])" for i in eachindex(Pspaces)
    psi = InfinitePEPS(Pspaces[i], Vspaces[i], Vspaces[i])
    env = CTMRGEnv(psi; Venv=Espaces[i])
    @testset "$func" for func in functions
        function f(state, env)
            if func != leading_boundary
                return rho(func(state, env, boundary_alg)[1])
            else
                return rho(func(env, state, boundary_alg))
            end
        end
        function ChainRulesCore.rrule(::typeof(f), state::InfinitePEPS{T}, envs::CTMRGEnv) where {T}
            y, env_vjp = pullback(state, envs) do A, x
                #return rho(func(A, x, boundary_alg)[1])
                if func != leading_boundary
                    return rho(func(A, x, boundary_alg)[1])
                else
                    return rho(func(x, A, boundary_alg))
                end
            end
            return y, x -> (NoTangent(), env_vjp(x)...)
        end
        if func != leading_boundary
            test_rrule(f, psi, env; check_inferred=false, atol=tol)
        else
            test_rrule(f, psi, env; check_inferred=false, atol=sqrt(tol))
        end
    end
end