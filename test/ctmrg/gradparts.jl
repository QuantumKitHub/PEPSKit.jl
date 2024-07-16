using Test
using Random
using PEPSKit
using TensorKit
using PEPSKit:
    NORTH,
    SOUTH,
    WEST,
    EAST,
    NORTHWEST,
    NORTHEAST,
    SOUTHEAST,
    SOUTHWEST,
    rotate_north,
    left_move,
    ctmrg_iter

using Zygote
using ChainRulesCore
using ChainRulesTestUtils

include(joinpath("..", "utility.jl"))

## Test spaces, tested functions and CTMRG algorithm
# --------------------------------------------------
χbond = 2
χenv = 4
Pspaces = [ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1)]
Vspaces = [ComplexSpace(χbond), Vect[FermionParity](0 => χbond / 2, 1 => χbond / 2)]
Espaces = [ComplexSpace(χenv), Vect[FermionParity](0 => χenv / 2, 1 => χenv / 2)]
tol = 1e-10
atol = 1e-6
boundary_algs = [
    CTMRG(; tol, verbosity=0, ctmrgscheme=:simultaneous),
    CTMRG(; tol, verbosity=0, ctmrgscheme=:sequential),
]

## Gauge invariant function of the environment
# --------------------------------------------
function rho(env)
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
@testset "Reverse rules for ctmrg_iter with spacetype $(Vspaces[i])" for i in
                                                                         eachindex(Pspaces)
    Random.seed!(42039482030)
    psi = InfinitePEPS(Pspaces[i], Vspaces[i], Vspaces[i])
    env = CTMRGEnv(psi, Espaces[i])

    @testset "$alg" for alg in boundary_algs
        @info "$(typeof(alg)) on $(Vspaces[i])"
        f(state, env) = rho(ctmrg_iter(state, env, alg)[1])

        # use rrule testing functionality but compute rrule via Zygote
        test_rrule(
            Zygote.ZygoteRuleConfig(),
            f,
            psi,
            env;
            check_inferred=false,
            atol,
            rrule_f=rrule_via_ad,
        )
    end
end
