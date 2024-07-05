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
functions = [left_move, ctmrg_iter, leading_boundary]
tol = 1e-8
projector_alg = ProjectorAlg(; trscheme=truncdim(χenv), fixedspace=true)
boundary_alg = CTMRG(; tol=tol, miniter=4, maxiter=100, verbosity=0, projector_alg)

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
@testset "Reverse rules for composite parts of the CTMRG fixed point with spacetype $(Vspaces[i])" for i in
                                                                                                       eachindex(
    Pspaces
)
    psi = InfinitePEPS(Pspaces[i], Vspaces[i], Vspaces[i])
    env = CTMRGEnv(psi; Venv=Espaces[i])

    @testset "$f" for f in functions
        atol = f == leading_boundary ? sqrt(tol) : tol
        g = if f == leading_boundary
            function (state, env)
                return rho(f(env, state, boundary_alg))
            end
        else
            function (state, env)
                return rho(f(state, env, boundary_alg)[1])
            end
        end

        # use rrule testing functionality but compute rrule via Zygote
        test_rrule(
            Zygote.ZygoteRuleConfig(),
            g,
            psi,
            env;
            check_inferred=false,
            atol,
            rrule_f=rrule_via_ad,
        )
    end
end
