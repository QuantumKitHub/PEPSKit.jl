abstract type AbstractInteraction end

"""
    struct OnSite <: AbstractInteraction

Trivial interaction representing terms that act on one isolated site.
"""
struct OnSite <: AbstractInteraction end

"""
    struct NearestNeighbor <: AbstractInteraction

Interaction representing nearest neighbor terms that act on two adjacent sites.
"""
struct NearestNeighbor <: AbstractInteraction end

"""
    struct NLocalOperator{I<:AbstractInteraction}
    
Operator in form of a `AbstractTensorMap` which is parametrized by an interaction type.
Mostly, this is used to define Hamiltonian terms and observables.
"""
struct NLocalOperator{I<:AbstractInteraction}
    op::AbstractTensorMap
end

@doc """
    operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::AbstractInteraction)

Contract a PEPS and a CTMRG environment to form an operator environment.
The open bonds correspond to the indices of an operator with the specified
`AbstractInteraction` type.
"""
operator_env

function operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::OnSite)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        @tensor opt = true ρ[-1; -2] :=
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][-1; 3 7 11 15] *
            conj(peps[r, c][-2; 4 8 12 16])
    end
end

function operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::NearestNeighbor)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        cnext = _next(c, size(peps, 2))
        @tensor opt = true ρ[-12 -18; -11 -20] :=
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][-12; 5 15 6 7] *
            conj(peps[r, c][-11; 8 19 10 9]) *
            peps[r, cnext][-18; 16 25 17 15] *
            conj(peps[r, cnext][-20; 22 26 21 19])
    end
end

@doc """
    MPSKit.expectation_value(peps::InfinitePEPS, env, O::NLocalOperator)

Evaluate the expectation value of any `NLocalOperator` on each unit-cell entry
of `peps` and `env`.
""" MPSKit.expectation_value(::InfinitePEPS, ::Any, ::NLocalOperator)

# Optimal contraction order is obtained by manually trying out some space sizes and using costcheck = warn
# in principle, we would like to use opt = true, but this does not give optimal results without also supplying costs
# However, due to a bug in tensoroperations this is currently not possible with integer labels.
# Thus, this is a workaround until the bug is fixed. (otherwise we'd need to rewrite all the labels to symbols...)

# 1-site operator expectation values on unit cell
function MPSKit.expectation_value(
    peps::InfinitePEPS, env::CTMRGEnv, O::NLocalOperator{OnSite}
)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        o = @tensor order = (6, 2, 5, 10, 14, 13, 11, 15, 7, 9, 1, 3, 4, 8, 12, 16, 18, 17) begin
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][17; 3 7 11 15] *
            conj(peps[r, c][18; 4 8 12 16]) *
            O.op[18; 17]
        end
        n = @tensor order = (9, 13, 10, 5, 1, 2, 4, 16, 6, 8, 14, 12, 17, 3, 7, 11, 15) begin
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][17; 3 7 11 15] *
            conj(peps[r, c][17; 4 8 12 16])
        end
        o / n
    end
end

#! format: off
function MPSKit.expectation_value(
    peps::InfinitePEPS, env, O::NLocalOperator{NearestNeighbor}
)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        cnext = _next(c, size(peps, 2))
        o = @tensor order = (
            28, 24, 23, 16, 25, 22, 26, 27, 17, 21, 4, 1, 3, 5, 7, 8, 9, 2, 6, 10, 14, 19,
            15, 13, 31, 32, 29, 30,
        ) begin # physical spaces
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][29; 5 15 6 7] *
            conj(peps[r, c][31; 8 19 10 9]) *
            peps[r, cnext][30; 16 25 17 15] *
            conj(peps[r, cnext][32; 22 26 21 19]) *
            O.op[31 32; 29 30]
        end

        n = @tensor order = (
            2, 3, 1, 5, 7, 28, 24, 23, 16, 25, 30, 22, 26, 27, 17, 21, 14, 15, 6, 4, 13, 29,
            8, 19, 10, 9,
        ) begin
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][29; 5 15 6 7] *
            conj(peps[r, c][29; 8 19 10 9]) *
            peps[r, cnext][30; 16 25 17 15] *
            conj(peps[r, cnext][30; 22 26 21 19])
        end
        o / n
    end
end
#! format: on

"""
    costfun(peps::InfinitePEPS, env, op::NLocalOperator{NearestNeighbor})
    
Compute the expectation value of a nearest-neighbor operator.
This is used to evaluate and differentiate the energy in ground-state PEPS optimizations.
"""
function costfun(peps::InfinitePEPS, env, op::NLocalOperator{NearestNeighbor})
    oh = sum(expectation_value(peps, env, op))
    ov = sum(expectation_value(rotl90(peps), rotl90(env), op))
    return real(oh + ov)
end
