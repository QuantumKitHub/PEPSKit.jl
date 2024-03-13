abstract type AbstractInteraction end

struct OnSite <: AbstractInteraction end
struct NearestNeighbor <: AbstractInteraction end

struct NLocalOperator{I<:AbstractInteraction}
    op::AbstractTensorMap
end

# Contraction of CTMRGEnv and PEPS tensors with open physical bonds
function operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::OnSite)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        @tensor ρ[-1; -2] :=
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

# Horizontally extended contraction of CTMRGEnv and PEPS tensors with open physical bonds
function operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::NearestNeighbor)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        cnext = _next(c, size(peps, 2))
        @tensor ρ[-11 -20; -12 -18] :=
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

# 1-site operator expectation values on unit cell
function MPSKit.expectation_value(peps::InfinitePEPS, env, O::NLocalOperator{OnSite})
    result = similar(peps.A, eltype(O.op))
    ρ = operator_env(peps, env, OnSite())

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        o = @tensor ρ[r, c][1; 2] * O.op[1; 2]
        n = @tensor ρ[r, c][1; 1]
        @diffset result[r, c] = o / n
    end

    return result
end

# 2-site operator expectation values on unit cell
function MPSKit.expectation_value(
    peps::InfinitePEPS, env, O::NLocalOperator{NearestNeighbor}
)
    result = similar(peps.A, eltype(O.op))
    ρ = operator_env(peps, env, NearestNeighbor())

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        o = @tensor ρ[r, c][1 2; 3 4] * O.op[1 2; 3 4]
        n = @tensor ρ[r, c][1 2; 1 2]
        @diffset result[r, c] = o / n
    end

    return result
end

# ⟨O⟩ from vertical and horizontal nearest-neighbor contributions
function costfun(peps::InfinitePEPS, env, op::NLocalOperator{NearestNeighbor})
    oh = sum(expectation_value(peps, env, op))
    ov = sum(expectation_value(rotl90(peps), rotl90(env), op))
    return real(oh + ov)
end
