# -------- For left-to-right correlator contraction --------

function start_correlator_left(
        i::CartesianIndex{2}, ρ::InfinitePEPO,
        O::PFTensor, env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(i)
    E_north = edge(env, NORTH, r - 1, c)
    E_south = edge(env, SOUTH, r + 1, c)
    E_west = edge(env, WEST, r, c - 1)
    C_northwest = corner(env, NORTHWEST, r - 1, c - 1)
    C_southwest = corner(env, SOUTHWEST, r + 1, c - 1)
    t = twistdual(ρ[r, c], 1:2)
    # TODO: part of these contractions is duplicated between the two output tensors,
    # so could be optimized further
    @autoopt @tensor Vn[χSE De; χNE] :=
        E_south[χSE Ds; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW Dw; χNW] * C_northwest[χNW; χN] *
        t[d d; Dn De Ds Dw] * E_north[χN Dn; χNE]
    @autoopt @tensor Vo[χSE De dstring; χNE] :=
        E_south[χSE Ds; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW Dw; χNW] * C_northwest[χNW; χN] *
        removeunit(O, 1)[d2; d1 dstring] *
        t[d1 d2; Dn De Ds Dw] * E_north[χN Dn; χNE]
    return Vn, Vo
end

function end_correlator_right_numerator(
        j::CartesianIndex{2}, V::CTMRGEdgeTensor{T, S, 3},
        ρ::InfinitePEPO, O::PFTensor, env::CTMRGEnv
    ) where {T, S}
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(j)
    E_north = edge(env, NORTH, r - 1, c)
    E_east = edge(env, EAST, r, c + 1)
    E_south = edge(env, SOUTH, r + 1, c)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)
    t = twistdual(ρ[r, c], 1:2)
    return @autoopt @tensor V[χSW DW dstring; χNW] *
        E_south[χSSE DS; χSW] * E_east[χNEE DE; χSEE] * E_north[χNW DN; χNNE] *
        C_northeast[χNNE; χNEE] * C_southeast[χSEE; χSSE] *
        t[d1 d2; DN DE DS DW] * removeunit(O, 4)[dstring d2; d1]
end

function end_correlator_right_denominator(
        j::CartesianIndex{2}, V::CTMRGEdgeTensor{T, S, 2}, env::CTMRGEnv
    ) where {T, S}
    r, c = Tuple(j)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    E_east = edge(env, EAST, r, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)
    return @autoopt @tensor V[χS DE; χN] * C_northeast[χN; χNE] *
        E_east[χNE DE; χSE] * C_southeast[χSE; χS]
end

# -------- For right-to-left correlator contraction --------

function start_correlator_right(
        i::CartesianIndex{2}, ρ::InfinitePEPO,
        O::PFTensor, env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(i)
    E_north = edge(env, NORTH, r - 1, c)
    E_east = edge(env, EAST, r, c + 1)
    E_south = edge(env, SOUTH, r + 1, c)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)
    t = twistdual(ρ[r, c], 1:2)
    @autoopt @tensor Vn[χNW DW; χSW] :=
        E_south[χSSE DS; χSW] * E_east[χNEE DE; χSEE] *
        E_north[χNW DN; χNNE] * C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] * t[d d; DN DE DS DW]
    @autoopt @tensor Vo[χNW DW dstring; χSW] :=
        E_south[χSSE DS; χSW] * E_east[χNEE DE; χSEE] *
        E_north[χNW DN; χNNE] * C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] *
        removeunit(O, 1)[d2; d1 dstring] * t[d1 d2; DN DE DS DW]
    return Vn, Vo
end


function end_correlator_left_numerator(
        j::CartesianIndex{2}, V::CTMRGEdgeTensor{T, S, 3},
        ρ::InfinitePEPO, O::PFTensor, env::CTMRGEnv
    ) where {T, S}
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(j)
    E_north = edge(env, NORTH, r - 1, c)
    E_south = edge(env, SOUTH, r + 1, c)
    E_west = edge(env, WEST, r, c - 1)
    C_northwest = corner(env, NORTHWEST, r - 1, c - 1)
    C_southwest = corner(env, SOUTHWEST, r + 1, c - 1)
    t = twistdual(ρ[r, c], 1:2)
    return @autoopt @tensor V[χNE DE dstring; χSE] *
        E_south[χSE DS; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW DW; χNW] * C_northwest[χNW; χN] *
        E_north[χN DN; χNE] *
        t[d1 d2; DN DE DS DW] * removeunit(O, 4)[dstring d2; d1]
end

function end_correlator_left_denominator(
        j::CartesianIndex{2}, V::CTMRGEdgeTensor{T, S, 2}, env::CTMRGEnv
    ) where {T, S}
    r, c = Tuple(j)
    C_northwest = corner(env, NORTHWEST, r - 1, c - 1)
    E_west = edge(env, WEST, r, c - 1)
    C_southwest = corner(env, SOUTHWEST, r + 1, c - 1)
    return @autoopt @tensor V[χN DE; χS] * C_southwest[χS; χSW] *
        E_west[χSW DE; χNW] * C_northwest[χNW; χN]
end
