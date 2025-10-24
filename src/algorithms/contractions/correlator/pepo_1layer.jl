function MPSKit.transfer_left(
        vec::AbstractTensorMap{T, S, 1, 2},
        O::MPOTensor{S}, A::MPSTensor{S}, Ab::MPSTensor{S}
    ) where {T, S}
    return @tensor y[-1; -2 -3] := vec[1; 2 4] *
        A[4 5; -3] * O[2 3; 5 -2] * conj(Ab[1 3; -1])
end

function MPSKit.transfer_left(
        v::GenericMPSTensor{S, 3}, O::MPOTensor{S}, A::MPSTensor{S}, Ab::MPSTensor{S}
    ) where {S}
    return @tensor t[d_string -1 -2; -3] := v[d_string 1 2; 4] *
        A[4 5; -3] * O[2 3; 5 -2] * conj(Ab[1 3; -1])
end

function start_correlator(
        i::CartesianIndex{2}, ρ::InfinitePEPO,
        O::MPOTensor, env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(i)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    E_west = env.edges[WEST, mod1(r, end), _prev(c, end)]
    C_northwest = env.corners[NORTHWEST, _prev(r, end), _prev(c, end)]
    C_southwest = env.corners[SOUTHWEST, _next(r, end), _prev(c, end)]
    t = twistdual(ρ[mod1(r, end), mod1(c, end)], 1:2)
    # TODO: part of these contractions is duplicated between the two output tensors,
    # so could be optimized further
    @autoopt @tensor Vn[χSE; De χNE] :=
        E_south[χSE Ds; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW Dw; χNW] * C_northwest[χNW; χN] *
        t[d d; Dn De Ds Dw] * E_north[χN Dn; χNE]
    @autoopt @tensor Vo[dstring χSE De; χNE] :=
        E_south[χSE Ds; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW Dw; χNW] * C_northwest[χNW; χN] *
        removeunit(O, 1)[d2; d1 dstring] *
        t[d1 d2; Dn De Ds Dw] * E_north[χN Dn; χNE]
    return Vn, Vo
end

function end_correlator_numerator(
        j::CartesianIndex{2}, V::AbstractTensorMap{T, S, 3, 1},
        ρ::InfinitePEPO,
        O::MPOTensor, env::CTMRGEnv
    ) where {T, S}
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(j)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]
    t = twistdual(ρ[mod1(r, end), mod1(c, end)], 1:2)
    return @autoopt @tensor V[dstring χSW DW; χNW] *
        E_south[χSSE DS; χSW] * E_east[χNEE DE; χSEE] * E_north[χNW DN; χNNE] *
        C_northeast[χNNE; χNEE] * C_southeast[χSEE; χSSE] *
        t[d1 d2; DN DE DS DW] * removeunit(O, 4)[dstring d2; d1]
end

function end_correlator_denominator(
        j::CartesianIndex{2}, V::AbstractTensorMap{T, S, 1, 2}, env::CTMRGEnv
    ) where {T, S}
    r, c = Tuple(j)
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]
    return @autoopt @tensor V[χS; DE χN] * C_northeast[χN; χNE] *
        E_east[χNE DE; χSE] * C_southeast[χSE; χS]
end
