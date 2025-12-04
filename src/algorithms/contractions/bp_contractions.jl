const PEPSMessage = AbstractTensorMap{<:Any, <:Any, 1, 1}

# Belief Propagation Updates
# --------------------------
function contract_north_message(
        A::PEPSSandwich, M_west::PEPSMessage, M_north::PEPSMessage, M_east::PEPSMessage
    )
    return @autoopt @tensor begin
        M_north′[DSt; DSb] :=
            ket(A)[d; DNt DEt DSt DWt] *
            conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_west[DWt; DWb] *
            M_north[DNt; DNb] *
            M_east[DEt; DEb]
    end
end
function contract_east_message(
        A::PEPSSandwich, M_north::PEPSMessage, M_east::PEPSMessage, M_south::PEPSMessage
    )
    return @autoopt @tensor begin
        M_east′[DWt; DWb] :=
            ket(A)[d; DNt DEt DSt DWt] *
            conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_north[DNt; DNb] *
            M_east[DEt; DEb] *
            M_south[DSt; DSb]
    end
end
function contract_south_message(
        A::PEPSSandwich, M_east::PEPSMessage, M_south::PEPSMessage, M_west::PEPSMessage
    )
    return @autoopt @tensor begin
        M_south′[DNt; DNb] :=
            ket(A)[d; DNt DEt DSt DWt] *
            conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_east[DEt; DEb] *
            M_south[DSt; DSb] *
            M_west[DWt; DWb]
    end
end
function contract_west_message(
        A::PEPSSandwich, M_south::PEPSMessage, M_west::PEPSMessage, M_north::PEPSMessage
    )
    return @autoopt @tensor begin
        M_west′[DEt; DEb] :=
            ket(A)[d; DNt DEt DSt DWt] *
            conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_south[DSt; DSb] *
            M_west[DWt; DWb] *
            M_north[DNt; DNb]
    end
end

absorb_north_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N' E S W] := A[d; N E S W] * M[N; N']
absorb_east_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N E' S W] := A[d; N E S W] * M[E; E']
absorb_south_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N E S' W] := A[d; N E S W] * M[S; S']
absorb_west_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N E S W'] := A[d; N E S W] * M[W; W']

# Belief Propagation Expectation values
# -------------------------------------
function MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, env::BPEnv)
    checklattice(peps, O)
    term_vals = dtmap([O.terms...]) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        contract_local_operator(inds, operator, peps, peps, env) /
            contract_local_norm(inds, peps, peps, env)
    end
    return sum(term_vals)
end

function contract_local_operator(
        inds::NTuple{1, CartesianIndex{2}},
        O::AbstractTensorMap{<:Any, <:Any, 1, 1},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    row, col = Tuple(only(inds))
    M_north = env.messages[NORTH, _prev(row, end), mod1(col, end)]
    M_east = env.messages[EAST, mod1(row, end), _next(col, end)]
    M_south = env.messages[SOUTH, _next(row, end), mod1(col, end)]
    M_west = env.messages[WEST, mod1(row, end), _prev(col, end)]

    return @autoopt @tensor begin
        ket[mod1(row, end), mod1(col, end)][dt; DNt DEt DSt DWt] *
            conj(bra[mod1(row, end), mod1(col, end)][db; DNb DEb DSb DWb]) *
            O[db; dt] *
            M_north[DNt; DNb] *
            M_east[DEt; DEb] *
            M_south[DSt; DSb] *
            M_west[DWt; DWb]
    end
end

function contract_local_norm(
        inds::NTuple{1, CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(only(inds))
    M_north = env.messages[NORTH, _prev(row, end), mod1(col, end)]
    M_east = env.messages[EAST, mod1(row, end), _next(col, end)]
    M_south = env.messages[SOUTH, _next(row, end), mod1(col, end)]
    M_west = env.messages[WEST, mod1(row, end), _prev(col, end)]

    return @autoopt @tensor begin
        ket[mod1(row, end), mod1(col, end)][d; DNt DEt DSt DWt] *
            conj(bra[mod1(row, end), mod1(col, end)][d; DNb DEb DSb DWb]) *
            M_north[DNt; DNb] *
            M_east[DEt; DEb] *
            M_south[DSt; DSb] *
            M_west[DWt; DWb]
    end
end

function contract_local_operator(
        inds::NTuple{2, CartesianIndex{2}},
        O::AbstractTensorMap{<:Any, <:Any, 2, 2},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    ind_relative = inds[2] - inds[1]
    return if ind_relative == CartesianIndex(1, 0)
        contract_vertical_operator(inds[1], O, ket, bra, env)
    elseif ind_relative == CartesianIndex(0, 1)
        contract_horizontal_operator(inds[1], O, ket, bra, env)
    else
        error("Not implemented")
    end
end

function contract_vertical_operator(
        coord::CartesianIndex{2},
        O::AbstractTensorMap{<:Any, <:Any, 2, 2},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    row, col = Tuple(coord)
    M_north = env.messages[NORTH, _prev(row, end), mod1(col, end)]
    M_northeast = env.messages[EAST, mod1(row, end), _next(col, end)]
    M_southeast = env.messages[EAST, _next(row, end), _next(col, end)]
    M_south = env.messages[SOUTH, mod1(row + 2, end), mod1(col, end)]
    M_southwest = env.messages[WEST, _next(row, end), _prev(col, end)]
    M_northwest = env.messages[WEST, mod1(row, end), _prev(col, end)]

    return @autoopt @tensor ket[mod1(row, end), mod1(col, end)][dNt; DNt DNEt DMt DNWt] *
        ket[_next(row, end), mod1(col, end)][dSt; DMt DSEt DSt DSWt] *
        conj(bra[mod1(row, end), mod1(col, end)][dNb; DNb DNEb DMb DNWb]) *
        conj(bra[_next(row, end), mod1(col, end)][dSb; DMb DSEb DSb DSWb]) *
        M_north[DNt; DNb] *
        M_northeast[DNEt; DNEb] *
        M_southeast[DSEt; DSEb] *
        M_south[DSt; DSb] *
        M_southwest[DSWt; DSWb] *
        M_northwest[DNWt; DNWb] *
        O[dNb dSb; dNt dSt]
end

function contract_horizontal_operator(
        coord::CartesianIndex{2},
        O::AbstractTensorMap{<:Any, <:Any, 2, 2},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    row, col = Tuple(coord)
    M_west = env.messages[WEST, mod1(row, end), _prev(col, end)]
    M_northwest = env.messages[NORTH, _prev(row, end), mod1(col, end)]
    M_northeast = env.messages[NORTH, _prev(row, end), _next(col, end)]
    M_east = env.messages[EAST, mod1(row, end), mod1(col + 2, end)]
    M_southeast = env.messages[SOUTH, _next(row, end), _next(col, end)]
    M_southwest = env.messages[SOUTH, _next(row, end), mod1(col, end)]
    A_west = ket[mod1(row, end), mod1(col, end)]
    Ā_west = bra[mod1(row, end), mod1(col, end)]
    A_east = ket[mod1(row, end), _next(col, end)]
    Ā_east = bra[mod1(row, end), _next(col, end)]

    return @autoopt @tensor begin
        A_west[dWt; DNWt DMt DSWt DWt] *
            A_east[dEt; DNEt DEt DSEt DMt] *
            conj(Ā_west[dWb; DNWb DMb DSWb DWb]) *
            conj(Ā_east[dEb; DNEb DEb DSEb DMb]) *
            M_west[DWt; DWb] *
            M_northwest[DNWt; DNWb] *
            M_northeast[DNEt; DNEb] *
            M_east[DEt; DEb] *
            M_southeast[DSEt; DSEb] *
            M_southwest[DSWt; DSWb] *
            O[dWb dEb; dWt dEt]
    end
end

function contract_local_norm(
        inds::NTuple{2, CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    ind_relative = inds[2] - inds[1]
    return if ind_relative == CartesianIndex(1, 0)
        contract_vertical_norm(inds[1], ket, bra, env)
    elseif ind_relative == CartesianIndex(0, 1)
        contract_horizontal_norm(inds[1], ket, bra, env)
    else
        error("Not implemented")
    end
end

function contract_vertical_norm(
        coord::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(coord)
    M_north = env.messages[NORTH, _prev(row, end), mod1(col, end)]
    M_northeast = env.messages[EAST, mod1(row, end), _next(col, end)]
    M_southeast = env.messages[EAST, _next(row, end), _next(col, end)]
    M_south = env.messages[SOUTH, mod1(row + 2, end), mod1(col, end)]
    M_southwest = env.messages[WEST, _next(row, end), _prev(col, end)]
    M_northwest = env.messages[WEST, mod1(row, end), _prev(col, end)]

    return @autoopt @tensor ket[mod1(row, end), mod1(col, end)][dN; DNt DNEt DMt DNWt] *
        ket[_next(row, end), mod1(col, end)][dS; DMt DSEt DSt DSWt] *
        conj(bra[mod1(row, end), mod1(col, end)][dN; DNb DNEb DMb DNWb]) *
        conj(bra[_next(row, end), mod1(col, end)][dS; DMb DSEb DSb DSWb]) *
        M_north[DNt; DNb] *
        M_northeast[DNEt; DNEb] *
        M_southeast[DSEt; DSEb] *
        M_south[DSt; DSb] *
        M_southwest[DSWt; DSWb] *
        M_northwest[DNWt; DNWb]
end

function contract_horizontal_norm(
        coord::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(coord)

    M_west = env.messages[WEST, mod1(row, end), _prev(col, end)]
    M_northwest = env.messages[NORTH, _prev(row, end), mod1(col, end)]
    M_northeast = env.messages[NORTH, _prev(row, end), _next(col, end)]
    M_east = env.messages[EAST, mod1(row, end), mod1(col + 2, end)]
    M_southeast = env.messages[SOUTH, _next(row, end), _next(col, end)]
    M_southwest = env.messages[SOUTH, _next(row, end), mod1(col, end)]

    A_west = ket[mod1(row, end), mod1(col, end)]
    Ā_west = bra[mod1(row, end), mod1(col, end)]
    A_east = ket[mod1(row, end), _next(col, end)]
    Ā_east = bra[mod1(row, end), _next(col, end)]

    return @autoopt @tensor begin
        A_west[dW; DNWt DMt DSWt DWt] *
            A_east[dE; DNEt DEt DSEt DMt] *
            conj(Ā_west[dW; DNWb DMb DSWb DWb]) *
            conj(Ā_east[dE; DNEb DEb DSEb DMb]) *
            M_west[DWt; DWb] *
            M_northwest[DNWt; DNWb] *
            M_northeast[DNEt; DNEb] *
            M_east[DEt; DEb] *
            M_southeast[DSEt; DSEb] *
            M_southwest[DSWt; DSWb]
    end
end
