const PEPSMessage = AbstractTensorMap{<:Any, <:Any, 1, 1}

# Belief Propagation Updates
# --------------------------
function contract_north_message(
        A::PEPSSandwich, M_west::PEPSMessage, M_north::PEPSMessage, M_east::PEPSMessage
    )
    return @autoopt @tensor begin
        M_north′[DSt; DSb] :=
            ket(A)[d; DNt DEt DSt DWt] * conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_west[DWb; DWt] * M_north[DNt; DNb] * M_east[DEt; DEb]
    end
end
function contract_east_message(
        A::PEPSSandwich, M_north::PEPSMessage, M_east::PEPSMessage, M_south::PEPSMessage
    )
    return @autoopt @tensor begin
        M_east′[DWt; DWb] :=
            ket(A)[d; DNt DEt DSt DWt] * conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_north[DNt; DNb] * M_east[DEt; DEb] * M_south[DSb; DSt]
    end
end
function contract_south_message(
        A::PEPSSandwich, M_east::PEPSMessage, M_south::PEPSMessage, M_west::PEPSMessage
    )
    return @autoopt @tensor begin
        M_south′[DNb; DNt] :=
            ket(A)[d; DNt DEt DSt DWt] * conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_east[DEt; DEb] * M_south[DSb; DSt] * M_west[DWb; DWt]
    end
end
function contract_west_message(
        A::PEPSSandwich, M_south::PEPSMessage, M_west::PEPSMessage, M_north::PEPSMessage
    )
    return @autoopt @tensor begin
        M_west′[DEb; DEt] :=
            ket(A)[d; DNt DEt DSt DWt] * conj(bra(A)[d; DNb DEb DSb DWb]) *
            M_south[DSb; DSt] * M_west[DWb; DWt] * M_north[DNt; DNb]
    end
end

absorb_north_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N' E S W] := A[d; N E S W] * M[N; N']
absorb_east_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N E' S W] := A[d; N E S W] * M[E; E']
absorb_south_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N E S' W] := A[d; N E S W] * M[S'; S]
absorb_west_message(A::PEPSTensor, M::PEPSMessage) =
    @tensor A′[d; N E S W'] := A[d; N E S W] * M[W'; W]

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
        inds::Vector{CartesianIndex{2}},
        O::AbstractTensorMap,
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    length(inds) == 1 && return contract_local_operator1x1(only(inds), O, ket, bra, env)

    if length(inds) == 2
        ind_relative = inds[2] - inds[1]
        if ind_relative == CartesianIndex(1, 0)
            return contract_local_operator2x1(inds[1], O, ket, bra, env)
        elseif ind_relative == CartesianIndex(0, 1)
            return contract_local_operator1x2(inds[1], O, ket, bra, env)
        end
    end
    error("No implementation for contractions for BP environments with $inds")
end
function contract_local_norm(
        inds::Vector{CartesianIndex{2}},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    length(inds) == 1 && return contract_local_norm1x1(only(inds), ket, bra, env)

    if length(inds) == 2
        ind_relative = inds[2] - inds[1]
        if ind_relative == CartesianIndex(1, 0)
            return contract_local_norm2x1(inds[1], ket, bra, env)
        elseif ind_relative == CartesianIndex(0, 1)
            return contract_local_norm1x2(inds[1], ket, bra, env)
        end
    end
    error("No implementation for contractions for BP environments with $inds")
end

function contract_local_operator1x1(
        ind::CartesianIndex{2},
        O::AbstractTensorMap{<:Any, <:Any, 1, 1},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    row, col = Tuple(ind)
    M_north = env[NORTH, row - 1, col]
    M_east = env[EAST, row, col + 1]
    M_south = env[SOUTH, row + 1, col]
    M_west = env[WEST, row, col - 1]

    return @autoopt @tensor begin
        ket[row, col][dt; DNt DEt DSt DWt] *
            conj(bra[row, col][db; DNb DEb DSb DWb]) *
            O[db; dt] *
            M_north[DNt; DNb] *
            M_east[DEt; DEb] *
            M_south[DSb; DSt] *
            M_west[DWb; DWt]
    end
end

function contract_local_norm1x1(
        ind::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(ind)
    M_north = env[NORTH, row - 1, col]
    M_east = env[EAST, row, col + 1]
    M_south = env[SOUTH, row + 1, col]
    M_west = env[WEST, row, col - 1]

    return @autoopt @tensor begin
        ket[row, col][d; DNt DEt DSt DWt] *
            conj(bra[row, col][d; DNb DEb DSb DWb]) *
            M_north[DNt; DNb] *
            M_east[DEt; DEb] *
            M_south[DSb; DSt] *
            M_west[DWb; DWt]
    end
end

function contract_local_operator2x1(
        coord::CartesianIndex{2},
        O::AbstractTensorMap{<:Any, <:Any, 2, 2},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    row, col = Tuple(coord)
    M_north = env[NORTH, row - 1, col]
    M_northeast = env[EAST, row, col + 1]
    M_southeast = env[EAST, row + 1, col + 1]
    M_south = env[SOUTH, row + 2, col]
    M_southwest = env[WEST, row + 1, col - 1]
    M_northwest = env[WEST, row, col - 1]

    return @autoopt @tensor ket[row, col][dNt; DNt DNEt DMt DNWt] *
        ket[row + 1, col][dSt; DMt DSEt DSt DSWt] *
        conj(bra[row, col][dNb; DNb DNEb DMb DNWb]) *
        conj(bra[row + 1, col][dSb; DMb DSEb DSb DSWb]) *
        M_north[DNt; DNb] *
        M_northeast[DNEt; DNEb] *
        M_southeast[DSEt; DSEb] *
        M_south[DSb; DSt] *
        M_southwest[DSWb; DSWt] *
        M_northwest[DNWb; DNWt] *
        O[dNb dSb; dNt dSt]
end

function contract_local_operator1x2(
        coord::CartesianIndex{2},
        O::AbstractTensorMap{<:Any, <:Any, 2, 2},
        ket::InfinitePEPS,
        bra::InfinitePEPS,
        env::BPEnv,
    )
    row, col = Tuple(coord)
    M_west = env[WEST, row, col - 1]
    M_northwest = env[NORTH, row - 1, col]
    M_northeast = env[NORTH, row - 1, col + 1]
    M_east = env[EAST, row, col + 2]
    M_southeast = env[SOUTH, row + 1, col + 1]
    M_southwest = env[SOUTH, row + 1, col]
    A_west = ket[row, col]
    Ā_west = bra[row, col]
    A_east = ket[row, col + 1]
    Ā_east = bra[row, col + 1]

    return @autoopt @tensor begin
        A_west[dWt; DNWt DMt DSWt DWt] *
            A_east[dEt; DNEt DEt DSEt DMt] *
            conj(Ā_west[dWb; DNWb DMb DSWb DWb]) *
            conj(Ā_east[dEb; DNEb DEb DSEb DMb]) *
            M_west[DWb; DWt] *
            M_northwest[DNWt; DNWb] *
            M_northeast[DNEt; DNEb] *
            M_east[DEt; DEb] *
            M_southeast[DSEb; DSEt] *
            M_southwest[DSWb; DSWt] *
            O[dWb dEb; dWt dEt]
    end
end

function contract_local_norm2x1(
        coord::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(coord)
    M_north = env[NORTH, row - 1, col]
    M_northeast = env[EAST, row, col + 1]
    M_southeast = env[EAST, row + 1, col + 1]
    M_south = env[SOUTH, row + 2, col]
    M_southwest = env[WEST, row + 1, col - 1]
    M_northwest = env[WEST, row, col - 1]

    return @autoopt @tensor ket[row, col][dN; DNt DNEt DMt DNWt] *
        ket[row + 1, col][dS; DMt DSEt DSt DSWt] *
        conj(bra[row, col][dN; DNb DNEb DMb DNWb]) *
        conj(bra[row + 1, col][dS; DMb DSEb DSb DSWb]) *
        M_north[DNt; DNb] *
        M_northeast[DNEt; DNEb] *
        M_southeast[DSEt; DSEb] *
        M_south[DSb; DSt] *
        M_southwest[DSWb; DSWt] *
        M_northwest[DNWb; DNWt]
end

function contract_local_norm1x2(
        coord::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(coord)

    M_west = env[WEST, row, col - 1]
    M_northwest = env[NORTH, row - 1, col]
    M_northeast = env[NORTH, row - 1, col + 1]
    M_east = env[EAST, row, col + 2]
    M_southeast = env[SOUTH, row + 1, col + 1]
    M_southwest = env[SOUTH, row + 1, col]

    A_west = ket[row, col]
    Ā_west = bra[row, col]
    A_east = ket[row, col + 1]
    Ā_east = bra[row, col + 1]

    return @autoopt @tensor begin
        A_west[dW; DNWt DMt DSWt DWt] *
            A_east[dE; DNEt DEt DSEt DMt] *
            conj(Ā_west[dW; DNWb DMb DSWb DWb]) *
            conj(Ā_east[dE; DNEb DEb DSEb DMb]) *
            M_west[DWb; DWt] *
            M_northwest[DNWt; DNWb] *
            M_northeast[DNEt; DNEb] *
            M_east[DEt; DEb] *
            M_southeast[DSEb; DSEt] *
            M_southwest[DSWb; DSWt]
    end
end
