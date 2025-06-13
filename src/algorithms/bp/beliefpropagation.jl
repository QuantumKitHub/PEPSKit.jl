@kwdef struct BeliefPropagation
    maxiter::Int = 10
end

function bp_iteration(network::InfiniteSquareNetwork, env::BPEnv, alg::BeliefPropagation)
    messages = similar(env.messages)
    for I in eachindex(IndexCartesian(), messages)
        dir, row, col = Tuple(I)
        if dir == NORTH
            row += 1
        elseif dir == EAST
            col += 1
        elseif dir == SOUTH
            row -= 1
        elseif dir == WEST
            col -= 1
        end
        messages[dir, mod1(row, end), mod1(col, end)] = update_message(I, network, env)
    end
    return BPEnv(messages)
end

function update_message(I::CartesianIndex{3}, network::InfiniteSquareNetwork, env::BPEnv)
    dir, row, col = Tuple(I)

    A = network[row, col]
    dir == SOUTH || (M_north = env.messages[NORTH, _prev(row, end), col])
    dir == WEST || (M_east = env.messages[EAST, row, _next(col, end)])
    dir == NORTH || (M_south = env.messages[SOUTH, _next(row, end), col])
    dir == EAST || (M_west = env.messages[WEST, row, _prev(col, end)])

    return if dir == NORTH
        contract_north_message(A, M_west, M_north, M_east)
    elseif dir == EAST
        contract_east_message(A, M_north, M_east, M_south)
    elseif dir == SOUTH
        contract_south_message(A, M_east, M_south, M_west)
    elseif dir == WEST
        contract_west_message(A, M_south, M_west, M_north)
    else
        throw(ArgumentError("Invalid direction $dir"))
    end
end

const PEPSMessage = AbstractTensorMap{<:Any,<:Any,1,1}

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
