# Belief Propagation reduced density matrices
# -------------------------------------------

# NOTE: currently restricted to 1x1, 2x1, and 1x2 patches, since evaluating larger patches
# without including loop corrections tends to be a crude and not very useful appoximation.

function _contract_densitymatrix(inds::NTuple{N, Val}, state, env::BPEnv) where {N}
    sites = _patch_inds(inds)
    return throw(
        ArgumentError(
            "Cannot contract a $(_patch_shape_string(sites)) patch using a `BPEnv`;
            only 1x1, 2x1, 1x2 patches are supported."
        )
    )
end

function reduced_densitymatrix1x1(
        ind::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(ind)
    M_north = env[NORTH, row - 1, col]
    M_east = env[EAST, row, col + 1]
    M_south = env[SOUTH, row + 1, col]
    M_west = env[WEST, row, col - 1]

    @autoopt @tensor ρ[dt; db] :=
        ket[row, col][dt; DNt DEt DSt DWt] *
        conj(bra[row, col][db; DNb DEb DSb DWb]) *
        M_north[DNt; DNb] *
        M_east[DEt; DEb] *
        M_south[DSb; DSt] *
        M_west[DWb; DWt]

    return ρ / str(ρ)
end

function reduced_densitymatrix2x1(
        coord::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BPEnv
    )
    row, col = Tuple(coord)
    M_north = env[NORTH, row - 1, col]
    M_northeast = env[EAST, row, col + 1]
    M_southeast = env[EAST, row + 1, col + 1]
    M_south = env[SOUTH, row + 2, col]
    M_southwest = env[WEST, row + 1, col - 1]
    M_northwest = env[WEST, row, col - 1]

    @autoopt @tensor ρ[dNt dSt; dNb dSb] :=
        ket[row, col][dNt; DNt DNEt DMt DNWt] *
        ket[row + 1, col][dSt; DMt DSEt DSt DSWt] *
        conj(bra[row, col][dNb; DNb DNEb DMb DNWb]) *
        conj(bra[row + 1, col][dSb; DMb DSEb DSb DSWb]) *
        M_north[DNt; DNb] *
        M_northeast[DNEt; DNEb] *
        M_southeast[DSEt; DSEb] *
        M_south[DSb; DSt] *
        M_southwest[DSWb; DSWt] *
        M_northwest[DNWb; DNWt]

    return ρ / str(ρ)
end

function reduced_densitymatrix1x2(
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
    Ā_west = bra[row, col]
    A_east = ket[row, col + 1]
    Ā_east = bra[row, col + 1]

    @autoopt @tensor ρ[dWt dEt; dWb dEb] :=
        A_west[dWt; DNWt DMt DSWt DWt] *
        A_east[dEt; DNEt DEt DSEt DMt] *
        conj(Ā_west[dWb; DNWb DMb DSWb DWb]) *
        conj(Ā_east[dEb; DNEb DEb DSEb DMb]) *
        M_west[DWb; DWt] *
        M_northwest[DNWt; DNWb] *
        M_northeast[DNEt; DNEb] *
        M_east[DEt; DEb] *
        M_southeast[DSEb; DSEt] *
        M_southwest[DSWb; DSWt]

    return ρ / str(ρ)
end
