abstract type AbstractTransferMatrix end;

struct HorizontalTransferMatrix{T,S} <: AbstractTransferMatrix
    above::TensorMap{T,S}
    network
    below::TensorMap{T,S}
end

# TransferMatrix acting on a vector using *
Base.:*(vec, tm::AbstractTransferMatrix) = tm(vec);
# Base.:*(vec, tm::TransferMatrix) = flip(tm)(vec);

# TransferMatrix acting as a function
function (d::HorizontalTransferMatrix)(vec)
    return transfer_left(vec, d.above, d.network, d.below)
end;

function transfer_left(
    vec::TensorMap{T,S,4,1}, above, network::Tuple{PEPSTensor,PEPSTensor}, below
) where {T,S}
    @autoopt @tensor vec[χS DEt Dstring DEb; χN] :=
        vec[χ1 DWt Dstring DWb; χ4] *
        above[χ4 DNt DNb; χN] *
        network[1][d; DNt DEt DSt DWt] *
        conj(network[2][d; DNb DEb DSb DWb]) *
        below[χS DSt DSb; χ1]
    return vec
end

function transfer_left(
    vec::TensorMap{T,S,3,1}, above, network::Tuple{PEPSTensor,PEPSTensor}, below
) where {T,S}
    @autoopt @tensor vec[χS DEt DEb; χN] :=
        vec[χ1 DWt DWb; χ4] *
        above[χ4 DNt DNb; χN] *
        network[1][d; DNt DEt DSt DWt] *
        conj(network[2][d; DNb DEb DSb DWb]) *
        below[χS DSt DSb; χ1]
    return vec
end

function transfer_left(
    vec::TensorMap{T,S,3,1},
    above,
    network::Tuple{PEPSTensor,AbstractTensorMap{T,S,1,1},PEPSTensor},
    below,
) where {T,S}
    @autoopt @tensor vec[χS DEt DEb; χN] :=
        vec[χ1 DWt DWb; χ4] *
        above[χ4 DNt DNb; χN] *
        network[1][dt; DNt DEt DSt DWt] *
        network[2][db; dt] *
        conj(network[3][db; DNb DEb DSb DWb]) *
        below[χS DSt DSb; χ1]
    return vec
end

function transfer_left(
    vec::TensorMap{T,S,3,1},
    above,
    network::Tuple{PEPSTensor,AbstractTensorMap{T,S,1,2},PEPSTensor},
    below,
) where {T,S}
    @autoopt @tensor vec[χS DEt Dstring DEb; χN] :=
        vec[χ1 DWt DWb; χ4] *
        above[χ4 DNt DNb; χN] *
        network[1][dt; DNt DEt DSt DWt] *
        network[2][db; dt Dstring] *
        conj(network[3][db; DNb DEb DSb DWb]) *
        below[χS DSt DSb; χ1]
    return vec
end

function transfer_left(
    vec::TensorMap{T,S,4,1},
    above,
    network::Tuple{PEPSTensor,AbstractTensorMap{T,S,2,1},PEPSTensor},
    below,
) where {T,S}
    @autoopt @tensor vec[χS DEt DEb; χN] :=
        vec[χ1 DWt Dstring DWb; χ4] *
        above[χ4 DNt DNb; χN] *
        network[1][dt; DNt DEt DSt DWt] *
        network[2][Dstring db; dt] *
        conj(network[3][db; DNb DEb DSb DWb]) *
        below[χS DSt DSb; χ1]
    return vec
end

function start_left(above, middle, below)
    @autoopt @tensor vec[χ1 DWt DWb; χ4] :=
        above[χ3; χ4] * middle[χ2 DWt DWb; χ3] * below[χ1; χ2]
    return vec
end

function end_right(vec, above, middle, below)
    return @autoopt @tensor vec[χ5 DEt DEb; χ2] *
        above[χ2; χ3] *
        middle[χ3 DEt DEb; χ4] *
        below[χ4; χ5]
end
