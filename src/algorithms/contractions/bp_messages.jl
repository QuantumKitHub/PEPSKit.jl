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
