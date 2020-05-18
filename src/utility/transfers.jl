#used for contracting peps boundaries
#v sits "north"

# transfer with no operator
function crosstransfer(v,A::AbstractArray,B::AbstractArray)
    length(A)==length(B) || throw(ArgumentError("lengths differ"))
    reduce((v,(a,b)) -> crosstransfer(v,a,b),zip(A,B),init=v)
end

function crosstransfer(v::MPSKit.MPSBondTensor,a::MPSKit.GenericMPSTensor,b::MPSKit.GenericMPSTensor)
    @tensor v[-1;-2] := v[1,2]*a[2,3,4,-2]*b[-1,3,4,1]
end

# a singele peps tensor
function crosstransfer(v::MPSKit.GenericMPSTensor,O::AbstractArray,A::AbstractArray,B::AbstractArray;dir = North,bO=O)
    (length(O) == length(A)) && (length(O) == length(B)) && (length(O) == length(bO)) ||
        throw(ArgumentError("Lengths differ"))

    for (bo,o,a,b) in zip(bO,O,A,B)
        v = crosstransfer(v,o,a,b;dir = dir,bo = bo)
    end
    v
end

function crosstransfer(v::MPSKit.GenericMPSTensor,o::PEPSType,a::MPSKit.GenericMPSTensor,b::MPSKit.GenericMPSTensor;dir=North,bo::PEPSType=o)
    o = rotate_north(o,dir);bo = rotate_north(bo,dir);
    @tensor v[-1 -2 -3; -4]:=v[1,2,3,4]*a[4,5,6,-4]*o[7,-2,5,2,8]*conj(bo[9,-3,6,3,8])*b[-1,7,9,1]
end

#a double peps tensor
function crosstransfer(v::MPSKit.GenericMPSTensor,O1::AbstractArray,O2::AbstractArray,A::AbstractArray,B::AbstractArray;dir=North)
    (length(O1) == length(A)) && (length(O1) == length(B)) && (length(O1) == length(O2)) ||
        throw(ArgumentError("Lengths differ"))

    for (o1,o2,a,b) in zip(O1,O2,A,B)
        v = crosstransfer(v,o1,o2,a,b;dir = dir)
    end

    return v
end

function crosstransfer(v::MPSKit.GenericMPSTensor,o1::PEPSType,o2::PEPSType,a::MPSKit.GenericMPSTensor,b::MPSKit.GenericMPSTensor;dir=North)
    o1 = rotate_north(o1,dir); o2 = rotate_north(o2,dir);
    @tensor v[-1 -2 -3 -4 -5;-6]:=v[1,2,3,4,5,6]*a[6,7,8,-6]*b[-1,9,12,1]*
    o1[9,-2,10,2,11]*conj(o1[12,-3,13,3,11])*
    o2[10,-4,7,4,14]*conj(o2[13,-5,8,5,14])
end

#used for fixpoint equations
function MPSKit.transfer_left(v::M,o::T,a::M, b::M,bo::T=o) where T <: PEPSType where M <: MPSKit.GenericMPSTensor
    @tensor v[-1 -2 -3;-4]:=v[1,2,3,4]*a[4,5,6,-4]*o[2,7,-2,5,8]*conj(bo[3,9,-3,6,8])*conj(b[1,7,9,-1])
end

function MPSKit.transfer_right(v::M,o::T,a::M, b::M,bo::T=o) where T <: PEPSType where M <: MPSKit.GenericMPSTensor
    @tensor v[-1 -2 -3;-4]:=v[1,2,3,4]*a[-1,5,6,1]*o[-2,7,2,5,8]*conj(bo[-3,9,3,6,8])*conj(b[-4,7,9,4])
end
