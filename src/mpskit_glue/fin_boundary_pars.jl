#=
This file allows us to call approximate(init,line_of_peps_tensors,state,alg)
and have it work
=#
struct LineEnv{S,P<:PEPSType, C<:MPSKit.GenericMPSTensor} <: Cache
    above :: S #assumed not to change
    middle :: Vector{P}

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{C}
    rightenvs::Vector{C}
end

function MPSKit.params(below::S,middle::Vector{P},above::S,leftstart::C,rightstart::C) where {C <: MPSKit.GenericMPSTensor,P<:PEPSType,S <: Union{<:FiniteMPS,<:MPSComoving}}
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(above)
        push!(leftenvs,similar(leftstart))
        push!(rightenvs,similar(rightstart))
    end

    return LineEnv{S,P,C}(above,middle,similar.(below.AL[1:end]),similar.(below.AR[1:end]),leftenvs,reverse(rightenvs))
end

function MPSKit.params(below::S,squash::Tuple{Vector{P},S}) where {S <: FiniteMPS,P<:PEPSType}
    (middle,above) = squash;

    left_tracer = isomorphism(space(middle[1],1)',space(middle[1],1)')
    right_tracer = isomorphism(space(middle[end],3)',space(middle[end],3)')
    @tensor leftstart[-1 -2 -3; -4]:=l_LL(above)[-1,-4]*left_tracer[-2,-3]
    @tensor rightstart[-1 -2 -3; -4]:=r_RR(above)[-1;-4]*right_tracer[-2,-3]

    params(below,middle,above,leftstart,rightstart);
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function MPSKit.poison!(ca::LineEnv,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end


#rightenv[ind] will be contracteable with the tensor on site [ind]
function MPSKit.rightenv(ca::LineEnv,ind,state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = transfer_right(ca.rightenvs[j+1],ca.middle[j],ca.above.AR[j],state.AR[j])
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function MPSKit.leftenv(ca::LineEnv,ind,state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if a != nothing
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = transfer_left(ca.leftenvs[j],ca.middle[j],ca.above.AL[j],state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.leftenvs[ind]
end


function downproject2(pos::Int,below::S,sq::Tuple{Vector{P},S},pars) where {P<:PEPSType,S<:Union{FiniteMPS,MPSComoving}}
    (middle,above) = sq;

    @tensor toret[-1 -2 -3; -4 -5 -6]:=
    leftenv(pars,pos,below)[-1,4,2,1]*
    above.AC[pos][1,5,3,13]*
    conj(middle[pos][2,-3,14,3,6])*
    middle[pos][4,-2,15,5,6]*
    above.AR[pos+1][13,11,9,7]*
    rightenv(pars,pos+1,below)[7,10,8,-6]*
    conj(middle[pos+1][14,-5,8,9,12])*
    middle[pos+1][15,-4,10,11,12]
end

function downproject(pos::Int,below::S,sq::Tuple{Vector{P},S},pars) where {P<:PEPSType,S<:Union{FiniteMPS,MPSComoving}}
    (middle,above) = sq;
    @tensor toret[-1 -2 -3; -4]:=leftenv(pars,pos,below)[-1,7,8,9]*above.AC[pos][9,3,5,1]*rightenv(pars,pos,below)[1,2,4,-4]*middle[pos][7,-2,2,3,6]*conj(middle[pos][8,-3,4,5,6])
end
