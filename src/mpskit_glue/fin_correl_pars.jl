#=
This file allows us to call approximate(init,line_of_peps_tensors,state,alg)
and have it work
=#
struct HamLineEnv{S,P<:PEPSType, N<:NN,C<:MPSKit.GenericMPSTensor} <: Cache
    above :: S #assumed not to change

    opperator :: N
    middle :: Vector{P}

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{C}
    rightenvs::Vector{C}

    left_ham_envs::Vector{C}
    right_ham_envs::Vector{C}
end

function MPSKit.environments(below::S,middle::Vector{P},o::N,above::S,leftstart::C,rightstart::C) where {N<:NN,S <: Union{<:FiniteMPS,<:MPSComoving},C <: MPSKit.GenericMPSTensor,P<:PEPSType}
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    hamleftenvs = [0*leftstart]
    hamrightenvs = [0*rightstart]

    for i in 1:length(above)
        push!(leftenvs,similar(leftstart))
        push!(rightenvs,similar(rightstart))

        push!(hamleftenvs,similar(leftstart))
        push!(hamrightenvs,similar(rightstart))
    end

    return HamLineEnv{S,P,N,C}(above,o,middle,similar.(below.AL[1:end]),similar.(below.AR[1:end]),leftenvs,reverse(rightenvs),hamleftenvs,reverse(hamrightenvs))
end

function MPSKit.environments(below::S,squash::Tuple{Vector{P},O,S}) where {S <: FiniteMPS,O<:NN,P<:PEPSType}
    (middle,opp,above) = squash;

    left_tracer = isomorphism(space(middle[1],1)',space(middle[1],1)')
    right_tracer = isomorphism(space(middle[end],3)',space(middle[end],3)')
    @tensor leftstart[-1 -2 -3; -4]:=l_LL(above)[-1,-4]*left_tracer[-2,-3]
    @tensor rightstart[-1 -2 -3; -4]:=r_RR(above)[-1;-4]*right_tracer[-2,-3]
    params(below,middle,opp,above,leftstart,rightstart);
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function MPSKit.poison!(ca::HamLineEnv,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function MPSKit.rightenv(ca::HamLineEnv,ind,state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = transfer_right(ca.rightenvs[j+1],ca.middle[j],ca.above.AR[j],state.AR[j])

            ca.right_ham_envs[j] = transfer_right(ca.right_ham_envs[j+1],ca.middle[j],ca.above.AR[j],state.AR[j])
            if j<length(state)
                @tensor car1[-1 -2 -3;-4]:=conj(state.AR[j][-4 -2 -3 -1])
                @tensor car2[-1 -2 -3;-4]:=conj(state.AR[j+1][-4 -2 -3 -1])

                ca.right_ham_envs[j]+=hamtransfer(ca.above.AR[j],ca.above.AR[j+1],car2,car1,ca.rightenvs[j+2],rotate_north(ca.middle[j+1],East),rotate_north(ca.middle[j],East),ca.opperator)
            end

            ca.rdependencies[j] = state.AR[j]
        end
    end

    return (ca.rightenvs[ind+1],ca.right_ham_envs[ind+1])
end

function MPSKit.leftenv(ca::HamLineEnv,ind,state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if a != nothing
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = transfer_left(ca.leftenvs[j],ca.middle[j],ca.above.AL[j],state.AL[j])

            ca.left_ham_envs[j+1] = transfer_left(ca.left_ham_envs[j],ca.middle[j],ca.above.AL[j],state.AL[j])
            if j > 1
                @tensor car1[-1 -2 -3;-4]:=conj(state.AL[j][-4 -2 -3 -1])
                @tensor car2[-1 -2 -3;-4]:=conj(state.AL[j-1][-4 -2 -3 -1])

                ca.left_ham_envs[j+1] += hamtransfer(car1,car2,ca.above.AL[j-1],ca.above.AL[j],ca.leftenvs[j-1],rotate_north(ca.middle[j-1],West),rotate_north(ca.middle[j],West),ca.opperator)
            end

            ca.ldependencies[j] = state.AL[j]
        end
    end

    return (ca.leftenvs[ind],ca.left_ham_envs[ind])
end


function downproject2(pos::Int,below::S,sq::Tuple{Vector{P},O,S},pars::HamLineEnv) where {P<:PEPSType,S<:Union{FiniteMPS,MPSComoving},O<:NN}
    (middle,opp,above) = sq;

    (le,hle) = leftenv(pars,pos,below)
    (re,hre) = rightenv(pars,pos+1,below)


    @tensor toret[-1 -2 -3; -4 -5 -6]:=
    hle[-1,4,2,1]*
    above.AC[pos][1,5,3,13]*
    conj(middle[pos][2,-3,14,3,6])*
    middle[pos][4,-2,15,5,6]*
    above.AR[pos+1][13,11,9,7]*
    re[7,10,8,-6]*
    conj(middle[pos+1][14,-5,8,9,12])*
    middle[pos+1][15,-4,10,11,12]

    @tensor toret[-1 -2 -3; -4 -5 -6]+=
    le[-1,4,2,1]*
    above.AC[pos][1,5,3,13]*
    conj(middle[pos][2,-3,14,3,6])*
    middle[pos][4,-2,15,5,6]*
    above.AR[pos+1][13,11,9,7]*
    hre[7,10,8,-6]*
    conj(middle[pos+1][14,-5,8,9,12])*
    middle[pos+1][15,-4,10,11,12]

    @tensor toret[-1 -2 -3; -4 -5 -6]+=
    le[-1,15,16,17]*
    above.AC[pos][17,13,10,1]*
    conj(middle[pos][16,-3,9,10,11])*
    middle[pos][15,-2,12,13,14]*
    above.AR[pos+1][1,4,6,2]*
    re[2,3,5,-6]*
    conj(middle[pos+1][9,-5,5,6,7])*
    middle[pos+1][12,-4,3,4,8]*
    opp.o[7,8,11,14]

    if pos > 1
        (le,_) = leftenv(pars,pos-1,below);
        @tensor toret[-1 -2 -3;-4 -5 -6]+=
        le[24,11,9,8]*
        above.AL[pos-1][8,12,10,7]*
        above.AC[pos][7,19,16,21]*
        above.AR[pos+1][21,4,5,6]*
        re[6,2,1,-6]*
        conj(below.AL[pos-1][24,25,26,-1])*
        middle[pos-1][11,25,18,12,14]*
        conj(middle[pos-1][9,26,15,10,13])*
        middle[pos][18,-2,22,19,20]*
        conj(middle[pos][15,-3,23,16,17])*
        middle[pos+1][22,-4,2,4,3]*
        conj(middle[pos+1][23,-5,1,5,3])*
        opp.o[13,14,17,20]

    end

    if pos+1 < length(below)
        (le,_) = leftenv(pars,pos,below)
        (re,_) = rightenv(pars,pos+2,below);

        @tensor toret[-1 -2 -3;-4 -5 -6]+=
        le[-1,2,1,4]*
        above.AC[pos][4,5,6,21]*
        above.AR[pos+1][21,19,12,7]*
        above.AR[pos+2][7,16,10,8]*
        re[8,15,9,26]*
        conj(below.AR[pos+2][-6,24,25,26])*
        middle[pos][2,-2,22,5,3]*
        conj(middle[pos][1,-3,23,6,3])*
        middle[pos+1][22,-4,18,19,20]*
        conj(middle[pos+1][23,-5,11,12,13])*
        middle[pos+2][18,24,15,16,17]*
        conj(middle[pos+2][11,25,9,10,14])*
        opp.o[13,20,14,17]
    end

    toret
end

function downproject(pos::Int,below::S,sq::Tuple{Vector{P},O,S},pars::HamLineEnv) where {P<:PEPSType,S<:Union{FiniteMPS,MPSComoving},O<:NN}
    #not yet optimal contraction order (or tested for that matter)
    (middle,opp,above) = sq;

    (le,hle) = leftenv(pars,pos,below)
    (re,hre) = rightenv(pars,pos,below)
    @tensor toret[-1 -2 -3; -4]:=hle[-1,7,8,9]*above.AC[pos][9,3,5,1]*re[1,2,4,-4]*middle[pos][7,-2,2,3,6]*conj(middle[pos][8,-3,4,5,6])
    @tensor toret[-1 -2 -3; -4]+=le[-1,7,8,9]*above.AC[pos][9,3,5,1]*hre[1,2,4,-4]*middle[pos][7,-2,2,3,6]*conj(middle[pos][8,-3,4,5,6])

    if pos > 1
        (le,_) = leftenv(pars,pos-1,below)

        @tensor toret[-1 -2 -3;-4]+=le[18,15,16,17]*above.AC[pos-1][17,6,13,1]*above.AR[pos][1,4,10,2]*re[2,3,9,-4]*conj(below.AL[pos-1][18,19,20,-1])*
            middle[pos-1][15,19,5,6,7]*conj(middle[pos-1][16,20,12,13,14])*middle[pos][5,-2,3,4,8]*conj(middle[pos][12,-3,9,10,11])*
            opp.o[14,7,11,8]
    end

    if pos<length(below)
        (le,_) = leftenv(pars,pos,below)
        (re,_) = rightenv(pars,pos+1,below)

        @tensor toret[-1 -2 -3;-4]+=le[-1,2,5,3]*above.AC[pos][3,4,1,18]*above.AR[pos+1][18,10,15,8]*re[8,9,14,12]*conj(below.AR[pos+1][-4,11,13,12])*
            middle[pos][2,-2,20,4,7]*conj(middle[pos][5,-3,19,1,6])*middle[pos+1][20,11,9,10,17]*conj(middle[pos+1][19,13,14,15,16])*
            opp.o[6,7,16,17]
    end
    toret
end
