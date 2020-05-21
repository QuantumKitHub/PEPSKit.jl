mutable struct NNHamChannels{E<:InfEnvManager,B,C,O<:NN} <: Cache
    opperator :: O
    envm::E

    lines::B
    ts::C
end

#generate bogus data
function MPSKit.params(peps::InfPEPS,opperator::NN;kwargs ...)
    #this should rather by a 4periodic array of arrays...
    lines = PeriodicArray{Any}(undef,4);
    ts = PeriodicArray{Any}(undef,4);

    pars = NNHamChannels(opperator,params(peps;kwargs...),lines,ts);

    return MPSKit.recalculate!(pars,peps;kwargs...)
end

#recalculate everything
function MPSKit.recalculate!(prevenv::NNHamChannels,peps::InfPEPS;kwargs...)
    MPSKit.recalculate!(prevenv.envm,peps;kwargs...);

    prevenv.lines = PeriodicArray(map(Dirs) do dir
        north_nncontr_impl(rotate_north(prevenv.envm,dir),rotate_north(prevenv.opperator,dir))
    end)

    prevenv.ts = PeriodicArray(map(Dirs) do dir
        north_nntchannel_impl(  rotate_north(prevenv.envm,dir),
                                circshift(prevenv.lines,4-dir),
                                rotate_north(prevenv.opperator,dir));
    end)

    prevenv
end

# j == the collumn
function north_inf_sum_RL(v,man,j)
    downfp = man.fp1LR[South,0,j];
    upfp = man.fp1RL[North,1,j];
    @tensor v[-1 -2 -3; -4]-=downfp[4,2,3,1]*v[1,2,3,4]*upfp[-1,-2,-3,-4]

    (v,chist) = linsolve(v,v,GMRES()) do x
        y=crosstransfer(x,man.peps[:,j],man.AR[East,:,j],man.AL[West,:,j])
        @tensor y[-1 -2 -3; -4]-=downfp[4,2,3,1]*y[1,2,3,4]*(upfp)[-1,-2,-3,-4]

        y=x-y
    end
    chist.converged == 0 && @info "failed to converge _ RL $(chist.normres)"

    return v
end

function north_inf_sum_LR(v,man,j)
    downfp = man.fp1RL[South,0,j];
    upfp = man.fp1LR[North,1,j];
    @tensor v[-1 -2 -3; -4]-=downfp[4,2,3,1]*v[1,2,3,4]*upfp[-1,-2,-3,-4]

    (v,chist) = linsolve(v,v,GMRES()) do x
        y=crosstransfer(x,man.peps[:,j],man.AL[East,:,j],man.AR[West,:,j])
        @tensor y[-1 -2 -3; -4]-=downfp[4,2,3,1]*y[1,2,3,4]*(upfp)[-1,-2,-3,-4]

        y=x-y
    end
    chist.converged == 0 && @info "failed to converge _ LR $(chist.normres)"

    return v
end

function north_linecontr_local(cnt,man::InfEnvManager,nn::NN,i,j)
    cnt = crosstransfer(cnt,man.peps[i,j],man.AL[East,i,j],man.AR[West,i,j])

    cnt += hamtransfer(
        man.AR[West,i,j],
        man.AR[West,i-1,j],
        man.AL[East,i-1,j],
        man.AL[East,i,j],
        man.fp1LR[North,i-1,j],
        man.peps[i-1,j],
        man.peps[i,j],
        nn)

    return cnt
end
#sum of all contributions north of (i,j)
function north_nncontr_impl(man::InfEnvManager,nn::NN)
    (nr,nc) = size(man.peps)

    # sums[i] = all contributions above row 1; collumn i
    sums = map(1:nc) do j
        cnt = TensorMap(zeros,ComplexF64,   space(man.AL[West,1,j],4)'*
                                            space(man.peps[1,j],North)'*
                                            space(man.peps[1,j],North),
                                            space(man.AL[East,1,j],1))

        for i in 1:nr
            cnt = north_linecontr_local(cnt,man,nn,i,j)
        end

        #infinite sum
        ecnt = north_inf_sum_LR(cnt,man,j);
    end

    # of course, we also want to know the sum of contributions above other unit cells ...
    lines = PeriodicArray{eltype(sums),2}(undef,nr,nc);
    for (i,j) in Iterators.product(1:nr,1:nc)
        if i == 1
            lines[i,j] = sums[j]
        else
            lines[i,j] = north_linecontr_local(lines[i-1,j],man,nn,i-1,j)
        end
    end


    return lines
end

#the T - contribution above (i,j)
#while cryptic - I can explain it using a piece of paper?
function north_nntchannel_loctransfer(cnt,man::InfEnvManager,lines,nn::NN,i,j)
    cnt = crosstransfer(cnt,man.peps[i,j],man.AR[East,i,j],man.AL[West,i,j])

    #collect west and east contributions from lines
    westcoords = rotate_north((i,j),size(man.peps),West);
    eastcoords = rotate_north((i,j),size(man.peps),East);
    cwcontr = lines[West][westcoords...];
    cecontr = lines[East][eastcoords...];

    # "add west contribution"
    @tensor cnt[-1 -2 -3;-4] += man.corner[SouthWest,i,j][-1,2]*
        cwcontr[2,10,11,1]*
        man.corner[NorthWest,i,j][1,9]*
        man.fp1LR[North,i,j][9,5,7,3]*
        man.AC[East,i,j][3,4,6,-4]*
        man.peps[i,j][10,-2,4,5,8]*
        conj(man.peps[i,j][11,-3,6,7,8])

    # "add east contribution"
    @tensor cnt[-1 -2 -3;-4] +=man.AC[West,i,j][-1,4,6,3]*
        man.fp1LR[North,i,j][3,5,7,9]*
        man.corner[NorthEast,i,j][9,1]*
        cecontr[1,10,11,2]*
        man.corner[SouthEast,i,j][2,-4]*
        man.peps[i,j][4,-2,10,5,8]*
        conj(man.peps[i,j][6,-3,11,7,8])

    # "vertical ham contribution"
    @tensor cnt[-1 -2 -3;-4]+=man.fp1RL[North,i-1,j][1,2,3,4]*
        man.AL[West,i-1,j][5,6,7,1]*
        man.AL[West,i,j][-1,8,9,5]*
        man.AR[East,i-1,j][4,10,11,12]*
        man.AR[East,i,j][12,13,14,-4]*
        man.peps[i-1,j][6,15,10,2,16]*
        conj(man.peps[i-1,j][7,17,11,3,18])*
        man.peps[i,j][8,-2,13,15,19]*
        conj(man.peps[i,j][9,-3,14,17,20])*
        nn[16,18,19,20]

    # "horleft contribution"
    @tensor cnt[-1 -2 -3;-4]+=man.fp1LR[North,i,j][1,4,6,2]*
        man.AC[East,i,j][2,3,5,-4]*
        man.corner[NorthWest,i,j][22,1]*
        man.AL[North,i,j-1][8,10,12,22]*
        man.fp1LR[West,i,j-1][15,9,11,8]*
        man.AR[South,i,j-1][7,13,14,15]*
        man.corner[SouthWest,i,j][-1,7]*
        man.peps[i,j-1][9,13,20,10,16]*
        conj(man.peps[i,j-1][11,14,18,12,17])*
        man.peps[i,j][20,-2,3,4,21]*
        conj(man.peps[i,j][18,-3,5,6,19])*
        nn[16,17,21,19]

    # "horright contribution"
    @tensor cnt[-1 -2 -3;-4]+=man.AC[West,i,j][-1,3,5,2]*
        man.fp1LR[North,i,j][2,4,6,1]*
        man.corner[NorthEast,i,j][1,22]*
        man.fp1LR[East,i,j+1][8,9,11,13]*
        man.AR[North,i,j+1][22,10,12,8]*
        man.AL[South,i,j+1][13,14,15,7]*
        man.corner[SouthEast,i,j][7,-4]*
        man.peps[i,j][3,-2,20,4,21]*
        conj(man.peps[i,j][5,-3,18,6,19])*
        man.peps[i,j+1][20,14,9,10,16]*
        conj(man.peps[i,j+1][18,15,11,12,17])*
        nn[21,19,16,17]

    return cnt
end

function north_nntchannel_impl(man::InfEnvManager,lines,nn::NN)
    (nr,nc) = size(man.peps);

    # sums[i] = all contributions above row 1; collumn i
    sums = map(1:nc) do j
        cnt = TensorMap(zeros,ComplexF64,space(man.AL[West,1,j],4)'*space(man.peps[1,j],North)'*space(man.peps[1,j],North),space(man.AL[East,1,j],1))

        for i in 1:nr
            cnt = north_nntchannel_loctransfer(cnt,man,lines,nn,i,j)
        end

        ecnt = north_inf_sum_RL(cnt,man,j)
    end

    # of course, we also want to know the sum of contributions above other unit cells ...
    ts = PeriodicArray{eltype(sums),2}(undef,nr,nc);
    for (i,j) in Iterators.product(1:nr,1:nc)
        if i == 1
            ts[i,j] = sums[j]
        else
            ts[i,j] = north_nntchannel_loctransfer(ts[i-1,j],man,lines,nn,i-1,j)
        end
    end

    return ts
end
