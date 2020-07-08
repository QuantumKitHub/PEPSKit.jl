mutable struct InfNNHamChannels{E<:InfEnvManager,B,C,O<:NN} <: Cache
    opperator :: O
    envm::E

    lines::B
    ts::C
end

#generate bogus data
function channels(envm::InfEnvManager,opperator::NN)
    lines = similar(envm.fp1);
    ts = similar(envm.fp1);

    pars = InfNNHamChannels(opperator,envm,lines,ts);

    return MPSKit.recalculate!(pars,envm.peps)
end

#recalculate everything
function MPSKit.recalculate!(prevenv::InfNNHamChannels,peps::InfPEPS)
    MPSKit.recalculate!(prevenv.envm,peps);

    prevenv.lines = PeriodicArray(fetch.(map(Dirs) do dir
        @Threads.spawn north_nncontr_impl(rotate_north(prevenv.envm,dir),rotate_north(prevenv.opperator,dir))
    end))

    prevenv.ts = PeriodicArray(fetch.(map(Dirs) do dir
        @Threads.spawn north_nntchannel_impl(  rotate_north(prevenv.envm,dir),
                                circshift(prevenv.lines,4-dir),
                                rotate_north(prevenv.opperator,dir));
    end))

    prevenv
end

# j == the collumn
function north_inf_sum_RL(v,man,j)
    downfp = fp1LR(man,South,0,j);
    upfp = fp1RL(man,North,1,j);
    @tensor v[-1 -2 -3; -4]-=downfp[4,2,3,1]*v[1,2,3,4]*upfp[-1,-2,-3,-4]

    (v,chist) = linsolve(v,v,GMRES()) do x
        y=crosstransfer(x,man.peps[:,j],AR(man,East,:,j),AL(man,West,:,j))
        @tensor y[-1 -2 -3; -4]-=downfp[4,2,3,1]*y[1,2,3,4]*upfp[-1,-2,-3,-4]

        y=x-y
    end
    chist.converged == 0 && @info "failed to converge _ RL $(chist.normres)"

    return v
end

function north_inf_sum_LR(v,man,j)
    downfp = fp1RL(man,South,0,j);
    upfp = fp1LR(man,North,1,j);
    @tensor v[-1 -2 -3; -4]-=downfp[4,2,3,1]*v[1,2,3,4]*upfp[-1,-2,-3,-4]

    (v,chist) = linsolve(v,v,GMRES()) do x
        y=crosstransfer(x,man.peps[:,j],AL(man,East,:,j),AR(man,West,:,j))
        @tensor y[-1 -2 -3; -4]-=downfp[4,2,3,1]*y[1,2,3,4]*(upfp)[-1,-2,-3,-4]

        y=x-y
    end
    chist.converged == 0 && @info "failed to converge _ LR $(chist.normres)"

    return v
end

function north_linecontr_local(cnt,man::InfEnvManager,nn::NN,i,j)
    cnt = crosstransfer(cnt,man.peps[i,j],AL(man,East,i,j),AR(man,West,i,j))

    cnt += hamtransfer(
        AR(man,West,i,j),
        AR(man,West,i-1,j),
        AL(man,East,i-1,j),
        AL(man,East,i,j),
        fp1LR(man,North,i-1,j),
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
        cnt = TensorMap(zeros,ComplexF64,   space(AL(man,West,1,j),4)'*
                                            space(man.peps[1,j],North)'*
                                            space(man.peps[1,j],North),
                                            space(AL(man,East,1,j),1))

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
    cnt = crosstransfer(cnt,man.peps[i,j],AR(man,East,i,j),AL(man,West,i,j))

    #collect west and east contributions from lines
    westcoords = rotate_north((i,j),size(man.peps),West);
    eastcoords = rotate_north((i,j),size(man.peps),East);
    cwcontr = lines[West][westcoords...];
    cecontr = lines[East][eastcoords...];
    tpeps = man.peps;

    @tensor cnt[-1 -2 -3;-4] +=
        corner(man,SouthWest,i,j)[-1,2]*
        cwcontr[2,10,11,1]*
        corner(man,NorthWest,i,j)[1,9]*
        fp1LR(man,North,i,j)[9,5,7,3]*
        AC(man,East,i,j)[3,4,6,-4]*
        tpeps[i,j][10,-2,4,5,8]*
        conj(tpeps[i,j][11,-3,6,7,8])

    # "add east contribution"
    @tensor cnt[-1 -2 -3;-4] +=
        AC(man,West,i,j)[-1,4,6,3]*
        fp1LR(man,North,i,j)[3,5,7,9]*
        corner(man,NorthEast,i,j)[9,1]*
        cecontr[1,10,11,2]*
        corner(man,SouthEast,i,j)[2,-4]*
        man.peps[i,j][4,-2,10,5,8]*
        conj(man.peps[i,j][6,-3,11,7,8])

    # "vertical ham contribution"
    @tensor cnt[-1 -2 -3;-4]+=
        fp1RL(man,North,i-1,j)[8,3,5,1]*
        AL(man,West,i-1,j)[20,6,7,8]*
        AR(man,East,i-1,j)[1,2,4,11]*
        AL(man,West,i,j)[-1,18,19,20]*
        AR(man,East,i,j)[11,12,15,-4]*
        man.peps[i-1,j][6,13,2,3,9]*
        conj(man.peps[i-1,j][7,16,4,5,10])*
        man.peps[i,j][18,-2,12,13,14]*
        conj(man.peps[i,j][19,-3,15,16,17])*
        nn[10,9,17,14]

    # "horleft contribution"
    @tensor cnt[-1 -2 -3;-4]+=
        fp1LR(man,North,i,j)[14,15,17,19]*
        AC(man,East,i,j)[19,20,21,-4]*
        corner(man,NorthWest,i,j)[1,14]*
        AL(man,North,i,j-1)[2,4,6,1]*
        fp1LR(man,West,i,j-1)[9,3,5,2]*
        AR(man,South,i,j-1)[22,7,8,9]*
        corner(man,SouthWest,i,j)[-1,22]*
        man.peps[i,j-1][3,7,12,4,10]*
        conj(man.peps[i,j-1][5,8,16,6,11])*
        man.peps[i,j][12,-2,20,15,13]*
        conj(man.peps[i,j][16,-3,21,17,18])*
        nn[11,10,18,13]

    # "horright contribution"
    @tensor cnt[-1 -2 -3;-4]+=
        AC(man,West,i,j)[-1,20,21,22]*
        fp1LR(man,North,i,j)[22,15,18,16]*
        corner(man,NorthEast,i,j)[16,2]*
        fp1LR(man,East,i,j+1)[3,4,6,8]*
        AR(man,North,i,j+1)[2,5,7,3]*
        AL(man,South,i,j+1)[8,9,10,1]*
        corner(man,SouthEast,i,j)[1,-4]*
        man.peps[i,j][20,-2,13,15,14]*
        conj(man.peps[i,j][21,-3,17,18,19])*
        man.peps[i,j+1][13,9,4,5,11]*
        conj(man.peps[i,j+1][17,10,6,7,12])*
        nn[19,14,12,11]

    return cnt
end

function north_nntchannel_impl(man::InfEnvManager,lines,nn::NN)
    (nr,nc) = size(man.peps);

    # sums[i] = all contributions above row 1; collumn i
    sums = map(1:nc) do j
        cnt = TensorMap(zeros,ComplexF64,space(AL(man,West,1,j),4)'*space(man.peps[1,j],North)'*space(man.peps[1,j],North),space(AL(man,East,1,j),1))

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
